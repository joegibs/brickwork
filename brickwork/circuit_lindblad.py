# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 11:41:45 2023

@author: jogib
"""



import itertools
import numpy as np
from quimb import *
import quimb.tensor as qtn
from quimb.utils import int2tup
from quimb.calc import check_dims_and_indices
import matplotlib.pyplot as plt
from opt_einsum import contract
from copy import deepcopy
import time
from autoray import do, dag, reshape, conj, get_dtype_name, transpose


#%%

def get_arrays(string,eps=0.3):
    
    if string=='2haar':
        return rand_uni(16)
    if string=='2haar#':
        g=rand_uni(4)#CNOT()@kron(hadamard(),np.identity(2))
        return pkron(g,dims=[2]*4,inds=[0,2])@pkron(g.T,[2]*4,inds=[1,3])
    if string=='identity':
        return np.identity(16)
    if string == 'rand_phase':
        # print("hey")
        return dissipation()
    
# def dissipation():
#     gamma = 5*10E-3
#     # return np.array(
#     #     [
#     #         [1, np.sqrt(gamma)],
#     #         [0, np.sqrt(1-gamma)],
#     #     ]
#     # )
#     return 0.5*(pauli("X")- 1j* pauli('Y'))
def rand_phase():
    return phase_gate(np.pi*np.random.rand())

def mps_to_mpo(a):
    return a.partial_trace(range(0,a.L))
def to_lindblad_space(tst):
    arr = []
    for i,t in enumerate(tst):
        # print(t)
        a = t.fuse({f'k{i}':[f'k{i}',f'b{i}']})
        arr.append(a)
    TN = qtn.TensorNetwork(arr)
    fin = qtn.MatrixProductState(TN.arrays)
    fin.normalize()
    return fin
def from_lindblad_space(fin):
    rev = qtn.TensorNetwork([i for i in fin])
    arr=[]
    for i,t in enumerate(rev):
        a = t.unfuse({f'k{i}':[f'k{i}',f'b{i}']},{f'k{i}':(2,2)})
        arr.append(a)
    TN = qtn.TensorNetwork(arr)
    ret = qtn.MatrixProductOperator(TN.arrays)
    return ret/ret.trace()
#%%
class circuit:
    """
    contains the densop and mixing things
    Returns
    -------
    None.
    """

    def __init__(
        self,
        num_elems,
        num_steps,
        gate="2haar",
        init="up",
        architecture="brick",
        bc="",
        meas_r: float=0.0,
        target=0,
        same=0,
        eps=0.1,
        phase=False,
        gate_holes=None
    ):
        """
        num_elems: number of elements in the chain
        gate: type of gate used
            -"bell" hadamard then cnot to make a bell state
            -"haar" haar random unitary operators
            -"match"
            -"markov"
             
        init: initialization of qubits
            -"up" all sites in u state
            -"comp" comp0101 comp followed by binary string of state
            -"rand" random init
        
        architecture: arrangement of gates
            -"brick": alternating pairs
            -"staircase": each step only has one gate
        """

        self.num_elems = num_elems
        self.architecture = architecture
        self.boundary_conditions=bc
        self.meas_r = float(meas_r)
        self.eps = eps
        self.phase=phase
        """ this need to be updated for different inits"""
        self.gate = gate

        # self.mps = MPS_rand_state(L=num_elems, bond_dim=50)
        if bc == "periodic":
            cyclic = True
        else:
            cyclic = False
        self.mps_init=qtn.MPS_computational_state("".join(["0" for x in range(self.num_elems)]),cyclic=False)

        self.mps = to_lindblad_space(mps_to_mpo(self.mps_init))

        self.dims = [2] * self.num_elems
        self.same = same

        self.num_steps = num_steps
        self.step_num = 0
        self.pairs = []
        self.target = 0

        self.gen_circ()
        self.step_num = 0
        self.step_tracker = 0

        self.target = target
        # self.rec_mut_inf = [self.mutinfo(self.target)]
        # self.rec_bip = []
        self.rec_ent = [self.ent()]
        self.rec_neg = [self.log_neg()]

        self.rec_sep_mut=[]
        self.rec_tri_mut=[]
        # self.rec_ren = [self.ent(alpha=2)]
        # self.rec_half = []

    ##################   Building the circuit       ###################

    def gen_circ(self):
        """
        build array of gates
        needs to be  dic of gate type:list of what it acts on
        number of steps
        each step is operation 
        """
        self.circ = []
        # call some generating thing
        while self.step_num<self.num_steps:
            """
            force every other to be a meas
            """
            self.circ.append(self.gen_step("gates", self.architecture))
            self.circ.append(self.gen_step("meas", self.meas_r))
            self.step_num += 1


    def gen_step(self, operation: str, architecture: str):
        """
        needs to generate a single step in circ
        operation: gates or measure
        architecture: depends on the operation but can be brick staircase for gates
                       or none all or rand for measure
        """
        step_dct = {}

        if operation == "gates":
            if architecture == "brick":
                pairs = self.gen_pairs((self.step_num + 1) % 2)
                step_dct.update({self.gate: pairs})
                
                
            if architecture == "stair":
                pairs = self.gen_staircase()
                step_dct.update({self.gate: pairs})
            if self.phase:
                step_dct.update({"rand_phase":[i for i in range(self.num_elems)]})
                
        elif operation == "meas":
            if type(architecture) == float:
                pairs = self.gen_rand_meas()
                step_dct.update({"meas": pairs})
            else:
                print("HET")
                pass
        return step_dct

    def gen_pairs(self, eoo):
        pairs = []
        # some control over where indexing starts
        if eoo:
            i = 0
        else:
            i = 1
        # get pairs
        while i + 2 <= self.num_elems:
            pairs.append([i, i + 1])
            i = i + 2
        return pairs

    def gen_rand_meas(self):
        tf = list(np.random.random_sample(self.num_elems) < self.meas_r)
        return [i for i in range(len(tf)) if tf[i] == 1]

    ############     running the circuit       ######################
    def do_step(self, num=None, rec=''):
        """
        tells the circuit to run and record different measures
        Parameters
        ----------
        num : int, optional
            number of steps to preform, defaults to evaluating all steps
        rec : string
            data to record can be combined
            -mut : mutual information
            -bip : bipartite entropy
            -state : plots partial trace value of the 1 state
        """
        # do things
        if num == None:
            for i in self.circ:
                for j in list(i.keys()):
                    # print(j,i[j])
                    self.do_operation(j, i[j])
                self.step_num = self.step_num + 1

                # record things
                if "mut" in rec:
                    self.rec_mut_inf.append(self.mutinfo(self.target))
                if "bip" in rec:
                    self.rec_bip.append(self.bipent())
                if "von" in rec:
                    self.rec_ent.append(self.ent())
                if "neg" in rec:
                    self.rec_neg.append(self.log_neg())
                if "sharp" in rec:
                    self.rec_ent.append(self.ent_sharp())
        else:
            self.step_tracker += num
            for st, i in enumerate(self.circ):
                # print(st)
                if st > self.step_tracker:
                    pass
                for j in list(i.keys()):
                    # print(j,i[j])
                    self.do_operation(j, i[j])
                self.step_num = self.step_num + 1

                    # record things
                if "meas" in i.keys():
                    # print("rec")
                    if "von" in rec:
                        self.rec_ent.append(self.ent())
                    if "sharp" in rec:
                        self.rec_ent.append(self.ent_sharp())
                    if "neg" in rec:
                        self.rec_neg.append(self.log_neg())
        if "sep_mut" in rec:
            self.rec_sep_mut = self.sep_mut()
        if "tri_mut" in rec:
            self.rec_tri_mut = self.tripartite_mut()
                    

    def do_operation(self, op, ps):
        if op =='meas':
            for pairs in ps:
                # print(type(pairs))
                # arr1= self.mps.partial_trace([pairs]).arrays[0]
                self.measure(pairs)
                # arr2= self.mps.partial_trace([pairs]).arrays[0]
                # print(arr1.round(3),arr2.round(3))

        else:
            # pairs = [tuple(i) for i in ps]
            # print(gate)
            # print(ps)
            self.old_mps = deepcopy(self.mps)
            # print(self.step_num)

            for pairs in ps:
                gate=get_arrays(op)
                # print(self.mps.norm())
                # print(gate)
                # print(gate, tuple(pairs))
                # if pairs[1] ==0:
                #     # print(pairs)
                #     self.mps = qtn.gate_TN_1D(self.mps,gate,pairs,contract='swap+split')
                # else:
                self.mps.gate(gate,pairs,contract='swap+split',inplace=True)
                # self.mps.normalize()
            
                
    def measure(self, ind):
        # meas_mps = from_lindblad_space(self.mps)
        # self.dop = normalize(self.dop)
        # print(ind)
        try:
            self.mps.measure(ind,inplace=True)
        # a, self.mps = cyc_measure(self.mps,ind)
        except:
            print("FARTS")
            pass
    
    def ent(self, alpha=1):
        if not self.mps.cyclic:
            return self.mps.entropy(int(self.num_elems/2))
        else:
            return cyclic_ent(self.mps,self.dims,[x for x in range(int(self.num_elems/2))], sysb=None)
    def ent_sharp(self):
        tst = from_lindblad_space(self.mps)
        traced_tst=ptr(tst.to_dense(),self.dims,[x for x in range(int(self.num_elems/2))])
        return entropy(traced_tst)
     
    def sep_mut(self):
        arr = [
            mutinf_subsys(
                self.mps.to_dense(),
                dims=self.dims,
                sysa=[1],
                sysb=[x]
            )
            for x in range(2,self.num_elems)
        ]
        return arr
    def log_neg(self):
        return self.mps.logneg_subsys([i for i in range(int(self.num_elems/2))],[i for i in range(int(self.num_elems/2),self.num_elems)])
    def polar_mut(self):
        arr = [
            mutinf_subsys(
                self.mps.to_dense(),
                dims=self.dims,
                sysa=[1],
                sysb=[x]
            )
            for x in range(2,self.num_elems)
        ]
        return arr
    
    def mutinfo(self, target=0):
        # this is mem bad
        arr = [
            ptr(self.mps, dims=self.dims, keep=[target, x]).round(4)
            for x in range(np.size(self.dims))
        ]
        mi = [
            mutinf(arr[x] if x != target else purify(arr[x]))
            for x in range(np.size(arr))
        ]
        return mi

    def bipent(self,x=None):
        if x is None:
            arr = [
                mutinf_subsys(
                    self.mps,
                    dims=self.dims,
                    sysa=list(range(x)),
                    sysb=list(range(x, self.num_elems)),
                )
                for x in range(self.num_elems)
            ]
        else:
            arr = mutinf_subsys(
                    self.mps,
                    dims=self.dims,
                    sysa=list(range(x)),
                    sysb=list(range(x, self.num_elems)),
                )
                
        return arr
    
    def tripartite_mut(self):
        elems=np.arange(0,self.num_elems)
        arr= np.array_split(elems,4)
        return tri_mutinf_subsys(self.mps.to_dense(),self.dims,arr[0],arr[1],arr[2])
        ###############################

    def print_state(self):
        for i in range(len(self.dims)):
            print(partial_trace(self.mps, self.dims, [i]))

    def plot_state(self):
        states = [ptr(self.mps, self.dims, x)[0][0] for x in range(self.num_elems)]
        plt.plot(states)
#%%
def cyc_measure(
        mps,
        site,
        remove=False,
        outcome=None,
        renorm=True,
        cur_orthog=None,
        get=None,
        inplace=False,
    ):
        r"""Measure this MPS at ``site``, including projecting the state.
        Optionally remove the site afterwards, yielding an MPS with one less
        site. In either case the orthogonality center of the returned MPS is
        ``min(site, new_L - 1)``.
        Parameters
        ----------
        site : int
            The site to measure.
        remove : bool, optional
            Whether to remove the site completely after projecting the
            measurement. If ``True``, sites greater than ``site`` will be
            retagged and reindex one down, and the MPS will have one less site.
            E.g::
                0-1-2-3-4-5-6
                       / / /  - measure and remove site 3
                0-1-2-4-5-6
                              - reindex sites (4, 5, 6) to (3, 4, 5)
                0-1-2-3-4-5
        outcome : None or int, optional
            Specify the desired outcome of the measurement. If ``None``, it
            will be randomly sampled according to the local density matrix.
        renorm : bool, optional
            Whether to renormalize the state post measurement.
        cur_orthog : None or int, optional
            If you already know the orthogonality center, you can supply it
            here for efficiencies sake.
        get : {None, 'outcome'}, optional
            If ``'outcome'``, simply return the outcome, and don't perform any
            projection.
        inplace : bool, optional
            Whether to perform the measurement in place or not.
        Returns
        -------
        outcome : int
            The measurement outcome, drawn from ``range(phys_dim)``.
        psi : MatrixProductState
            The measured state, if ``get != 'outcome'``.
        """

        tn = mps if inplace else mps.copy()
        L = tn.L
        d = mps.phys_dim(site)

        # make sure MPS is canonicalized
        # if cur_orthog is not None:
        #     tn.shift_orthogonality_center(cur_orthog, site)
        # else:
        # locs = [site,np.mod(site+1,mps.num_tensors)]
        # tn.canonize_cyclic([site,site+1])

        # local tensor and physical dim
        t = tn[site]
        ind = tn.site_ind(site)

        # diagonal of reduced density matrix = probs
        tii = t.contract(t.H, output_inds=(ind,))
        p = do('real', tii.data)
        # print(p,sum(p))
        p=p/sum(p)
        if outcome is None:
            # sample an outcome
            outcome = do('random.choice', do('arange', d, like=p), p=p)

        if get == 'outcome':
            return outcome

        # project the outcome and renormalize
        t.isel_({ind: outcome})

        if renorm:
            t.modify(data=t.data / p[outcome]**0.5)

        if remove:
            # contract the projected tensor into neighbor
            if site == L - 1:
                tn ^= slice(site - 1, site + 1)
            else:
                tn ^= slice(site, site + 2)

            # adjust structure for one less spin
            for i in range(site + 1, L):
                tn[i].reindex_({tn.site_ind(i): tn.site_ind(i - 1)})
                tn[i].retag_({tn.site_tag(i): tn.site_tag(i - 1)})
            tn._L = L - 1
        else:
            # simply re-expand tensor dimensions (with zeros)
            t.new_ind(ind, size=d, axis=-1)

        return outcome, tn


def cyclic_ent(ogmps,dims,sysa,sysb=None , **kws):
    
    num_elems = len(dims)
    if sysb is None:
        sysb = sysb = [i for i in range(num_elems) if i not in sysa]
        
    mps = ogmps.copy(deep=True)
    mpsH = mps.H
    # mpsH.retag_({'ket': ''})
    # this automatically reindexes the TN
    mpsH.site_ind_id = 'b{}'

    # define two subsystems
    sysa = range(0, int(num_elems/2))
    sysb = [i for i in range(num_elems) if i not in sysa]

    # join indices for sysb only
    mps.reindex_sites('dummy_ptr{}', sysb, inplace=True)
    mpsH.reindex_sites('dummy_ptr{}', sysb, inplace=True)

    rho_ab = (mpsH | mps)
    rho_ab
    right_ix = [f'b{i}' for i in sysa]
    left_ix = [f'k{i}' for i in sysa]

    rho_ab_lo = rho_ab.aslinearoperator(left_ix, right_ix)
    S_a = - approx_spectral_function(rho_ab_lo, f=xlogx, R=10)
    return S_a

def tri_mutinf_subsys(
    psi_abc, dims, sysa, sysb, sysc, approx_thresh=2**13, **approx_opts
):
    """Calculate the mutual information of two subsystems of a pure state,
    possibly using an approximate lanczos method for large subsytems.
    Parameters
    ----------
    psi_abc : vector
        Tri-partite pure state.
    dims : sequence of int
        The sub dimensions of the state.
    sysa : sequence of int
        The index(es) of the subsystem(s) to consider part of 'A'.
    sysb : sequence of int
        The index(es) of the subsystem(s) to consider part of 'B'.
    approx_thresh : int, optional
        The size of subsystem at which to switch to the approximate lanczos
        method. Set to ``None`` to never use the approximation.
    approx_opts
        Supplied to :func:`entropy_subsys_approx`, if used.
    Returns
    -------
    float
        The mutual information.
    See Also
    --------
    mutinf, entropy_subsys, entropy_subsys_approx, logneg_subsys
    """
    sysa, sysb, sysc = int2tup(sysa), int2tup(sysb), int2tup(sysc)

    check_dims_and_indices(dims, sysa, sysb, sysc)

    sz_a = prod(d for i, d in enumerate(dims) if i in sysa)
    sz_b = prod(d for i, d in enumerate(dims) if i in sysb)
    sz_c = prod(d for i, d in enumerate(dims) if i in sysc)

    sz_d = prod(dims) // (sz_a * sz_b * sz_b)

    kws = {"approx_thresh": approx_thresh, **approx_opts}

    #possible bug case
    # if sz_d == 1:
    #     hab = 0.0
    #     ha = hb = entropy_subsys(psi_abc, dims, sysa, **kws)
    # else:
    hab = entropy_subsys(psi_abc, dims, sysa + sysb, **kws)
    hac = entropy_subsys(psi_abc, dims,sysa + sysc, **kws)
    hbc = entropy_subsys(psi_abc, dims,sysc + sysb, **kws)
    
    habc = entropy_subsys(psi_abc, dims,sysa + sysb+ sysc, **kws)

    ha = entropy_subsys(psi_abc, dims,sysa, **kws)
    hb = entropy_subsys(psi_abc, dims,sysb, **kws)
    hc = entropy_subsys(psi_abc, dims,sysb, **kws)


    return hb + ha - hab - hac - hbc + habc

# #%%

# arr_vonq=[]

# interval =[0]#np.linspace(0.,0.3,4) 
# Sites=5
# num_samples = 100
# eps=0.1
# gate="2haar"

# start=time.time()
# for i in interval:
#     print(i)
#     numstep = 50

    
#     circ = circuit(Sites, numstep, meas_r=float(i), gate=gate, architecture="brick")
#     circ.do_step(num=numstep, rec="von")
#     data_von = circ.rec_ent
    
#     for j in range(1,num_samples):
#         circ = circuit(5, numstep, meas_r=float(i), gate=gate, architecture="brick")
#         circ.do_step(num=numstep, rec="von")
#         data_von = np.average(np.array([data_von, circ.rec_ent]), axis=0,weights=[j,1] )

#     arr_vonq.append(data_von)
# end=time.time()


#%%
# fig, ax = plt.subplots()
# ax.plot(np.transpose([[arr_vonq[i][j] for j in range(0,np.shape(arr_vonq)[1],2)] for i in range(np.shape(arr_vonq)[0])]))

# ax.legend(title='meas_r',labels=np.round(interval,3))
# plt.title(f"Mut_info, Gate:{gate}, Sites:{Sites}, Samples:{num_samples}, Time={np.round(end-start,3)}")
# # ax.set_yscale('log')

# plt.show()
# #%%
# p = MPS_rand_state(L=20, bond_dim=50)
# p.show()  # 1D tensor networks also have a ascii ``show`` method
# p.left_canonize()
# p.show()

# #quimb.tensor.tensor_1d.gate_TN_1D(tn, G, where, contract=False, tags=None, propagate_tags='sites', info=None, inplace=False, cur_orthog=None, **compress_opts)


# #read the circuit
# '''
# X
# already done in do_step
# '''
# #get operations
# """
# X
# need to add aditional cases but lets go with haar for now
# """
# #get pairs
# """
# X
# done need to adjust a bit but ok
# """
# #do operation
# """
# multiple gates at once is being weird just do ikron for now

# p=gate_TN_1D(p,qu.hadamard()&qu.hadamard()&qu.hadamard(),[1,2,3],contract=True)
# """

# def get_operators(step):
#     key = step.keys[0]
#     if key = 'haar':
#         return rand_uni(4)
#     if key = 'H':
        
# def get_operators(circ):
#     circ.circ
    
# tensor_1

# #%%
# dims = [2] * 4  # overall space of 10 qubits
# X = pauli('X')
# Z= pauli('Z')
# gates = [X&Z for x in range(2)]
# # IIIXXIIIII = ikron(X, dims, inds=[3, 4])
# iiii = pkron(gates,dims,inds=[(0,1),(1,2)])