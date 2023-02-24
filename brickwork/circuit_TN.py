

import itertools
import numpy as np
from quimb import *
import quimb.tensor as qtn
from quimb.utils import int2tup
from quimb.calc import check_dims_and_indices
import matplotlib.pyplot as plt
from opt_einsum import contract
import time


#%%

def get_arrays(string):
    if string=='match':
        return rand_match()
    if string=='2haar':
        return rand_uni(4)
    if string=='identity':
        return np.identity(4)
    if string =='markov':
        return markov(0.1)
    
def match(theta, phi):
    return np.array(
        [
            [np.cos(theta), 0, 0, -np.sin(theta)],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [np.sin(theta), 0, 0, np.cos(theta)],
        ]
    )
# def match(theta, phi):
#     return np.array(
#         [
#             [1, 0, 0, 0],
#             [0, 0, 1, 0],
#             [0, 1, 0, 0],
#             [0, 0, 0, 1],
#         ]
#     )


# def markov():
#     # need to check this currently a left matrix....
#     M = np.random.rand(4, 4)
#     for i in range(10):
#         M = M / np.sum(M, axis=0, keepdims=True)
#         M = M / np.sum(M, axis=1, keepdims=True)
#     return M


# def markov():
#     # need to check this currently a left matrix....
#     M = np.array(
#         [
#             [1, 0, 0, 0],
#             [0, 0, 1, 0],
#             [0, 1, 0, 0],
#             [0, 0, 0, 1],
#         ])
#     return M

# def markov(eps):
#     #make identity
#     M=np.identity(4)
#     #subtract small num
#     M=M-np.diag([eps]*4)
#     #add upper diag
#     M=M+np.diag([eps/2]*3,1)#+np.diag([eps/2]*3,-1)
#     M[0,3]=eps/2
#     M[3,0]=eps/2
#     return M
def markov(eps):
    # need to check this currently a left matrix....
    M = np.array(
        [
            [1-eps, 0, 0, eps],
            [0, 1-eps, eps, 0],
            [0, eps, 1-eps, 0],
            [eps, 0, 0, 1-eps],
        ])
    return M
def rand_markov():
    pass


def rand_match():
    arr = 2 * np.pi * np.random.rand(2)
    return match(arr[0]/2, arr[1]/2)
# def rand_match():
#     arr = 2 * np.pi * np.random.rand(2)
#     return match(2 * np.pi *0.4, 2 * np.pi *0.1)

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
        bc="periodic",
        meas_r: float=0.0,
        target=0,
        same=0,
        eps=0.1
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
        """ this need to be updated for different inits"""
        self.gate = gate

        # self.mps = MPS_rand_state(L=num_elems, bond_dim=50)
        self.mps=qtn.MPS_computational_state("".join(["0" for x in range(self.num_elems)]))

        # if init == "up":
        #     self.mps = computational_state(
        #         "".join(["0" for x in range(self.num_elems)]), qtype="dop", sparse=False
        #     )
        # elif "comp" in init:
        #     self.mps = computational_state(
        #         init.removeprefix("comp"), qtype="dop", sparse=False
        #     )
        # elif init == "upB":
        #     self.mps = computational_state(
        #         "".join(["0" for x in range(self.num_elems)]), qtype="ket", sparse=False
        #     )
        # elif init == "rand":
        #     if self.classical:
        #         self.mps = computational_state(''.join(f'{x}' for x in [np.random.choice([0, 1]) for i in range(num_elems)]),qtype="dop",sparse=False)
        #     else:
        #         self.mps = rand_product_state(self.num_elems,qtype="dop")
            
        # elif init == "randB":
        #     self.mps = rand_product_state(self.num_elems)

        self.dims = [2] * self.num_elems
        self.same = same


        self.markov = markov(self.eps)
        self.match = rand_match()

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
        i=0
        while self.step_num<self.num_steps:
            """
            force every other to be a meas
            """
            if i % 2 == 0:
                self.circ.append(self.gen_step("gates", self.architecture))
            if i % 2 == 1:
                self.circ.append(self.gen_step("meas", self.meas_r))
                self.step_num += 1
            i+=1


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
        if self.boundary_conditions == "periodic":
            if not eoo and not self.num_elems%2:
                pairs.append([self.num_elems-1,0])
        return pairs

    def gen_staircase(self):
        pairs = []
        if (self.step_num + 1) % self.num_elems == 0:
            self.step_num = self.step_num + 1
        pairs.append(
            [self.step_num % self.num_elems, (self.step_num + 1) % self.num_elems]
        )
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
        if "sep_mut" in rec:
            self.rec_sep_mut = self.sep_mut()
        if "tri_mut" in rec:
            self.rec_tri_mut = self.tripartite_mut()
                    

    def do_operation(self, op, ps):
        if op =='meas':
            for pairs in ps:
                self.measure(pairs)
        else:
            pairs = [tuple(i) for i in ps]
            gate=get_arrays(op)
            # print(gate)
            for pairs in ps:
                self.mps = qtn.gate_TN_1D(self.mps,gate,pairs,contract='swap+split')
                # self.mps.normalize()
            
            
        # Needs some work
        # if op == "bell":
        #     for pair in ps:
        #         # print(pair)
        #         had = ikron(hadamard(), [2] * self.num_elems, pair[0])
        #         step1 = had @ self.dop @ had.H
        #         cn = pkron(CNOT(), [2] * self.num_elems, pair)
        #         self.dop = cn @ step1 @ cn.H

        # elif op == "haar":
        #     for pair in ps:
        #         haar = ikron(rand_uni(4), [2] * self.num_elems, pair)
        #         self.dop = haar @ self.dop @ haar.H
        #         # self.dop.round(4)

        # elif op == "match":
        #     for pair in ps:
        #         if self.same:
        #             mar = self.match
        #         else:
        #             mar = rand_match()
        #         mat = qu(ikron(mar, [2] * self.num_elems, pair))
        #         self.dop = mat @ self.dop @ mat.H

        # elif op == "meas":
        #     for pair in ps:
        #         self.measure(pair)
                
    def measure(self, ind):
        # self.dop = normalize(self.dop)

        a, self.mps = self.mps.measure(ind)
    
    def ent(self, alpha=1):
        if alpha == 1:
            return self.mps.entropy(int(self.num_elems/2))

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
    hac = entropy_subsys(psi_abc, dims, sysa + sysc, **kws)
    hbc = entropy_subsys(psi_abc, dims, sysc + sysb, **kws)
    
    habc = entropy_subsys(psi_abc, dims, sysa + sysb+ sysc, **kws)

    ha = entropy_subsys(psi_abc, dims, sysa, **kws)
    hb = entropy_subsys(psi_abc, dims, sysb, **kws)
    hc = entropy_subsys(psi_abc, dims, sysb, **kws)


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