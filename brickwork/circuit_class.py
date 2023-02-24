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
        gate="markov",
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
        self.mps=qu(qtn.MPS_computational_state("".join(["0" for x in range(self.num_elems)])).to_dense(),sparse=True,qtype="ket")

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
            # for pairs in ps:
            self.measure(ps)
        else:
            pairs = [tuple(i) for i in ps]
            
            #same gate
            gate=get_arrays(op)
            # print(gate)
            
            ops=[qu(gate,sparse=True) for i in ps]
            A = pkron(kron(*ops),dims=[2]*self.num_elems,inds=np.array(ps).flatten())
            self.mps = A@self.mps
                
    def measure(self, inds):
        if len(inds)==0:
            pass
        else:
            ops = [pauli("Z") for i in inds]
            A= pkron(kron(*ops),self.dims,inds).real
            p=qu(self.mps,sparse=False,qtype='ket').real
            
            el, ev = eigh(A)
            js = np.arange(el.size)
            pj = p.flatten()
            # then choose one
            j = np.random.choice(js, p=pj)
            eigenvalue = el[j]

            # now combine whole eigenspace
            P = projector((el, ev), eigenvalue=eigenvalue, tol=1e-8).real
            # P = np.identity(2**self.num_elems)-P

            total_prob = np.sum(P@p).real
            # print("@@@tot_prob : ",total_prob)
            if total_prob == 0:
                return
            arr = qu((P@p/total_prob),sparse=False,qtype='ket')
            self.old_mps = self.mps
            self.mps = arr
        
    def ent(self, alpha=1):
        if alpha == 1:
            return entropy_subsys(qu(self.mps,sparse=False),dims=[2]*self.num_elems,sysa=[i for i in range(int(self.num_elems/2))])

    def sep_mut(self):
        arr = [
            mutinf_subsys(
                qu(self.mps,sparse=False),
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
                qu(self.mps,sparse=False),
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
        return tri_mutinf_subsys(qu(self.mps,sparse=False),self.dims,arr[0],arr[1],arr[2])
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