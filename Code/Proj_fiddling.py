"""
TODO:
    -better way to visualize
    -markov
        check tailored markov matricies
    -more commenting
    -can possibly remove quimb but will need to do some optimizations
"""

import itertools
import numpy as np
from quimb import *
import matplotlib.pyplot as plt
from opt_einsum import contract
import time


#%%


def match(theta, phi):
    return np.array(
        [
            [np.cos(theta), 0, 0, -np.sin(theta)],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [np.sin(theta), 0, 0, np.cos(theta)],
        ]
    )


def markov():
    # need to check this currently a left matrix....
    M = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ])
    return M

def markov(eps):
    #make identity
    M=np.identity(4)
    #subtract small num
    M=M-np.diag([eps]*4)
    #add upper diag
    M=M+np.diag([eps/2]*3,1)#+np.diag([eps/2]*3,-1)
    M[0,3]=eps/2
    M[3,0]=eps/2
    return M
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
        gate="bell",
        init="up",
        architecture="brick",
        meas_r=0.0,
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
        self.meas_r = meas_r
        self.classical = 0
        self.eps = eps
        """ this need to be updated for different inits"""
        self.gate = gate
        if self.gate == "markov":
            self.classical = 1
            
        if init == "up":
            self.dop = computational_state(
                "".join(["0" for x in range(self.num_elems)]), qtype="dop", sparse=False
            )
        elif "comp" in init:
            self.dop = computational_state(
                init.removeprefix("comp"), qtype="dop", sparse=False
            )
        elif init == "upB":
            self.dop = computational_state(
                "".join(["0" for x in range(self.num_elems)]), qtype="ket", sparse=False
            )
        elif init == "rand":
            if self.classical:
                self.dop = computational_state(''.join(f'{x}' for x in [np.random.choice([0, 1]) for i in range(num_elems)]),qtype="dop",sparse=False)
            else:
                self.dop = rand_product_state(self.num_elems,qtype="dop")
            
        elif init == "randB":
            self.dop = rand_product_state(self.num_elems)

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
        self.rec_mut_inf = [self.mutinfo(self.target)]
        self.rec_bip = []
        self.rec_ent = [self.ent()]
        self.rec_ren = [self.ent(alpha=2)]
        self.rec_half = []

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
        for i in range(self.num_steps):
            """
            force every other to be a meas
            """
            if i % 2 == 0:
                self.circ.append(self.gen_step("gates", self.architecture))
            if i % 2 == 1:
                self.circ.append(self.gen_step("meas", self.meas_r))

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
                # print(self.step_num)
                pairs = self.gen_pairs((self.step_num + 1) % 2)
                step_dct.update({self.gate: pairs})
                # self.step_num += 1
            if architecture == "stair":
                pairs = self.gen_staircase()
                step_dct.update({self.gate: pairs})

        elif operation == "meas":
            if type(architecture) == float:
                pairs = self.gen_rand_meas()
                step_dct.update({"meas": pairs})
                self.step_num += 1
            else:
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
    def do_step(self, num=None, rec=None):
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
                    if "mut" in rec:
                        self.rec_mut_inf.append(self.mutinfo(self.target))
                    if "bip" in rec:
                        self.rec_bip.append(self.bipent())
                    if "state" in rec:
                        self.plot_state()
                    if "von" in rec:
                        self.rec_ent.append(self.ent())
                    if "reny" in rec:
                        self.rec_ren.append(self.ent(alpha = 2))
                    if "halfMI" in rec:
                        self.rec_half.append(self.bipent(x=int(self.num_elems/2)))
                        

    def do_operation(self, op, ps):

        # Needs some work
        if op == "bell":
            for pair in ps:
                # print(pair)
                had = ikron(hadamard(), [2] * self.num_elems, pair[0])
                step1 = had @ self.dop @ had.H
                cn = pkron(CNOT(), [2] * self.num_elems, pair)
                self.dop = cn @ step1 @ cn.H

        elif op == "haar":
            for pair in ps:
                haar = ikron(rand_uni(4), [2] * self.num_elems, pair)
                self.dop = haar @ self.dop @ haar.H
                # self.dop.round(4)

        elif op == "match":
            for pair in ps:
                if self.same:
                    mar = self.match
                else:
                    mar = rand_match()
                mat = qu(ikron(mar, [2] * self.num_elems, pair))
                self.dop = mat @ self.dop @ mat.H

        elif op == "markov":
            for pair in ps:
                if self.same:
                    mar = self.markov
                else:
                    mar = markov(self.eps)
                mat = qu(ikron(mar, [2] * self.num_elems, pair))
                self.dop = self.lmc(mat)
                # self.dop = D / trace(D)

        elif op == "meas":
            for pair in ps:
                self.measure(pair)

    def lmc(self, mat):
        """
        slow
        """
        D = np.zeros_like(self.dop)
        sqrtmat = np.sqrt(mat)
        for i in np.ndindex(np.shape(mat)):
            mk = np.zeros_like(mat)
            mk[i] = sqrtmat[i]
            D += mk @ self.dop @ qu(mk).H
        return D / trace(D)

    def measure(self, ind):
        # self.dop = normalize(self.dop)

        if not self.classical:
            a, self.dop = measure(
                np.array(self.dop), ikron(pauli("Z"), [2] * self.num_elems, ind)
            )
        else:
            a, self.dop = self.class_measure(
                np.array(self.dop), ikron(pauli("Z"), [2] * self.num_elems, ind)
            )

    def class_measure(self, p, A, eigenvalue=None, tol=1e-5):
        """
        adapted from quimb package
        """
        el, ev = eigh(A)
        js = np.arange(el.size)

        pj = contract("jk,kl,lj->j", np.sqrt(ev.H), p, np.sqrt(ev)).real
        # then choose one

        j = np.random.choice(js, p=pj)

        eigenvalue = el[j]

        # now combine whole eigenspace
        P = projector((el, ev), eigenvalue=eigenvalue, tol=tol)
        total_prob = np.sum(pj[abs(el - eigenvalue) < tol])

        # now collapse the state
        if isvec(p):
            p_after = P @ (p / total_prob ** 0.5)
        else:
            p_after = (P @ p @ P.H) / total_prob

        # print(trace(p_after))
        return eigenvalue, p_after
    
    def ent(self, alpha=1):
        if alpha == 1:
            return entropy(self.dop)
        else:
            a = np.asarray(self.dop)
            if np.ndim(a) == 1:
                evals = a
            else:
                evals = eigvalsh(a)
            
            evals = evals[evals > 0.0]
            return 1/(1-alpha)*np.log2(sum(evals**alpha))
    
    def mutinfo(self, target=0):
        # this is mem bad
        arr = [
            ptr(self.dop, dims=self.dims, keep=[target, x]).round(4)
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
                    self.dop,
                    dims=self.dims,
                    sysa=list(range(x)),
                    sysb=list(range(x, self.num_elems)),
                )
                for x in range(self.num_elems)
            ]
        else:
            arr = mutinf_subsys(
                    self.dop,
                    dims=self.dims,
                    sysa=list(range(x)),
                    sysb=list(range(x, self.num_elems)),
                )
                
        return arr
        ###############################

    def print_state(self):
        for i in range(len(self.dims)):
            print(partial_trace(self.dop, self.dims, [i]))

    def plot_state(self):
        states = [ptr(self.dop, self.dims, x)[0][0] for x in range(self.num_elems)]
        plt.plot(states)


#%%
data_mark = []
start=time.time()
for i in np.linspace(0,0.5,3):
    arr_von=[]
    arr_mut=[]
    arr_ren=[]
    interval =np.linspace(0.,1,3) 
    Sites=5
    num_samples = 2
    eps=i
    gate="markov"
    

    for i in interval:
        print(i)
        numstep = 30
    
        
        circ = circuit(Sites, numstep, init="up", meas_r=float(i), gate=gate, architecture="brick")
        circ.do_step(num=numstep, rec="von reny halfMI")
        data_von = circ.rec_ent
        data_mut= circ.rec_half
        data_ren= circ.rec_ren
        
        for j in range(1,num_samples):
            circ = circuit(5, numstep, init="up", meas_r=float(i), gate=gate, architecture="brick")
            circ.do_step(num=numstep, rec="von reny halfMI")
            data_von = np.average(np.array([data_von, circ.rec_ent]), axis=0,weights=[j,1] )
            data_mut = np.average(np.array([data_mut, circ.rec_half]), axis=0,weights=[j,1] )
            data_ren = np.average(np.array([data_ren, circ.rec_ren]), axis=0,weights=[j,1] )
    
        arr_von.append(data_von)
        arr_mut.append(data_mut)
        arr_ren.append(data_ren)
    data_mark.append([arr_von,arr_mut,arr_ren])
end=time.time()
#%%
fig, ax = plt.subplots()
test_arr=test_arr = [i[1] for i in data_mark]
for arr in test_arr:
    ax.plot(np.transpose([[arr[i][j] for j in range(0,np.shape(arr)[1],2)] for i in range(np.shape(arr)[0])]))
# ax.plot(np.transpose([[arr_mutq[i][j] for j in range(0,np.shape(arr_mutq)[1],2)] for i in range(np.shape(arr_mutq)[0])]))

    ax.legend(title='meas_r',labels=np.round(interval,3))
plt.title(f"Mut_info, Gate:{gate}, Sites:{Sites}, Samples:{num_samples}, Time={np.round(end-start,3)}")

plt.show()
#%%
start = time.time()
irand = np.random.rand(32,32)
for j in range(30):
    irand = irand @ np.random.rand(32,32)
t_total = time.time() - start
#%%
"""
My own brickwork code because I'm confused
"""

def cc(gate):

    return np.transpose(np.matrix.conjugate(gate))

def kron_l(l):
    
    m = l[0]
    for j in range(len(l)-1):
        m = np.kron(m, l[j+1])
    return m

def run_Acirc(q, j, rho, meas):
    
    if j%2 == 0:
        u = rand_uni(4)
        for i in range(q//2 - 1):
            u = np.kron(u, rand_uni(4))
        if q%2 == 1:
            u = np.kron(u, np.identity(2))
    else:
        u = np.identity(2)
        for i in range((q-1)//2):
            u = np.kron(u, rand_uni(4))
        if q%2 == 0:
            u = np.kron(u, np.identity(2))
            
    
    rho = cc(u) @ rho @ u 
    m_c = 0
    
    """
    do my thing here
    """
    
    for i in range(q):
        mr = np.random.rand()
        if mr < meas:
            proj_1 = [np.identity(2) for i in range(qbits)]
            proj_2 = [np.identity(2) for i in range(qbits)]
            proj_1[i] = proj_1[i] @ np.array([[1,0],[0,0]])
            proj_2[i] = proj_2[i] @ np.array([[0,0],[0,1]])
            pm1 = kron_l(proj_1)
            pm2 = kron_l(proj_2)
            mprob1 = np.real(np.trace(rho @ pm1))
            mprob2 = np.real(np.trace(rho @ pm2))
            if np.random.rand() < mprob1:
                rho = pm1 @ rho @ pm1 / (mprob1)
            else:
                rho = pm2 @ rho @ pm2 / (mprob2)
            m_c = 1
        
    return rho, u, m_c

qbits = 5; layers=30
rho = np.zeros((2**qbits, 2**qbits), dtype='complex'); rho[0,0] = 1
mc = 0
for j in range(layers):
    rho, u, m_c = run_Acirc(qbits, j, rho, meas = 0.1)
    mc += m_c