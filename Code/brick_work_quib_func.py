"""
TODO:
    -better way to visualize
    -markov
        sumi Mij=1
        probs dont sum to 1 obvs need to make not densop?
        dim mismatch if ket, neg probs if skrew it
    -comments

"""

import itertools
# from operator import add
# from telnetlib import EL
# from tkinter import ARC
import numpy as np
from quimb import *
import matplotlib.pyplot as plt
from opt_einsum import contract


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
    M = np.random.rand(4, 4)
    for i in range(10):
        M = M / np.sum(M, axis=0, keepdims=True)
        M = M / np.sum(M, axis=1, keepdims=True)
    return M


def rand_markov():
    pass


def rand_match():
    arr = 2 * np.pi * np.random.rand(2)
    return match(arr[0], arr[1])


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
        same = 0,
    ):
        """
        num_elems: number of elements in the chain
        gate: type of gate used
            -"bell" hadamard then cnot to make a bell state
            -"haar" haar random unitary operators
        init: initialization of qubits
        
        architecture: arrangement of gates
            -"brick": alternating pairs
            -"staircase": each step only has one gate
        """

        self.num_elems = num_elems
        self.architecture = architecture
        self.meas_r = meas_r
        self.classical = 0
        """ this need to be updated for different inits"""

        if init == "up":
            self.dop = computational_state(
                "".join(["0" for x in range(self.num_elems)]), qtype="dop", sparse=False
            )
        elif init == "upB":
            self.dop = computational_state(
                "".join(["0" for x in range(self.num_elems)]), qtype="ket", sparse=False
            )
        elif init == "rand":
            self.dop = rand_product_state(self.num_elems)
            self.dop = qu(self.dop, qtype="dop")
        elif init == "randB":
            self.dop = rand_product_state(self.num_elems)

        self.dims = [2] * self.num_elems
        self.gate = gate
        self.same = same
        
        if self.gate == "markov":
            self.classical = 1
        self.markov = markov()

        self.num_steps = num_steps
        self.step_num = 0
        self.pairs = []
        self.target = 0

        self.gen_circ()
        self.step_num = 0
        self.step_tracker=0

        self.target = target
        self.rec_mut_inf = [self.mutinfo(self.target)]

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
    def do_step(self,num=None,rec=None):
        # do things
        if num == None:
            for i in self.circ:
                for j in list(i.keys()):
                    # print(j,i[j])
                    self.do_operation(j, i[j])
                self.step_num = self.step_num + 1
    
                # record things
                if rec == 'mut':
                    self.rec_mut_inf.append(self.mutinfo(self.target))
                elif rec == 'bip':
                    self.rec_bip.append(self.bipent())
        else:
            self.step_tracker+=num
            for st,i in enumerate(self.circ):
                if st>self.step_tracker:
                    pass
                for j in list(i.keys()):
                    # print(j,i[j])
                    self.do_operation(j, i[j])
                self.step_num = self.step_num + 1
    
                # record things
                if rec == 'mut':
                    self.rec_mut_inf.append(self.mutinfo(self.target))
                elif rec == 'bip':
                    self.rec_bip.append(self.bipent())

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
                mat = qu(ikron(rand_match(), [2] * self.num_elems, pair))
                self.dop = mat @ self.dop @ mat.H

        elif op == "markov":
            for pair in ps:
                if self.same: mar=self.markov
                else: mar = markov()
                mat = qu(ikron(self.markov, [2] * self.num_elems, pair))
                self.dop = self.lmc(mat)
                # self.dop = D / trace(D)

        elif op == "meas":
            for pair in ps:
                self.measure(pair)
                
    def lmc(self, mat):
        '''
        slow
        '''
        D=np.zeros_like(self.dop)
        sqrtmat=np.sqrt(mat)
        for i in np.ndindex(np.shape(mat)):
            mk = np.zeros_like(mat)
            mk[i]=sqrtmat[i]
            D+= mk@ self.dop @qu(mk).H
        return D/trace(D)
    
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
    def bipent
        ###############################

    def print_state(self):
        for i in range(len(self.dims)):
            print(partial_trace(self.dop, self.dims, [i]))


#%%
numstep = 1*11
circ = circuit(6, numstep, init="up", meas_r=0.2, gate="markov", architecture="brick",same=0)
#%%
# for i in range(numstep):
circ.do_step()
# circ.print_state()
# print()
#%%
plt.imshow(
    np.log(np.array(circ.rec_mut_inf).round(3))[:, 1:],
    extent=(1, circ.num_elems, numstep, 0),
)  # ,[x for x in range(circ.num_elems)][1:])
plt.xticks(ticks=[x for x in range(1, circ.num_elems)])
plt.title("Log Mutual Information site 0")
plt.ylabel("step number")
plt.xlabel("site number")
plt.colorbar()
#%%
mat=np.diagonal(circ.dop).real
plt.bar([x for x in range(mat.size)],mat)
plt.title("Real part of Diagonal of rho")

#%%
plt.plot(np.nansum(np.log(np.array(circ.rec_mut_inf)), 1))


#%%
numstep = 1*10
circ = circuit(5, numstep, init="up", meas_r=0.5, gate="markov", architecture="brick")

for i in range(1,7):
    mat=np.diagonal(circ.dop).real
    plt.subplot(2, 3, i)
    plt.bar([x for x in range(mat.size)],mat)
    circ.do_step(num=2)
    

plt.show()
