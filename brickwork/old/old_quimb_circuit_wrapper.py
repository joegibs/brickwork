# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 19:16:27 2022

@author: jogib
"""
import numpy as np

import quimb as qu
import quimb.tensor as qtn
import matplotlib.pyplot as plt
import hypernetx
#%%
# N = 2
# circ = qtn.Circuit(N)
# circ.apply_gate("H", 0)
# circ.apply_gate("CNOT", 0, 1)
def match(theta, phi):
    return np.array(
        [
            [np.cos(theta), 0, 0, -np.sin(theta)],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [np.sin(theta), 0, 0, np.cos(theta)],
        ]
    )


def rand_match():
    arr = 2 * np.pi * np.random.rand(2)
    return match(arr[0], arr[1])

class circuit_wrap:

    def __init__(self, num_elems, gate="bell", architecture="brick"):
        """
        num_elems: number of elements in the chain
        gate: type of gate used
            -"bell" hadamard then cnot to make a bell state
            -"haar" haar random unitary operators
        init: initialization of qqubits
        
        architecture: arrangement of gates
            -"brick": alternating pairs
            -"staircase": each step only has one gate
        """

        self.num_elems = num_elems
        self.architecture = architecture

        """ this need to be updated for different inits"""

        self.circ = qtn.Circuit(self.num_elems)
        # for x in list(range(self.num_elems)):
        #     self.circ.apply_gate('H', x)

        self.gate = gate

        self.step_num = 0
        self.pairs = []
        self.target = 0
        self.mutinf_arr=[]


    def gen_step(self, eoo):
        if self.architecture == "brick":
            self.gen_pairs(eoo)
        elif self.architecture == "staircase":
            self.gen_staircase()

        # print(self.pairs)
        for i in self.pairs:
            self.add_gates_to_pair(i)
        self.step_num = self.step_num + 1

    def gen_pairs(self, eoo):
        self.pairs = []
        # some control over where indexing starts
        if eoo:
            i = 0
        else:
            i = 1
        # get pairs
        while i + 2 <= self.num_elems:
            self.pairs.append([i, i + 1])
            i = i + 2

    def gen_staircase(self):
        self.pairs = []
        if (self.step_num + 1) % self.num_elems == 0:
            self.step_num = self.step_num + 1
        self.pairs.append(
            [self.step_num % self.num_elems, (self.step_num + 1) % self.num_elems]
        )

    def add_gates_to_pair(self, pair):

        # Needs some work
        if self.gate == "bell":
            self.circ.apply_gate('H', pair[0], gate_round=self.step_num)
            self.circ.apply_gate('CNOT', pair[0],pair[1], gate_round=self.step_num)

        elif self.gate == "haar":
            self.circ.apply_gate_raw(qu.rand_uni(4), pair[0],pair[1], gate_round=self.step_num)

        elif self.gate == "match":
            self.circ.apply_gate_raw(rand_match(), (pair[0],pair[1]), gate_round=self.step_num)

    def mutinfo(self, target=0):
        # this is mem bad
        
        # p=[self.circ.partial_trace([target,x]) if x != target else np.identity(4) for x in range(self.num_elems)]
        # # print(p)
        # mi = [
        #     1-np.trace(np.dot(np.transpose(np.conjugate(x)),x)) for x in p 
        # ]
        mi = [qu.entropy(self.circ.partial_trace([target,x])) if x != target else 0 for x in range(self.num_elems)]
        # print(mi)
        return mi

    def circ_gen(self, num_steps, target=0):
        # self.mutinf_arr = self.mutinfo(target)
        self.mutinf_arr=[self.mutinfo(target)]
        for i in range(num_steps):
            self.gen_step(1 - i % 2)
            # print(self.mutinfo(target))
            # self.draw()
            self.mutinf_arr.append( self.mutinfo(target))
    
    def draw(self):
        self.circ.psi.draw(color=['PSI0', 'H', 'CNOT', 'RZ', 'RX', 'CZ'])
        # self.circ.psi.draw(color=['PSI0'] + [f'ROUND_{i}' for i in range(10)])

        # self.circ.psi.draw(color=[f'I{i}' for i in r"ange(self.num_elems)])


        # H = hypernetx.Hypergraph(self.circ.psi.ind_map)
        # hypernetx.draw(H, pos=self.circ.psi.draw(get='pos'))
    def draw_bar(self):
        '''
        draws a bar plot of the probabilities of every state
        not sure how usefull but here we are

        Returns
        -------
        None.

        '''
        arr = [circ.circ.amplitude("{0:06b}".format(x)) for x in range(2**self.num_elems)]
        plt.bar(["{0:06b}".format(x) for x in range(2**6)],np.power(arr,2))
#%%
circ = circuit_wrap(8, gate="match", architecture="brick")
circ.circ_gen(25, 0)
circ.draw()

#%%
plt.imshow(circ.mutinf_arr)
plt.title("Log Mutual Information with site 0")
plt.ylabel("step number")
plt.xlabel("site number")
plt.colorbar()
#%%
qu.purify(self.circ.partial_trace([target,target])
#%%
arr = []
for y in range(20):
    circ = circuit_wrap(6, gate="match", architecture="brick")
    circ.circ_gen(y, 0)
    arr.append([qu.mutinf(circ.circ.partial_trace([5,x])) if x != 5 else 0. for x in range(6)])

plt.imshow(np.log(arr))
plt.title("Log Mutual Information with site 0")
plt.ylabel("step number")
plt.xlabel("site number")
plt.colorbar()