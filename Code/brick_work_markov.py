# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 16:05:17 2022

@author: jogib
"""
import itertools
from operator import add
# from telnetlib import EL
# from tkinter import ARC
import numpy as np
from quimb import *
import matplotlib.pyplot as plt

#%%
def markov():
    # need to check this currently a left matrix....
    M = np.random.rand(4, 4)
    M = M / np.sum(M, axis=0, keepdims=True)
    return M


# def markov():
#     M=np.array([[0.6,0.2],[0.4,0.8]])
#     return M
def decompose_4by1(arr):
    u1 = np.array([[arr[0][0] + arr[1][0]], [arr[2][0] + arr[3][0]]])
    u2 = np.array([[arr[0][0] + arr[2][0]], [arr[1][0] + arr[3][0]]])
    return u1, u2


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
        gate="bell",
        init="up",
        architecture="brick",
        meas_r=0.0,
        target=0,
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

        """ this need to be updated for different inits"""

        if init == "up":
            self.dop = np.array(
                [np.array([[1.0], [0.0]]) for x in range(self.num_elems)]
            )

        self.dims = [2] * self.num_elems
        self.gate = gate

        self.num_steps = num_steps
        self.step_num = 0
        self.pairs = []
        self.target = 0

        self.gen_circ()
        self.step_num = 0

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
                pairs = self.gen_pairs(self.step_num % 2)
                step_dct.update({self.gate: pairs})
                # self.step_num += 1

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
    def do_step(self):
        # do things
        for i in self.circ:
            for j in list(i.keys()):
                # print(j,i[j])
                self.do_operation(j, i[j])
            self.step_num = self.step_num + 1

            # record things
            self.rec_mut_inf.append(self.mutinfo(self.target))

    def do_operation(self, op, ps):

        if op == "markov":
            for pair in ps:
                # combine
                arr = kron(self.dop[pair[0]], self.dop[pair[1]])
                # markov
                mat = np.matmul(markov(), arr)
                # seperate
                self.dop[pair[0]], self.dop[pair[1]] = decompose_4by1(mat)

        elif op == "meas":
            for pair in ps:
                self.measure(pair)

    def measure(self, ind):
        a, self.dop = self.measure_state(self.dop, ind)

    def measure_state(self, p, ind):
        pj = [p[ind][x][0] for x in range(2)]
        # print(pj)
        el = [np.array([[1.0], [0.0]]), np.array([[0.0], [1.0]])]
        # then choose one
        j = np.random.choice([0, 1], p=pj)
        eigenvalue = el[j]

        p[ind] = eigenvalue

        return eigenvalue, p

    def mutinfo(self, target=0):
        # this is mem bad
        arr = [kron(self.dop[target], self.dop[x]) for x in range(self.num_elems)]
        mi = [mutinf(arr[x]) for x in range(self.num_elems)]
        return mi
        ###############################

    def print_state(self):
        print(self.dop)


#%%
numstep = 20
circ = circuit(7, numstep, init="rand", meas_r=0., gate="haar")
#%%
# for i in range(numstep):
circ.do_step()
# circ.print_state()
# print()
#%%
plt.imshow(np.log(np.array(circ.rec_mut_inf).round(3)))
plt.title("Log Mutual Information site 0")
plt.ylabel("step number")
plt.xlabel("site number")
plt.colorbar()
#%%
