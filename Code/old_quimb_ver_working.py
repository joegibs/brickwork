# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 13:19:49 2022

@author: jogib

TODO
-better switch conditions
-actual match gate

"""
import itertools
from operator import add
import numpy as np
from quimb import *
import matplotlib.pyplot as plt

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

    def __init__(self, num_elems, gate="bell", init="up", architecture="brick"):
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

        if init == "up":
            self.dop = computational_state(
                "".join(["0" for x in range(self.num_elems)]), qtype="dop", sparse=True
            )
        elif init == "rand":
            self.dop = rand_product_state(self.num_elems)
            self.dop = qu(self.dop, qtype="dop")

        self.dims = [2] * self.num_elems
        self.gate = gate

        self.step_num = 0
        self.pairs = []
        self.target = 0

    def gen_step(self, eoo):
        if self.architecture == "brick":
            self.gen_pairs(eoo)
        elif self.architecture == "staircase":
            self.gen_staircase()

        # print(self.pairs)
        for i in self.pairs:
            self.do_operation(i)
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

    def do_operation(self, pair):

        # Needs some work
        if self.gate == "bell":
            # print(pair[0])
            had = ikron(hadamard(), [2] * self.num_elems, pair[0])
            step1 = had @ self.dop @ had.H
            cn = pkron(CNOT(), [2] * self.num_elems, pair)
            self.dop = cn @ step1 @ cn.H
            self.dop.round(4)

        elif self.gate == "haar":
            haar = ikron(rand_uni(2), [2] * self.num_elems, pair)
            self.dop = haar @ self.dop @ haar.H
            self.dop.round(4)

        elif self.gate == "match":
            mat = ikron(rand_match(), [2] * self.num_elems, pair)
            self.dop = mat @ self.dop @ mat.H
            self.dop.round(4)

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

    def mut_info_array_gen(self, num_steps, target=0):
        arr = [circ.mutinfo(target)]
        for i in range(num_steps):
            circ.gen_step(1 - i % 2)
            arr.append(circ.mutinfo(target))
        return arr


#%%
circ = circuit(9, gate="match", init="rand", architecture="brick")
arr = circ.mut_info_array_gen(25, 0)


#%%
plt.imshow(np.log(np.array(arr).round(3)))
plt.title("Log Mutual Information with site 0")
plt.ylabel("step number")
plt.xlabel("site number")
plt.colorbar()
#%%
dop = computational_state("".join(["0" for x in range(2)]), sparse=False)
had = ikron(hadamard(), [2] * 2, 0)
step1 = had @ dop
end = pkron(CNOT(), [2] * 2, [0, 1]) @ step1
# had2 = ikron(hadamard(),[2]*3,1)
# step2 = had@ end
# en2 = pkron(CNOT(),[2]*3,[1,2])@step2
#%%
step1 = had @ end
end = pkron(CNOT(), [2] * 2, [0, 1]) @ step1
#%%
step1 = had @ end
end = pkron(CNOT(), [2] * 2, [0, 1]) @ step1
