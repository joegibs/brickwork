# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 10:00:01 2022

@author: jogib

todo:
    -step generating function
        fix up selection for pairs
    -actual op for do op
    -integrate scalar bits for testing
    -typing
    -change to density matrix for ciruit eventually
"""
import numpy as np
from pprint import pprint

np.set_printoptions(suppress=True)

#%%
class density_matrix:
    def __init__(self, num_elems):
        self.value = np.Identity(num_elems)


class circuit:
    """
    contains the bits and mixing functions

    Returns
    -------
    None.

    """

    def __init__(self, num_elems):
        self.num_elems = num_elems
        self.elem_arr = [bits() for x in range(self.num_elems)]
        self.step_num = 0
        self.pairs = []

    def gen_step(self, eoo):
        self.gen_pairs(eoo)
        for i in self.pairs:
            self.do_operation(i)

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

    def do_operation(self, pair):
        angle = np.pi / 4
        op = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

        self.elem_arr[pair[0]].value = np.dot(op, self.elem_arr[pair[0]].value)
        self.elem_arr[pair[1]].value = np.dot(op, self.elem_arr[pair[1]].value)

    def print_arr(self):
        print()
        for i in self.elem_arr:
            pprint(i.value)


#%%
circ = circuit(6)
circ.gen_step(0)
print(circ.pairs)
circ.print_arr()
circ.gen_step(1)
circ.print_arr()

########################################################33
#%%
class bits:
    def __init__(self):
        self.value = np.array([[1, 0], [0, 1]])


class circuit:
    """
    contains the bits and mixing functions

    Returns
    -------
    None.

    """

    def __init__(self, num_elems):
        self.num_elems = num_elems
        self.elem_arr = [bits() for x in range(self.num_elems)]
        self.step_num = 0
        self.pairs = []

    def gen_step(self, eoo):
        self.gen_pairs(eoo)
        for i in self.pairs:
            self.do_operation(i)

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

    def do_operation(self, pair):
        angle = np.pi / 4
        op = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

        self.elem_arr[pair[0]].value = np.dot(op, self.elem_arr[pair[0]].value)
        self.elem_arr[pair[1]].value = np.dot(op, self.elem_arr[pair[1]].value)

    def print_arr(self):
        print()
        for i in self.elem_arr:
            pprint(i.value)


#%%
circ = circuit(6)
circ.gen_step(0)
print(circ.pairs)
circ.print_arr()
circ.gen_step(1)
circ.print_arr()

#%%
"""
scalar bits
"""


class bits:
    def __init__(self):
        self.value = 1


class circuit:
    """
    contains the bits and mixing functions

    Returns
    -------
    None.

    """

    def __init__(self, num_elems):
        self.num_elems = num_elems
        self.elem_arr = [bits() for x in range(self.num_elems)]
        self.step_num = 0
        self.pairs = []

    def gen_step(self, eoo):
        self.gen_pairs(eoo)
        for i in self.pairs:
            self.do_operation(i)

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

    def do_operation(self, pair):
        angle = np.pi / 4
        op = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        # get
        val1 = self.elem_arr[pair[0]].value
        val2 = self.elem_arr[pair[1]].value
        # op
        val1 = val1 + 1
        val2 = val2 + 1
        # write
        self.elem_arr[pair[0]].value = val1
        self.elem_arr[pair[1]].value = val2

    def print_arr(self):
        print()
        for i in self.elem_arr:
            print(i.value, " ", end="")
