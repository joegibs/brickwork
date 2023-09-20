# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 11:49:18 2023

@author: jogib
"""

import numpy as np
from quimb import *
import quimb.tensor as qtn

#%% dephase?
A = qtn.MPO_rand_herm(2,bond_dim=5, tags=['HAM'])

print(entropy(A.to_dense()))
print(mutinf(A.to_dense(),dims=[2]*2,sysa=[0]))

b = dephase(A.to_dense(),0.5)

print(entropy(b))
print(mutinf(b,dims=[2]*2,sysa=[0]))
#%%
