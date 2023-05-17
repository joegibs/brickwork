# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 15:59:30 2023

@author: jogib
"""

import numpy as np
from quimb import *
import quimb.tensor as qtn
#%%
psi = computational_state("1111")
had =ikron(hadamard(),[2]*4,1)
cnot = ikron(CNOT(),[2]*4,[1,2])

fin = cnot@had@psi
print(entropy_subsys(psi,[2]*4,[0,1]))
print(entropy_subsys(fin,[2]*4,[0,1]))
# print(entropy_subsys(fin,[2]*4,[1,2]))
# print(entropy_subsys(fin,[2]*4,[2,3]))
#%%
rho = computational_state("0000",qtype='dop')

had =ikron(hadamard(),[2]*4,0)
cnot = ikron(CNOT(),[2]*4,[0,1])

fin = cnot.T@had.T@rho@had@cnot
print(entropy_subsys(psi,[2]*4,[0]))
print(entropy_subsys(fin,[2]*4,[0]))
# print(entropy_subsys(fin,[2]*4,[1,2]))
# print(entropy_subsys(fin,[2]*4,[2,3]))

#%%

M = ptr(fin,[2]*4,[2,3])

print(M@M)


#%%
t=1
for i in range(30,1,-5):
    t*=i
print(t)

#%%
psi = computational_state("0000")
psi1 = computational_state("0110")
psi2 = computational_state("1001")
psi3 = computational_state("1111")

tot = psi + psi1 + psi2 + psi3

# print((tot.T & tot).real)


rho = (tot.T & tot)

rterho= ptr(rho,[2]*4,[1,2])
#%%
rterho = rterho/4
d,u= np.linalg.eig(rterho)