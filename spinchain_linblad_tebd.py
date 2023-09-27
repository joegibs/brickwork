#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 12:35:27 2023

@author: joeg
"""

import numpy as np
from quimb import *
import quimb.tensor as qtn
import matplotlib.pyplot as plt

#%%
def ent_sharp(mps,dims,yup=(2,2)):
    tst = from_lindblad_space(mps,yup)
    traced_tst=ptr(tst.to_dense(),dims,[x for x in range(int(len(dims)/2))])
    # print(trace(traced_tst))
    return entropy(traced_tst)
def ent(mps):
    return mps.entropy(int(mps.L/2))
def ent_rho(rho,dims):
    traced_tst=ptr(rho,dims,[x for x in range(int(len(dims)/2))])
    return entropy(traced_tst)
def neg_sharp(mps,dims):
    L=len(dims)
    tst = from_lindblad_space(mps)
    return logneg_subsys(tst.to_dense(),dims,[i for i in range(int(L/2))],sysb=[i for i in range(int(L/2),L)])
def mut_sharp(mps,dims):
    L=len(dims)
    tst = from_lindblad_space(mps)
    return mutinf_subsys(tst.to_dense(),dims,[i for i in range(int(L/2))],sysb=[i for i in range(int(L/2),L)])

def mps_to_mpo(a):
    return a.partial_trace(range(0,a.L))

def to_lindblad_space(tst):
    arr = []
    for i,t in enumerate(tst):
        # print(t)
        a = t.fuse({f'k{i}':[f'k{i}',f'b{i}']})
        arr.append(a)
    TN = qtn.TensorNetwork(arr)
    fin = qtn.MatrixProductState(TN.arrays)
    fin.normalize()
    return fin
def from_lindblad_space(fin,dims=(2,2)):
    rev = qtn.TensorNetwork([i for i in fin])
    arr=[]
    for i,t in enumerate(rev):
        # print(dims)
        a = t.unfuse({f'k{i}':[f'k{i}',f'b{i}']},{f'k{i}':dims})
        arr.append(a)
    TN = qtn.TensorNetwork(arr)
    ret = qtn.MatrixProductOperator(TN.arrays)
    return ret/ret.trace()
#%%
#%%
L=6
psi0 = qtn.MPS_computational_state('000100')

builder = qtn.SpinHam1D(S=1/2)

# specify the interaction term (defaults to all sites)
builder += 1, 'X', 'X'
builder += 1.0, 'Z'

H=builder.build_local_ham(L)
# add random z-fields to each site
# np.random.seed(2)
# for i in range(L):
#     builder[i] += 2 * np.random.rand() - 1, 'Z'
    
H = builder.build_local_ham(L)

tebd = qtn.TEBD(psi0, H)

# Since entanglement will not grow too much, we can set quite
#     a small cutoff for splitting after each gate application
tebd.split_opts['cutoff'] = 1e-12

# times we are interested in
ts = np.linspace(0, 30, 101)

be_t_b0 = []  # block entropy

# generate the state at each time in ts
#     and target error 1e-3 for whole evolution
for psit in tebd.at_times(ts, tol=1e-3):
    be_b = []
    
    # there is one more site than bond, so start with mag
    #     this also sets the orthog center to 0
    be_t_b0.append(psit.entropy(int(L/2)))
  
tebd.err  #  should be < tol=1e-3
plt.plot(ts,be_t_b0)
#%%
#%%
#build state
L=6
for i in range(0,2):
    psir = qtn.MPS_computational_state('000100')
    # psir = qtn.MPS_computational_state('100001')
    rhor = psir.ptr([i for i in range(0,L)])
    psi0=to_lindblad_space(rhor)
    
    builder = qtn.SpinHam1D(S=3/2)
    #hmmmmmm
    j=1
    builder += 1/4, kron(pauli('X'),np.identity(2)),kron(pauli('X'),np.identity(2))
    builder += -1/4, kron(np.identity(2),pauli('X').T),kron(np.identity(2),pauli('X').T)
    
    builder += 1/2, kron(pauli('Z'),np.identity(2))
    builder += -1/2, kron(np.identity(2),pauli('Z').T)
    # measurement operator
    a=i
    m=np.identity(2)+a*num(2)
    
    # builder += 1/2, kron(m,np.identity(2))
    # builder += -1/2, kron(np.identity(2),m.T)
    builder += 1/2, kron(m,m.H.T)
    builder += -1/4*gamma, kron(m.H@pZ,np.identity(2))
    builder += -1/4*gamma, kron(np.identity(2),(m.H@m).T)
    
    H=builder.build_local_ham(L)
    
    tebd = qtn.TEBD(psi0, H)
    
    # Since entanglement will not grow too much, we can set quite
    #     a small cutoff for splitting after each gate application
    tebd.split_opts['cutoff'] = 1e-12
    
    # times we are interested in
    ts = np.linspace(0, 30, 101)
    
    be_t_b = []  # block entropy
    be_t_b2 = []  # block entropy
    
    
    # generate the state at each time in ts
    #     and target error 1e-3 for whole evolution
    for psit in tebd.at_times(ts, tol=1e-3):
        be_b = []
        
        # there is one more site than bond, so start with mag
        #     this also sets the orthog center to 0
        be_t_b.append(ent_sharp(psit,[2]*L))
        be_t_b2.append(psit.entropy(int(L/2)))
    
        # be_t_b.append(psit.entropy(int(L/2)))
    
      
    tebd.err  #  should be < tol=1e-
    plt.plot(ts,be_t_b)
plt.plot(ts,be_t_b0)