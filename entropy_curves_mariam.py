#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 09:32:36 2023

@author: joeg
"""
import numpy as np
from quimb import *
import quimb.tensor as qtn
import matplotlib.pyplot as plt
#%% base case
L=12
# for i in range(0,1):
# psi0 = qtn.MPS_rand_state(L,bond_dim=2,phys_dim=2)
psi0 = qtn.MPS_computational_state('0'*L)
# builder = qtn.SpinHam1D(S=1/2)
labels=[]
for i in range(5):
    hj=np.random.rand(4)*2
    # hj=[1,1]
    # H = builder.build_local_ham(L)
    builder = qtn.SpinHam1D(S=1/2)
    builder += hj[0], 'Z', 'Z'
    builder += hj[1], 'X'
    
    H=builder.build_local_ham(L)
    
    # H = qtn.ham_1d_mbl(L=L,dh=0)
    H = qtn.ham_1d_heis(L=L,j=hj[:3],bz=hj[4])
    
    tebd = qtn.TEBD(psi0, H)
    
    # Since entanglement will not grow too much, we can set quite
    #     a small cutoff for splitting after each gate application
    tebd.split_opts['cutoff'] = 1e-12
    
    # times we are interested in
    ts = np.linspace(0, 100, 201)
    
    be_t_b0 = []  # block entropy
    ne_t_b0 = []  # block entropy
    
    # generate the state at each time in ts
    #     and target error 1e-3 for whole evolution
    for psit in tebd.at_times(ts, tol=1e-3):
        be_b = []
        
        # there is one more site than bond, so start with mag
        #     this also sets the orthog center to 0
        be_t_b0.append(psit.entropy(int(L/2)))
        # ne_t_b0.append(psit.logneg_subsys([0,1,2],[3,4,5]))
    
      
    tebd.err  #  should be < tol=1e-3
    labels.append(f"{hj[0].round(2)},{hj[1].round(2)}")
    plt.plot(ts,be_t_b0)
plt.title(f"{L} Qubit ZZ+X")
plt.ylabel("bip ent")
plt.xlabel("time")
plt.legend(labels=labels,title = "Coeff j,h")
#%%
#%% base case
L=12
# for i in range(0,1):
# psi0 = qtn.MPS_rand_state(L,bond_dim=2,phys_dim=2)
psi0 = qtn.MPS_computational_state('0'*L)
# builder = qtn.SpinHam1D(S=1/2)
labels=[]
for i in range(10):
    hj=np.random.rand(4)*2
    # hj=[1,1]
    # H = builder.build_local_ham(L)
    # builder = qtn.SpinHam1D(S=1/2)
    # builder += hj[0], 'Z', 'Z'
    # builder += hj[1], 'X'
    
    H=builder.build_local_ham(L)
    
    # H = qtn.ham_1d_mbl(L=L,dh=0)
    H = qtn.ham_1d_heis(L=L,j=hj[:3],bz=hj[3])
    
    tebd = qtn.TEBD(psi0, H)
    
    # Since entanglement will not grow too much, we can set quite
    #     a small cutoff for splitting after each gate application
    tebd.split_opts['cutoff'] = 1e-12
    
    # times we are interested in
    ts = np.linspace(0, 100, 201)
    
    be_t_b0 = []  # block entropy
    ne_t_b0 = []  # block entropy
    
    # generate the state at each time in ts
    #     and target error 1e-3 for whole evolution
    for psit in tebd.at_times(ts, tol=1e-3):
        be_b = []
        
        # there is one more site than bond, so start with mag
        #     this also sets the orthog center to 0
        be_t_b0.append(psit.entropy(int(L/2)))
        # ne_t_b0.append(psit.logneg_subsys([0,1,2],[3,4,5]))
    
      
    tebd.err  #  should be < tol=1e-3
    labels.append(f"{hj[0].round(2)},{hj[1].round(2)}{hj[2].round(2)},{hj[3].round(2)}")
    plt.plot(ts,be_t_b0)
plt.title(f"{L} Qubit Ham_heis")
plt.ylabel("bip ent")
plt.xlabel("time")
plt.legend(labels=labels,title = "Coeff jx,jy,jz,b")
#%%
#%% base case
L=12
# for i in range(0,1):
# psi0 = qtn.MPS_rand_state(L,bond_dim=2,phys_dim=2)
psi0 = qtn.MPS_computational_state('0'*L)
# builder = qtn.SpinHam1D(S=1/2)
labels=[]
for i in range(10):
    hj=np.random.rand(4)*2
    # hj=[1,1]
    # H = builder.build_local_ham(L)
    # builder = qtn.SpinHam1D(S=1/2)
    # builder += hj[0], 'Z', 'Z'
    # builder += hj[1], 'X'
    
    H=builder.build_local_ham(L)
    
    # H = qtn.ham_1d_mbl(L=L,dh=0)
    H = qtn.ham_1d_heis(L=L,j=hj[:3],bz=hj[3])
    
    tebd = qtn.TEBD(psi0, H)
    
    # Since entanglement will not grow too much, we can set quite
    #     a small cutoff for splitting after each gate application
    tebd.split_opts['cutoff'] = 1e-12
    
    # times we are interested in
    ts = np.linspace(0, 100, 201)
    
    be_t_b0 = []  # block entropy
    ne_t_b0 = []  # block entropy
    
    # generate the state at each time in ts
    #     and target error 1e-3 for whole evolution
    for psit in tebd.at_times(ts, tol=1e-3):
        be_b = []
        
        # there is one more site than bond, so start with mag
        #     this also sets the orthog center to 0
        be_t_b0.append(psit.entropy(int(L/2)))
        # ne_t_b0.append(psit.logneg_subsys([0,1,2],[3,4,5]))
    
      
    tebd.err  #  should be < tol=1e-3
    labels.append(f"{hj[0].round(2)},{hj[1].round(2)}{hj[2].round(2)},{hj[3].round(2)}")
    plt.plot(ts,be_t_b0)
plt.title(f"{L} Qubit Ham_heis")
plt.ylabel("bip ent")
plt.xlabel("time")
plt.legend(labels=labels,title = "Coeff jx,jy,jz,b")
#%%
#%% base case
L=12
# for i in range(0,1):
# psi0 = qtn.MPS_rand_state(L,bond_dim=2,phys_dim=2)
psi0 = qtn.MPS_computational_state('0'*L)
# builder = qtn.SpinHam1D(S=1/2)
labels=[]
for i in range(10):
    hj=np.random.rand(2)*2
    # hj=[1,1]
    # H = builder.build_local_ham(L)
    # builder = qtn.SpinHam1D(S=1/2)
    # builder += hj[0], 'Z', 'Z'
    # builder += hj[1], 'X'
    
    H=builder.build_local_ham(L)
    
    # H = qtn.ham_1d_mbl(L=L,dh=0)
    H = qtn.ham_1d_XY(L=L,j=hj,bz=0)
    
    tebd = qtn.TEBD(psi0, H)
    
    # Since entanglement will not grow too much, we can set quite
    #     a small cutoff for splitting after each gate application
    tebd.split_opts['cutoff'] = 1e-12
    
    # times we are interested in
    ts = np.linspace(0, 200, 201)
    
    be_t_b0 = []  # block entropy
    ne_t_b0 = []  # block entropy
    
    # generate the state at each time in ts
    #     and target error 1e-3 for whole evolution
    for psit in tebd.at_times(ts, tol=1e-3):
        be_b = []
        
        # there is one more site than bond, so start with mag
        #     this also sets the orthog center to 0
        be_t_b0.append(psit.entropy(int(L/2)))
        # ne_t_b0.append(psit.logneg_subsys([0,1,2],[3,4,5]))
    
      
    tebd.err  #  should be < tol=1e-3
    labels.append(f"{hj[0].round(2)},{hj[1].round(2)}")
    plt.plot(ts,be_t_b0)
plt.title(f"{L} Qubit XY")
plt.ylabel("bip ent")
plt.xlabel("time")
plt.legend(labels=labels,title = "Coeff jx,jy")
