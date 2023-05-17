# -*- coding: utf-8 -*-
"""
Created on Wed May  3 13:08:30 2023

@author: jogib
"""

import numpy as np
from quimb import *
import quimb.tensor as qtn
import matplotlib.pyplot as plt
#%%
psa = [[[0,1],[2,3]],[[1,2]]]
# psa = [[[0,1]],[[0,1]]]
dim=4
# def do_operation(state, gate, ps):
#     pairs = [tuple(i) for i in ps]
#     for p in pairs:
#         A = pkron(kron(gate),dims=[2]*4,inds=np.array(p).flatten())
#     return A@state
def markov_alt(eps):
    # need to check this currently a left matrix....
    M=[]
    for i in range(4):
        arr=[]
        tot = 1
        for j in range(3):
            samp = np.random.uniform(0,tot)
            tot -=samp
            arr.append(samp)
        arr.append(tot)
        M.append(arr)
    return M

def do_operation(state, gate, ps):
    pairs = [tuple(i) for i in ps]
    
    ops=[gate for i in ps]
    A = pkron(kron(*ops),dims=[2]*dim,inds=np.array(ps).flatten())
    return A@state

def measure_s(state, inds):
    sts=[]
    tst = np.diag(state.flatten())
    for i in range(dim):
        if i in inds:
            ops = pauli("Z")
            states = ptr(tst,[2]*dim,i)
            _,newi=measure(states,ops)
            sts.append(newi)
        else: 
            state_n = ptr(tst,[2]*dim,i)
            sts.append(state_n)

    arr = np.diag(kron(*sts)).real
    return arr
def mutinfo(state,target):
    mi = mutinf(state,dims=[2]*dim)
    return mi
def ent_calc(matrix):
    sums=0
    for i in range(4):
        sums-=matrix[i,i]*np.log2(matrix[i,i])
    return sums
#%% classical MI
arrc= []
maxmi = 0
maxm = []
for i in range(10000):
    
    
    # mi1 = mutinf_subsys(np.array(state),dims=[2]*2,sysa=[0],sysb=[1])
    M =np.array(markov_alt(0.1))#np.array([[1,0.4,0.4,0],[0,0.3,0.3,0],[0,0.3,0.3,0],[0,0,0,1]])
    mis=[]
    for i in range(4):
        state=[0.,0,0,0.]
        state[i]=1
        s1=M.T@state
        tst=np.diag(np.transpose(s1).flatten())
        ha = entropy(ptr(tst,[2]*2,0))
        hb = entropy(ptr(tst,[2]*2,1))
        hab=entropy(tst)
        
        mi2 = mutinf(tst,sysa=0)#ha+hb-hab#mutinf(tst,dims=[2]*2)#mutinf_subsys(s1,dims=[2]*2,sysa=[0],sysb=[1])
        mis.append(ha)
    
    if np.mean(mis)>maxmi:
        maxmi = np.mean(mis)
        maxm=M
    arrc.append(np.mean(mis))

plt.hist(arrc,bins=30,density=True)
#%% quantum entropy
arr= []
maxhi = 0
maxh = []
for i in range(100000):
    
    
    # mi1 = mutinf_subsys(np.array(state),dims=[2]*2,sysa=[0],sysb=[1])
    M =rand_uni(4)
    mis=[]
    for i in range(4):
        state=[0.,0,0,0.]
        state[i]=1
        s1=M@qu(state,qtype='ket')
        # tst=np.diag(np.transpose(s1).flatten())
        ha = entropy(ptr(s1,[2]*2,0))
        miab = mutinf_subsys(s1,[2]*2,0,1)
        
        mi2 = miab#mutinf(tst,dims=[2]*2)#mutinf_subsys(s1,dims=[2]*2,sysa=[0],sysb=[1])
        mis.append(ha)
    
    if np.mean(mis)>maxhi:
        maxhi = np.mean(mis)
        maxh=M
    arr.append(np.mean(mis))

plt.hist(arr,bins=50,density=True)

#%%
#%% quantum noise MI
arr= []
maxhi = 0
maxh = []
p=0.
for i in range(10000):
    
    
    # mi1 = mutinf_subsys(np.array(state),dims=[2]*2,sysa=[0],sysb=[1])
    M =rand_uni(4)
    mis=[]
    for i in range(4):
        state=[0.,0,0,0.]
        state[i]=1
        s1=M@qu(state,qtype='dop')@M.H
        # tst=np.diag(np.transpose(s1).flatten())
        dop1=qu(s1,qtype='dop')
        Z = pauli('z',dim=2)&pauli('z',dim=2)

        # site1 = ptr(dop1,[2]*2,0)
        # site2 = ptr(dop1,[2]*2,1)
        dop2 = (1-p)*dop1 + p*(Z@dop1@Z)
        dop2 = (1-p)*dop1 + p/4*(np.identity(4))

        # ds2 = (1-p)*site2 + p*(Z@site2@Z.H)
        # dop2 = ds1&ds2

        hab=entropy(dop2)
        miab = mutinf(dop2,sysa=0)#ha+hb-hab#mutinf(dop2)
        
        mi2 = round(miab,8)#mutinf(tst,dims=[2]*2)#mutinf_subsys(s1,dims=[2]*2,sysa=[0],sysb=[1])
        mis.append(mi2)
    
    if np.mean(mis)>maxhi:
        maxhi = np.mean(mis)
        maxh=M
    arr.append(np.mean(mis))

plt.hist(arr,bins=30,density=True)
#%%

M=np.array([[1,0.4,0.4,0],[0,0.3,0.3,0],[0,0.3,0.3,0],[0,0,0,1]])
s=computational_state("1101").real
s_history=[s]
              
i=0
steps=40
while i<steps/2:
    j=0
    for ps in psa:
        s = do_operation(s,M,ps)
        if i==987987 and j ==0:
            print('hey')
            s = measure_s(s,[0])
        j+=1
        s_history.append(s)
    i+=1
#%%
data = [np.transpose(i)[0] for i in s_history]

s_hist=[]
m_hist=[]
for i in s_history:
    state=[]
    mi = []
    for j in range(dim):
        tst=np.diag(np.transpose(i).flatten())
        state.append(ptr(tst,[2]*dim,j)[0,0])
        if j==0:
            m_hist.append(mutinfo(tst,0))
    s_hist.append(state)
    
    # m_hist.append(mi)
#%%
fig, ax = plt.subplots()
ax.plot(m_hist)
#%%
fig, ax = plt.subplots()
ax.imshow(s_hist)
def format_coord(x, y):
    col = round(x)
    row = round(y)
    nrows, ncols = X.shape
    if 0 <= col < ncols and 0 <= row < nrows:
        z = X[row, col]
        return f'x={x:1.4f}, y={y:1.4f}, z={z:1.4f}'
    else:
        return f'x={x:1.4f}, y={y:1.4f}'


ax.format_coord = format_coord
plt.show()

#%%%
from scipy.sparse import diags
og=test
tst=qu(diags(og.flatten()).tocsr(),sparse=True,qtype='dop')
ha = entropy(ptr(tst,[2]*4,[0,1]))
hb = entropy(ptr(tst,[2]*4,[2,3]))
hab = entropy(ptr(tst,[2]*4,range(4)))
print(ha,hb)
print(ha+hb-hab)
print(mutinf_subsys(
        tst,
        dims=[2]*2,
        sysa=list(range(1)),
        sysb=list(range(1, 2)),
    ))