# -*- coding: utf-8 -*-
"""
Created on Wed May 10 16:44:53 2023

@author: jogib
"""

import numpy as np
from quimb import *
import quimb.tensor as qtn
import matplotlib.pyplot as plt
#%%
dim=4
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
def markov_dir(eps):
        m=[]
        for i in range(4):
            m.append(np.random.dirichlet(np.ones(4),size=1)[0])
        return np.array(m)

def do_operation(state, gate, ps):
    pairs = [tuple(i) for i in ps]
    
    ops=[gate for i in ps]
    A = pkron(kron(*ops),dims=[2]*dim,inds=np.array(ps).flatten())
    return A@state
def do_operationq(state, gate, ps):
    pairs = [tuple(i) for i in ps]
    
    ops=[gate for i in ps]
    A = pkron(kron(*ops),dims=[2]*dim,inds=np.array(ps).flatten())
    return A@state@A.H

def do_operation_randc(state, gate, ps):
    pairs = [tuple(i) for i in ps]
    
    ops=[np.array(markov_dir(0.1)) for i in ps]
    A = pkron(kron(*ops),dims=[2]*dim,inds=np.array(ps).flatten())
    return A@state
def do_operation_randq(state, gate, ps):
    pairs = [tuple(i) for i in ps]
    
    ops=[rand_uni(4) for i in ps]
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

#%%classical MI

              
arrc = []
for i in range(10000):
    M =np.array(markov_dir(0.1))#np.array([[1,0.4,0.4,0],[0,0.3,0.3,0],[0,0.3,0.3,0],[0,0,0,1]])
    mis=[]
    for i in range(40):
        s=np.random.dirichlet(np.ones(16),size=1)[0]
        tst=np.diag(np.transpose(s).flatten())
        hai = entropy(ptr(tst,[2]*4,[0,1]))
        hbi = entropy(ptr(tst,[2]*4,[1,2]))
        habi=entropy(tst)
        
        for ps in psa:
            s = do_operation(s,M,ps)
            
        tst=np.diag(np.transpose(s).flatten())
        ha= entropy(ptr(tst,[2]*4,[0,1]))
        hb = entropy(ptr(tst,[2]*4,[1,2]))
        hab=entropy(tst)
        mis.append(ha+hb-hab +habi -hai-hbi)

    
    arrc.append(np.mean(mis))

plt.hist(arrc,bins=30,density=True)
#%%
#%%classical MI random M

              
arrcr = []
for i in range(10000):
    M =np.array(markov_dir(0.1))#np.array([[1,0.4,0.4,0],[0,0.3,0.3,0],[0,0.3,0.3,0],[0,0,0,1]])
    mis=[]
    s=np.random.dirichlet(np.ones(16),size=1)[0]
    tst=np.diag(np.transpose(s).flatten())
    hai = entropy(ptr(tst,[2]*4,[0,1]))
    hbi = entropy(ptr(tst,[2]*4,[1,2]))
    habi=entropy(tst)
    
    for ps in psa:
        s = do_operation_randc(s,M,ps)
        
    tst=np.diag(np.transpose(s).flatten())
    ha= entropy(ptr(tst,[2]*4,[0,1]))
    hb = entropy(ptr(tst,[2]*4,[1,2]))
    hab=entropy(tst)
    mis.append(ha+hb-hab +habi -hai-hbi)

    
    arrcr.append(np.mean(mis))

plt.hist(arrcr,bins=30,density=True)

#%%quantum MI
arr=[]
for i in range(10000):
    M =rand_uni(4)#np.array([[1,0.4,0.4,0],[0,0.3,0.3,0],[0,0.3,0.3,0],[0,0,0,1]])
    mis=[]
    for i in range(40):
        s=rand_ket(16)
        hai = entropy_subsys(s,[2]*4,[0,1])
        hbi = entropy_subsys(s,[2]*4,[2,3])
        habi=entropy_subsys(s,[2]*4,[0,1,2,3])
        
        for ps in psa:
            s = do_operation(s,M,ps)
            
        ha = entropy_subsys(s,[2]*4,[0,1])
        hb = entropy_subsys(s,[2]*4,[2,3])
        hab=entropy_subsys(s,[2]*4,[0,1,2,3])
        mis.append(ha+hb-hab +habi -hai-hbi)

    
    arr.append(np.mean(mis))

plt.hist(arr,bins=30,density=True)
#%%quantum MI random
arrr=[]
for i in range(10000):
    M =rand_uni(4)#np.array([[1,0.4,0.4,0],[0,0.3,0.3,0],[0,0.3,0.3,0],[0,0,0,1]])
    mis=[]
    s=rand_ket(16)
    hai = entropy_subsys(s,[2]*4,[0,1])
    hbi = entropy_subsys(s,[2]*4,[2,3])
    habi=entropy_subsys(s,[2]*4,[0,1,2,3])
    
    for ps in psa:
        s = do_operation_randq(s,M,ps)
        
    ha = entropy_subsys(s,[2]*4,[0,1])
    hb = entropy_subsys(s,[2]*4,[2,3])
    hab=entropy_subsys(s,[2]*4,[0,1,2,3])
    mis.append(ha+hb-hab +habi -hai-hbi)

    
    arrr.append(np.mean(mis))

plt.hist(arrr,bins=30,density=True)
#%%classical MI

              
arrc3 = []
for i in range(10000):
    M =np.array(markove(np.random.rand()))#np.array([[1,0.4,0.4,0],[0,0.3,0.3,0],[0,0.3,0.3,0],[0,0,0,1]])
    mis=[]
    for i in range(40):
        s=np.random.dirichlet(np.ones(16),size=1)[0]
        tst=np.diag(np.transpose(s).flatten())
        hai = entropy(ptr(tst,[2]*4,[0,1]))
        hbi = entropy(ptr(tst,[2]*4,[1,2]))
        habi=entropy(tst)
        
        for ps in psa:
            s = do_operation(s,M,ps)
            
        tst=np.diag(np.transpose(s).flatten())
        ha= entropy(ptr(tst,[2]*4,[0,1]))
        hb = entropy(ptr(tst,[2]*4,[1,2]))
        hab=entropy(tst)
        mis.append(ha+hb-hab +habi -hai-hbi)

    
    arrc3.append(np.mean(mis))

plt.hist(arrc3,bins=30,density=True)
#%%
bins = np.linspace(-1.5, 1.5, 200)
plt.hist(arr, bins, alpha=0.5, label='Unitary',density=True)
plt.hist(arrc, bins, alpha=0.5, label='Markov',density=True)
plt.hist(arrc3, bins, alpha=0.5, label='Markov_cross',density=True)

# plt.hist(arrr, bins, alpha=0.5, label='Random Quantum',density=True)
# plt.hist(arrcr, bins, alpha=0.5, label='Random Classical',density=True)

plt.legend(loc='upper right')
plt.title("Delta Bi-partite MI after Two Layers Random Initial State")

plt.show()
#%%
#%%classical entropy

              
arrc = []
for i in range(10000):
    M =np.array(markov_dir(0.1))#np.array([[1,0.4,0.4,0],[0,0.3,0.3,0],[0,0.3,0.3,0],[0,0,0,1]])
    mis=[]
    for i in range(40):
        s=np.random.dirichlet(np.ones(16),size=1)[0]
        tst=np.diag(np.transpose(s).flatten())
        hai = entropy(ptr(tst,[2]*4,[0,1]))

        
        for ps in psa:
            s = do_operation(s,M,ps)
            
        tst=np.diag(np.transpose(s).flatten())
        ha= entropy(ptr(tst,[2]*4,[0,1]))

        mis.append(ha-hai)

    
    arrc.append(np.mean(mis))

plt.hist(arrc,bins=30,density=True)

#%%quantum Entropy
arr=[]
for i in range(10000):
    M =rand_uni(4)#np.array([[1,0.4,0.4,0],[0,0.3,0.3,0],[0,0.3,0.3,0],[0,0,0,1]])
    mis=[]
    for i in range(40):
        s=rand_ket(16)
        tst=np.diag(np.transpose(s).flatten())
        hai = entropy(ptr(tst,[2]*4,[0,1]))

        
        for ps in psa:
            s = do_operation(s,M,ps)
            
        tst=np.diag(np.transpose(s).flatten())
        ha= entropy(ptr(tst,[2]*4,[0,1]))

        mis.append(ha-hai)

    
    arr.append(np.mean(mis))

plt.hist(arr,bins=30,density=True)
#%%classical entropy cross

              
arrc3 = []
for i in range(10000):
    M =np.array(markove(np.random.rand()))#np.array([[1,0.4,0.4,0],[0,0.3,0.3,0],[0,0.3,0.3,0],[0,0,0,1]])
    mis=[]
    for i in range(40):
        s=np.random.dirichlet(np.ones(16),size=1)[0]
        tst=np.diag(np.transpose(s).flatten())
        hai = entropy(ptr(tst,[2]*4,[0,1]))

        
        for ps in psa:
            s = do_operation(s,M,ps)
            
        tst=np.diag(np.transpose(s).flatten())
        ha= entropy(ptr(tst,[2]*4,[0,1]))

        mis.append(ha-hai)

    
    arrc3.append(np.mean(mis))

plt.hist(arrc3,bins=30,density=True)
#%%
bins = np.linspace(-.5, .5, 100)
plt.hist(arr, bins, alpha=0.5, label='Unitary',density=True)
plt.hist(arrc, bins, alpha=0.5, label='Markov',density=True)
plt.hist(arrc3, bins, alpha=0.5, label='Markov_cross',density=True)

plt.legend(loc='upper right')
plt.title("Delta Bi-partite Entropy after Two Layers Random Initial State")

plt.show()
    #%%
data = [np.transpose(i)[0] for i in s_history]

s_hist=[]
for i in s_history:
    state=[]
    for j in range(4):
        tst=np.diag(np.transpose(i).flatten())
        state.append(ptr(tst,[2]*4,j)[0,0])
    print(state)
    s_hist.append(state)

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

#%%
arr=[]
for i in range(10000):
    mis=[]
    s=rand_ket(2**10)
    hai = entropy_subsys(s,[2]*10,[0,1,2,3])#ptr(tst,[2]*4,[0,1]))

    # mis.append(hai)
    arr.append(hai*0.69314478)#np.mean(mis))

plt.hist(arr,bins=200,density=True)
plt.title("page curve N=10 Na=4")
#%%
n=10
m=4
tot=0
for k in range(n+1,n*m):
    tot+= 1/k-(m-1)/(2*n)
print(tot)
