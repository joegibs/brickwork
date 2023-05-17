# -*- coding: utf-8 -*-
"""
Created on Tue May  9 14:03:23 2023

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
def markov(eps):
    M = np.random.randint(0,high=20,size=(4, 4))
    for i in range(15):
        M = M / np.sum(M, axis=0, keepdims=True)
        # M = M / np.sum(M, axis=1, keepdims=True)
    if np.isnan(np.min(M)):
        M=markov(eps)
    return M
def markove(eps):
    M = np.array(
        [
            [1-eps, 0, 0, eps],
            [0, 1-eps, eps, 0],
            [0, eps, 1-eps, 0],
            [eps, 0, 0, 1-eps],
        ])
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

#%% classical entropy
arrc= []
maxmi = 0
maxm = []
for i in range(10000):
    
    
    # mi1 = mutinf_subsys(np.array(state),dims=[2]*2,sysa=[0],sysb=[1])
    M =np.array(markov_dir(0.1))#np.array([[1,0.4,0.4,0],[0,0.3,0.3,0],[0,0.3,0.3,0],[0,0,0,1]])
    mis=[]
    for i in range(40):
        # state=[0.,0,0,0.]
        # state[i]=1
        state=np.random.dirichlet(np.ones(4),size=1)[0]
        tst=np.diag(np.transpose(state).flatten())
        hainit = entropy(ptr(tst,[2]*2,0))
        s1=M@state
        tst=np.diag(np.transpose(s1).flatten())
        ha = entropy(ptr(tst,[2]*2,0))

        mis.append(ha-hainit)
    
    if np.mean(mis)>maxmi:
        maxmi = np.mean(mis)
        maxm=M
    arrc.append(np.mean(mis))

plt.hist(arrc,bins=30,density=True)
plt.title("Classical bi-partite Entropy")
#%% quantum entropy
arr= []
maxhi = 0
maxh = []
for i in range(10000):
    
    
    # mi1 = mutinf_subsys(np.array(state),dims=[2]*2,sysa=[0],sysb=[1])
    M =rand_uni(4)
    mis=[]
    for i in range(40):
        # state=[0.,0,0,0.]
        # state[i]=1
        s=rand_ket(4)
        hinit = entropy_subsys(s,[2]*2,0)
        s1=M@qu(s,qtype='ket')
        # tst=np.diag(np.transpose(s1).flatten())
        ha = entropy_subsys(s1,[2]*2,0)
        # miab = mutinf_subsys(s1,[2]*2,0,1)
        
        # mi2 = miab#mutinf(tst,dims=[2]*2)#mutinf_subsys(s1,dims=[2]*2,sysa=[0],sysb=[1])
        mis.append(ha-hinit)
    
    if np.mean(mis)>maxhi:
        maxhi = np.mean(mis)
        maxh=M
    arr.append(np.mean(mis))

plt.hist(arr,bins=50,density=True)
plt.title("Quantum bi-partite Entropy")
#%% classical cross
arrc3= []
maxmi = 0
maxm = []
for i in range(10000):
    
    
    # mi1 = mutinf_subsys(np.array(state),dims=[2]*2,sysa=[0],sysb=[1])
    M =np.array(markove(np.random.rand()))#np.array([[1,0.4,0.4,0],[0,0.3,0.3,0],[0,0.3,0.3,0],[0,0,0,1]])
    mis=[]
    for i in range(40):
        # state=[0.,0,0,0.]
        # state[i]=1
        state=np.random.dirichlet(np.ones(4),size=1)[0]
        tst=np.diag(np.transpose(state).flatten())
        hainit = entropy(ptr(tst,[2]*2,0))
        s1=M@state
        tst=np.diag(np.transpose(s1).flatten())
        ha = entropy(ptr(tst,[2]*2,0))

        mis.append(ha-hainit)
    
    if np.mean(mis)>maxmi:
        maxmi = np.mean(mis)
        maxm=M
    arrc3.append(np.mean(mis))

plt.hist(arrc3,bins=30,density=True)
plt.title("Classical bi-partite Entropy")
#%%
bins = np.linspace(-0.5, 0.5, 100)
plt.hist(arr, bins, alpha=0.5, label='Unitary',density=True)
plt.hist(arrc, bins, alpha=0.5, label='Markov',density=True)
# plt.hist(arrc3, bins, alpha=0.5, label='Markov_cross',density=True)

plt.legend(loc='upper right')
plt.title("Delta Bi-partite Entropy after One Operation Random Initial State")

plt.show()
#%% classical MI rand
arrc= []
maxmi = 0
maxm = []
for i in range(10000):
    
    
    # mi1 = mutinf_subsys(np.array(state),dims=[2]*2,sysa=[0],sysb=[1])
    M =np.array(markov_dir(0.1))#np.array([[1,0.4,0.4,0],[0,0.3,0.3,0],[0,0.3,0.3,0],[0,0,0,1]])
    mis=[]
    for i in range(40):
        # state=[0.,0,0,0.]
        # state[i]=1
        state=np.random.dirichlet(np.ones(4),size=1)[0]
        tst=np.diag(np.transpose(state).flatten())
        hai = entropy(ptr(tst,[2]*2,0))
        hbi = entropy(ptr(tst,[2]*2,1))
        habi=entropy(tst)
        s1=M.T@state
        tst=np.diag(np.transpose(s1).flatten())
        ha = entropy(ptr(tst,[2]*2,0))
        hb = entropy(ptr(tst,[2]*2,1))
        hab=entropy(tst)
        
        mi2 = mutinf(tst,sysa=0)#ha+hb-hab#mutinf(tst,dims=[2]*2)#mutinf_subsys(s1,dims=[2]*2,sysa=[0],sysb=[1])
        mis.append(ha+hb-hab +habi -hai-hbi)
    
    if np.mean(mis)>maxmi:
        maxmi = np.mean(mis)
        maxm=M
    arrc.append(np.mean(mis))

plt.hist(arrc,bins=30,density=True)
plt.title("Classical bi-partite MI")
#%% quantum MI rand
arr= []
maxhi = 0
maxh = []
for i in range(10000):
    
    
    # mi1 = mutinf_subsys(np.array(state),dims=[2]*2,sysa=[0],sysb=[1])
    M =rand_uni(4)
    mis=[]
    for i in range(40):
        # state=[0.,0,0,0.]
        # state[i]=1
        s=rand_ket(4)
        hai = entropy_subsys(s,[2]*2,np.array([0]))
        hbi = entropy_subsys(s,[2]*2,[1])

        habi = entropy_subsys(s,[2]*2,[0,1])

        s1=M@s
        # tst=np.diag(np.transpose(s1).flatten())
        ha = entropy_subsys(s1,[2]*2,[0])
        hb = entropy_subsys(s1,[2]*2,[1])

        hab = entropy_subsys(s1,[2]*2,[0,1])

        mis.append(ha+hb-hab-(hai+hbi-habi))
    
    if np.mean(mis)>maxhi:
        maxhi = np.mean(mis)
        maxh=M
    arr.append(np.mean(mis))

plt.hist(arr,bins=50,density=True)
plt.title("Quantum bi-partite MI")
#%% classical MI cross rand
arrc3= []
maxmi = 0
maxm = []
for i in range(10000):
    
    
    # mi1 = mutinf_subsys(np.array(state),dims=[2]*2,sysa=[0],sysb=[1])
    M =np.array(markove(np.random.rand()))#np.array([[1,0.4,0.4,0],[0,0.3,0.3,0],[0,0.3,0.3,0],[0,0,0,1]])
    mis=[]
    for i in range(40):
        # state=[0.,0,0,0.]
        # state[i]=1
        state=np.random.dirichlet(np.ones(4),size=1)[0]
        tst=np.diag(np.transpose(state).flatten())
        hai = entropy(ptr(tst,[2]*2,0))
        hbi = entropy(ptr(tst,[2]*2,1))
        habi=entropy(tst)
        s1=M.T@state
        tst=np.diag(np.transpose(s1).flatten())
        ha = entropy(ptr(tst,[2]*2,0))
        hb = entropy(ptr(tst,[2]*2,1))
        hab=entropy(tst)
        
        mi2 = mutinf(tst,sysa=0)#ha+hb-hab#mutinf(tst,dims=[2]*2)#mutinf_subsys(s1,dims=[2]*2,sysa=[0],sysb=[1])
        mis.append(ha+hb-hab +habi -hai-hbi)
    
    if np.mean(mis)>maxmi:
        maxmi = np.mean(mis)
        maxm=M
    arrc3.append(np.mean(mis))

plt.hist(arrc3,bins=30,density=True)
plt.title("Classical bi-partite MI")
#%%
#%%
bins = np.linspace(-1, 1, 100)
plt.hist(arr, bins, alpha=0.5, label='Unitary',density=True)
plt.hist(arrc, bins, alpha=0.5, label='Markov',density=True)
# plt.hist(arrc3, bins, alpha=0.5, label='Markov_cross',density=True)

plt.legend(loc='upper right')
plt.title("Delta Bi-partite MI after One Operation Random Initial State")

plt.show()
#%%

x=[]
y=[]
for i in range(100):
    diri = np.random.dirichlet([1,1],size=(1))[0]
    x.append(diri[0])
    y.append(diri[1])
plt.plot(x,y,'.')
#%%
#%% classical entropy
arrc= []
maxmi = 0
maxm = []
for i in range(100000):
    
    
    # mi1 = mutinf_subsys(np.array(state),dims=[2]*2,sysa=[0],sysb=[1])
    M =np.array(markov_dir(0.1))#np.array([[1,0.4,0.4,0],[0,0.3,0.3,0],[0,0.3,0.3,0],[0,0,0,1]])
    mis=[]
    for i in range(4):
        state=[0.,0,0,0.]
        state[i]=1
        # state=np.random.dirichlet(np.ones(4),size=1)[0]
        tst=np.diag(np.transpose(state).flatten())
        hainit = entropy(ptr(tst,[2]*2,0))
        s1=M.T@state
        tst=np.diag(np.transpose(s1).flatten())
        ha = entropy(ptr(tst,[2]*2,0))

        mis.append(ha-hainit)
    
    if np.mean(mis)>maxmi:
        maxmi = np.mean(mis)
        maxm=M
    arrc.append(np.mean(mis))

plt.hist(arrc,bins=30,density=True)
plt.title("Classical bi-partite Entropy")
#%% quantum entropy
arr= []
maxhi = 0
maxh = []
for i in range(100000):
    
    
    # mi1 = mutinf_subsys(np.array(state),dims=[2]*2,sysa=[0],sysb=[1])
    M =rand_uni(4)
    mis=[]
    for i in range(4):
        s=[0.,0.,0.,0.]
        s[i]=1
        s=qu(s,qtype='ket')
        # s=rand_ket(4)
        hinit = entropy_subsys(s,[2]*2,np.array([0]))
        s1=M@s
        # tst=np.diag(np.transpose(s1).flatten())
        ha = entropy_subsys(s1,[2]*2,[0])
        # miab = mutinf_subsys(s1,[2]*2,0,1)
        
        # mi2 = miab#mutinf(tst,dims=[2]*2)#mutinf_subsys(s1,dims=[2]*2,sysa=[0],sysb=[1])
        mis.append(ha-hinit)
    
    if np.mean(mis)>maxhi:
        maxhi = np.mean(mis)
        maxh=M
    arr.append(np.mean(mis))

plt.hist(arr,bins=50,density=True)
plt.title("Quantum bi-partite Entropy")
#%%

#%% classical entropy
arrc3= []
maxmi = 0
maxm = []
for i in range(10000):
    
    
    # mi1 = mutinf_subsys(np.array(state),dims=[2]*2,sysa=[0],sysb=[1])
    M =np.array(markove(np.random.rand()))#np.array([[1,0.4,0.4,0],[0,0.3,0.3,0],[0,0.3,0.3,0],[0,0,0,1]])
    mis=[]
    for i in range(4):
        state=[0.,0,0,0.]
        state[i]=1
        # state=np.random.dirichlet(np.ones(4),size=1)[0]
        tst=np.diag(np.transpose(state).flatten())
        hainit = entropy(ptr(tst,[2]*2,0))
        s1=M.T@state
        tst=np.diag(np.transpose(s1).flatten())
        ha = entropy(ptr(tst,[2]*2,0))

        mis.append(ha-hainit)
    
    if np.mean(mis)>maxmi:
        maxmi = np.mean(mis)
        maxm=M
    arrc3.append(np.mean(mis))

plt.hist(arrc2,bins=30,density=True)
plt.title("Classical bi-partite Entropy")
#%%
bins = np.linspace(-0.5, 1, 100)
plt.hist(arr, bins, alpha=0.5, label='Unitary',density=True)
plt.hist(arrc, bins, alpha=0.5, label='Markov',density=True)
plt.hist(arrc2, bins, alpha=0.5, label='Markov_bias',density=True)
plt.hist(arrc3, bins, alpha=0.5, label='Markov_cross',density=True)

plt.legend(loc='upper left')
plt.title("Delta Bi-partite Entropy after One Operation")

plt.show()
#%%
#%% classical MI
arrc= []
maxmi = 0
maxm = []
for i in range(10000):
    
    
    # mi1 = mutinf_subsys(np.array(state),dims=[2]*2,sysa=[0],sysb=[1])
    M =np.array(markov_dir(0.1))#np.array([[1,0.4,0.4,0],[0,0.3,0.3,0],[0,0.3,0.3,0],[0,0,0,1]])
    mis=[]
    for i in range(4):
        state=[0.,0,0,0.]
        state[i]=1
        # state=np.random.dirichlet(np.ones(4),size=1)[0]
        tst=np.diag(np.transpose(state).flatten())
        hai = entropy(ptr(tst,[2]*2,0))
        hbi = entropy(ptr(tst,[2]*2,1))
        habi = entropy(ptr(tst,[2]*2,[0,1]))

        s1=M.T@state
        tst=np.diag(np.transpose(s1).flatten())
        ha = entropy(ptr(tst,[2]*2,0))
        hb = entropy(ptr(tst,[2]*2,1))
        hab = entropy(ptr(tst,[2]*2,[0,1]))
        mis.append(ha+hb-hab-(hai+hbi-habi))
    
    if np.mean(mis)>maxmi:
        maxmi = np.mean(mis)
        maxm=M
    arrc.append(np.mean(mis))

plt.hist(arrc,bins=30,density=True)
plt.title("Classical bi-partite MI")
#%% quantum MI
arr= []
maxhi = 0
maxh = []
for i in range(10000):
    
    
    # mi1 = mutinf_subsys(np.array(state),dims=[2]*2,sysa=[0],sysb=[1])
    M =rand_uni(4)
    mis=[]
    for i in range(4):
        s=[0.,0.,0.,0.]
        s[i]=1
        s=qu(s,qtype='ket')
        # s=rand_ket(4)
        hai = entropy_subsys(s,[2]*2,np.array([0]))
        hbi = entropy_subsys(s,[2]*2,[1])

        habi = entropy_subsys(s,[2]*2,[0,1])

        s1=M@s
        # tst=np.diag(np.transpose(s1).flatten())
        ha = entropy_subsys(s1,[2]*2,[0])
        hb = entropy_subsys(s1,[2]*2,[1])

        hab = entropy_subsys(s1,[2]*2,[0,1])

        # miab = mutinf_subsys(s1,[2]*2,0,1)
        
        # mi2 = miab#mutinf(tst,dims=[2]*2)#mutinf_subsys(s1,dims=[2]*2,sysa=[0],sysb=[1])
        mis.append(ha+hb-hab-(hai+hbi-habi))
    
    if np.mean(mis)>maxhi:
        maxhi = np.mean(mis)
        maxh=M
    arr.append(np.mean(mis))

plt.hist(arr,bins=50,density=True)
plt.title("Quantum bi-partite Mi")
#%%

#%% classical entropy
arrc3= []
maxmi = 0
maxm = []
for i in range(10000):
    
    
    # mi1 = mutinf_subsys(np.array(state),dims=[2]*2,sysa=[0],sysb=[1])
    M =np.array(markove(np.random.rand()))#np.array([[1,0.4,0.4,0],[0,0.3,0.3,0],[0,0.3,0.3,0],[0,0,0,1]])
    mis=[]
    for i in range(4):
        state=[0.,0,0,0.]
        state[i]=1
        # state=np.random.dirichlet(np.ones(4),size=1)[0]
        tst=np.diag(np.transpose(state).flatten())
        hai = entropy(ptr(tst,[2]*2,0))
        hbi = entropy(ptr(tst,[2]*2,1))
        habi = entropy(ptr(tst,[2]*2,[0,1]))

        s1=M.T@state
        tst=np.diag(np.transpose(s1).flatten())
        ha = entropy(ptr(tst,[2]*2,0))
        hb = entropy(ptr(tst,[2]*2,1))
        hab = entropy(ptr(tst,[2]*2,[0,1]))
        mis.append(ha+hb-hab-(hai+hbi-habi))
    
    if np.mean(mis)>maxmi:
        maxmi = np.mean(mis)
        maxm=M
    arrc3.append(np.mean(mis))


plt.hist(arrc3,bins=30,density=True)
plt.title("Classical bi-partite MI")
#%%
bins = np.linspace(-0., 2, 100)
plt.hist(arr, bins, alpha=0.5, label='Unitary',density=True)
plt.hist(arrc, bins, alpha=0.5, label='Markov',density=True)
# plt.hist(arrc2, bins, alpha=0.5, label='Markov_bias',density=True)
plt.hist(arrc3, bins, alpha=0.5, label='Markov_cross',density=True)

plt.legend(loc='upper right')
plt.title("Delta Bi-partite MI after One Operation")

plt.show()
#%%
