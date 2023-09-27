# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 11:49:18 2023

@author: jogib
"""

import numpy as np
from quimb import *
import quimb.tensor as qtn
import matplotlib.pyplot as plt

#%%
p = qtn.MPS_rand_state(L=4, bond_dim=2)

#%%
tst = p.H&p
tst2 =p.combine(p.H)

#%%
p.H.to_dense()
np.flatten(np.transpose(p.to_dense()))

#%%
tst = kron(np.transpose(p.to_dense()),p.to_dense())
#%%
for i in range(3):
    print(kron(p.arrays[i],p.H.arrays[i]))
#%%
p.add_tag('KET')
q = p.H.retag({'KET': 'BRA'})
qp = q & p

#%% heyo this is stupid but it works
tst = p.partial_trace(range(0,2))

#%% This also works
arr = []
for i,t in enumerate(tst):
    print(t)
    a = t.fuse({f'k{i}':[f'k{i}',f'b{i}']})
    arr.append(a)
TN = qtn.TensorNetwork(arr)
fin = qtn.MatrixProductState(TN.arrays)
#%% note entropy calc is weird get double
entropy_subsys(tst.to_dense(),[2]*4,[0,1])
print(p.entropy(2)) #mps is same as mpo

print(fin.entropy(2)) #lindblad space is approx double
#%%
operator = rand_uni(4)

superop = kron(operator, identity(4)) - kron(identity(4) ,operator.T)
#%% apply gate
p.gate_split(operator,(1,2),inplace=True)
print(p.entropy(2))
# tst.
fin.gate_split(superop,(1,2),inplace=True)
print(fin.entropy(2))
#entropy gets weirder

#%% reverse reverse

rev = qtn.TensorNetwork([i for i in fin])
arr=[]
for i,t in enumerate(rev):
    a = t.unfuse({f'k{i}':[f'k{i}',f'b{i}']},{f'k{i}':(2,2)})
    arr.append(a)
TN = qtn.TensorNetwork(arr)
fin = qtn.MatrixProductOperator(TN.arrays)

    
    
#%% full check/ functions this works
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
def from_lindblad_space(fin):
    rev = qtn.TensorNetwork([i for i in fin])
    arr=[]
    for i,t in enumerate(rev):
        a = t.unfuse({f'k{i}':[f'k{i}',f'b{i}']},{f'k{i}':(2,2)})
        arr.append(a)
    TN = qtn.TensorNetwork(arr)
    ret = qtn.MatrixProductOperator(TN.arrays)
    return ret/ret.trace()
#%% to and back from lindblad space
n=2
p = qtn.MPS_rand_state(L=n, bond_dim=2)
rho = mps_to_mpo(p)
print(rho.arrays)
rhop = to_lindblad_space(rho)
rho2 = from_lindblad_space(rhop)
print(rho2.arrays)
print(np.isclose(rho.to_dense(),rho2.to_dense()))
   #%% 
print(np.shape(rho.arrays[0]),np.shape(rho2.arrays[0]))


    
#%%
# HxI + IxHt
def H_sharp():
    h = rand_uni(2) 
    return ikron(h,dims=[2]*2,inds=[0])@ikron(h.T,dims=[2]*2,inds=[1])

#ikron(h,dims=[2]*2,inds=[0])+

#%% check bell state prep

# weird becuase it only works without the transpose

psi = qtn.MPS_computational_state('00')
g=CNOT()@pkron(hadamard(),[2]*2,[0])
psi_end = psi.gate(g,[0,1],contract='swap+split')

rho = psi.partial_trace(range(0,2))
rho_end= g@rho.to_dense()@g.H

sup_rho = to_lindblad_space(rho)
gsharp = pkron(g,dims=[2]*4,inds=[0,2])@pkron(g,[2]*4,inds=[1,3])
sup_rho_fin = sup_rho.gate(gsharp,[0,1],contract='swap+split')

# g=CNOT()
# gsharp = pkron(g,dims=[2]*4,inds=[0,2])@pkron(g,[2]*4,inds=[1,3])
# sup_rho.gate(gsharp,[0,1],contract='swap+split',inplace=True)
# rho_end= g@rho_end@g.H

check = from_lindblad_space(sup_rho_fin)
print(check.to_dense())
print(np.all(np.isclose(check.to_dense(),rho_end)))
#%% check bell state into linblad space
tst=qtn.MPS_ghz_state(2)
tst_rho=tst.partial_trace(range(0,2))
sup_tst = to_lindblad_space(tst_rho)
chk = from_lindblad_space(sup_tst)
print(np.isclose(tst_rho.to_dense(),chk.to_dense()))
#%% chec random state init
tst=qtn.MPS_rand_state(2,2)
tst_rho=tst.partial_trace(range(0,2))
sup_tst = to_lindblad_space(tst_rho)
chk = from_lindblad_space(sup_tst)
print(np.isclose(tst_rho.to_dense(),chk.to_dense()))

#%% check random gate

#this one works properly
psi = qtn.MPS_computational_state('00')
g=rand_uni(4)
psi_end = psi.gate(g,[0,1],contract='swap+split')

rho = psi.partial_trace(range(0,2))
rho_end= g@rho.to_dense()@g.H
rho_end=rho_end/trace(rho_end)

sup_rho = to_lindblad_space(rho)
gsharp = pkron(g,dims=[2]*4,inds=[0,2])@pkron(g.T,[2]*4,inds=[1,3])


sup_rho_fin = sup_rho.gate(gsharp,[0,1],contract='swap+split')

check = from_lindblad_space(sup_rho_fin)
print(np.round(check.to_dense(),2))
print((np.round(rho_end,2)))
print(np.all(np.isclose(check.to_dense(),rho_end)))

#%% this wont work because quimb doesn't like single site mps inits
a = qtn.MatrixProductState.from_dense(kron(np.array([[0.5],[0.5]]),qtn.MPS_computational_state("0").to_dense()),[2]*2)
b = mps_to_mpo(a)
c=to_lindblad_space(b)
d=from_lindblad_space(c)
#%% check for mixed states

a=qtn.MPS_computational_state('00')
a1=qtn.MPS_computational_state('01')
a2=qtn.MPS_computational_state('10')
a3=qtn.MPS_computational_state('11')


ar=mps_to_mpo(a)
a1r=mps_to_mpo(a1)
a2r=mps_to_mpo(a2)
a3r=mps_to_mpo(a3)


rho=(ar+a1r+a2r+a3r)/4 #normalized
sup_rho = to_lindblad_space(rho) #normalized
check = from_lindblad_space(sup_rho) #normalized

print(np.all(np.isclose(rho.to_dense(),check.to_dense())))

#%% check brickwork
gates=[rand_herm(4),rand_herm(4),rand_herm(4)]
gates=[np.identity(4),np.identity(4),np.identity(4)]
# gates=[CNOT(),CNOT(),rand_herm(4)]
# gates=[rand_uni(4),np.identity(4),np.identity(4)]



psa = [[0,1],[2,3],[1,2]]

psi = qtn.MPS_computational_state('0000')
rho = psi.partial_trace([0,1,2,3])
sup_rho = to_lindblad_space(rho)
rho=rho.to_dense()


def do_operation(site,g,mps):
    gate=g
    new_state=mps.gate(gate,site,contract='swap+split')
    return new_state

def do_operation_rho(site,g,rho):
    gate=pkron(g,[2]*4,site)
    new_state=gate@np.reshape(rho,[16,16])@gate.H
    return new_state

def do_operation_lindblad(site,g,mps_lind):
    gate=pkron(g,dims=[2]*4,inds=[0,2])@pkron(g.T,[2]*4,inds=[1,3])
    new_state=mps_lind.gate(gate,site,contract='swap+split')
    return new_state

for i,site in enumerate(psa):
    psi = do_operation(site,gates[i],psi)
    rho=do_operation_rho(site,gates[i],rho)
    rho=rho/trace(rho)
    sup_rho = do_operation_lindblad(site,gates[i],sup_rho)

fin = from_lindblad_space(sup_rho)
print(np.all(np.isclose(rho,fin.to_dense())))
a=psi.ptr([0,1,2,3]).to_dense()
a=np.round(a,11)
print(trace(a))
a=a/(trace(a))       
print(np.all(np.isclose(a,rho)))
#%% recheck bell_state_prep ok this works? maybe something about the gates idk
gates=[ikron(hadamard(),[2]*2,0),CNOT()]
psa = [[0,1],[0,1]]

psi = qtn.MPS_computational_state('00')
rho = psi.partial_trace([0,1])
sup_rho = to_lindblad_space(rho)
rho=rho.to_dense()


def do_operation(site,g,mps):
    gate=g
    new_state=mps.gate(gate,site,contract='swap+split')
    return new_state

def do_operation_rho(site,g,rho):
    gate=pkron(g,[2]*2,site)
    new_state=gate@np.reshape(rho,[4,4])@gate.H
    return new_state

def do_operation_lindblad(site,g,mps_lind):
    gate=pkron(g,dims=[2]*4,inds=[0,2])@pkron(g.T,[2]*4,inds=[1,3])
    new_state=mps_lind.gate(gate,site,contract='swap+split')
    return new_state

for i,site in enumerate(psa):
    psi = do_operation(site,gates[i],psi)
    rho=do_operation_rho(site,gates[i],rho)
    rho=rho/trace(rho)
    sup_rho = do_operation_lindblad(site,gates[i],sup_rho)

fin = from_lindblad_space(sup_rho)
print(np.all(np.isclose(rho,fin.to_dense())))
a=psi.ptr([0,1,2,3])
a=kron(psi.to_dense(),np.transpose(np.conjugate(psi.to_dense())))
print(np.all(np.isclose(a,rho)))
#%%entropy test
def ent_sharp(mps,dims):
    tst = from_lindblad_space(mps)
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
#%%
#%% chec random state entropy
tst=qtn.MPS_rand_state(2,2)
tst_rho=tst.partial_trace(range(0,2))
sup_tst = to_lindblad_space(tst_rho)
chk = from_lindblad_space(sup_tst)
print(ent_sharp(sup_tst,[2]*2))
print(ent(tst))
print(np.isclose(tst_rho.to_dense(),chk.to_dense()))

#%% bell state entropy check
gates=[ikron(hadamard(),[2]*2,0),CNOT()]
dims=[2]*2
psa = [[0,1],[0,1]]

psi = qtn.MPS_computational_state('00')
rho = psi.partial_trace([0,1])
sup_rho = to_lindblad_space(rho)
rho=rho.to_dense()


def do_operation(site,g,mps):
    gate=g
    new_state=mps.gate(gate,site,contract='swap+split')
    return new_state

def do_operation_rho(site,g,rho):
    gate=pkron(g,[2]*2,site)
    new_state=gate@np.reshape(rho,[4,4])@gate.H
    return new_state

def do_operation_lindblad(site,g,mps_lind):
    gate=pkron(g,dims=[2]*4,inds=[0,2])@pkron(g.T,[2]*4,inds=[1,3])
    new_state=mps_lind.gate(gate,site,contract='swap+split')
    return new_state

for i,site in enumerate(psa):
    psi = do_operation(site,gates[i],psi)
    rho=do_operation_rho(site,gates[i],rho)
    rho=rho/trace(rho)
    sup_rho = do_operation_lindblad(site,gates[i],sup_rho)

print(np.isclose(ent_rho(rho,dims),ent(psi)))
print(np.isclose(ent_sharp(sup_rho,dims),ent(psi)))
#%% double check transpose commutes across the exp
from scipy.linalg import expm
m = rand_uni(2)
mt = m.T
exm = expm(m)
exmt = expm(mt)
print(np.isclose(np.transpose(exm),exmt))
#%%#%% rand state entropy check
gates=[rand_uni(4)]
dims=[2]*2
psa = [[0,1]]

psi = qtn.MPS_computational_state('00')
rho = psi.partial_trace([0,1])
sup_rho = to_lindblad_space(rho)
rho=rho.to_dense()


def do_operation(site,g,mps):
    gate=g
    new_state=mps.gate(gate,site,contract='swap+split')
    return new_state

def do_operation_rho(site,g,rho):
    gate=pkron(g,[2]*2,site)
    new_state=gate@np.reshape(rho,[4,4])@gate.H
    return new_state

def do_operation_lindblad(site,g,mps_lind):
    gate=pkron(g,dims=[2]*4,inds=[0,1])@pkron(g.T,[2]*4,inds=[2,3])
    new_state=mps_lind.gate(gate,site,contract='swap+split')
    return new_state

for i,site in enumerate(psa):
    psi = do_operation(site,gates[i],psi)
    rho=do_operation_rho(site,gates[i],rho)
    rho=rho/trace(rho)
    sup_rho = do_operation_lindblad(site,gates[i],sup_rho)

fin = from_lindblad_space(sup_rho)
print(np.round(fin.to_dense(),2),'\n',np.round(rho/trace(rho),2))
print(ent_sharp(sup_rho,dims),ent(psi),ent_rho(rho,dims))
print(np.isclose(ent_rho(rho,dims),ent(psi)))
print(np.isclose(ent_sharp(sup_rho,dims),ent(psi)))
#%% blehh this isnt being nice lets just try tebd

L = 44
zeros = '0' * ((L - 2) // 3)
binary = zeros + '1' + zeros + '1' + zeros
print('psi0:', f"|{binary}>")
psi0 = qtn.MPS_computational_state(binary)
psi0.show() 
H = qtn.ham_1d_heis(L)
tebd = qtn.TEBD(psi0, H)

# Since entanglement will not grow too much, we can set quite
#     a small cutoff for splitting after each gate application
tebd.split_opts['cutoff'] = 1e-12

# times we are interested in
ts = np.linspace(0, 80, 101)

be_t_b = []  # block entropy

# range of bonds, and sites
js = np.arange(0, L)
bs = np.arange(1, L)
# generate the state at each time in ts
#     and target error 1e-3 for whole evolution
for psit in tebd.at_times(ts, tol=1e-3):
    be_b = []
    
    # there is one more site than bond, so start with mag
    #     this also sets the orthog center to 0
    be_t_b.append(psit.entropy(int(L/2)))
  
tebd.err  #  should be < tol=1e-3

H = qtn.MPO_ham_heis(L)
print("Initial energy:", qtn.expec_TN_1D(psi0.H, H, psi0))
print("Final energy:", qtn.expec_TN_1D(tebd.pt.H , H, tebd.pt))
#%%
L=6
psi0 = qtn.MPS_computational_state('000100')

builder = qtn.SpinHam1D(S=1/2)

# specify the interaction term (defaults to all sites)
builder += 1, 'Z', 'Z'
builder += 1.0, 'X'

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

#%% yolo attempt
#build state
L=6
psir = qtn.MPS_computational_state('000100')
# psir = qtn.MPS_computational_state('100001')
rhor = psir.ptr([0,1,2,3,4,5])
psi0=to_lindblad_space(rhor)

builder = qtn.SpinHam1D(S=3/2)
#hmmmmmm
j=1
builder += 1/4, kron(pauli('Z'),np.identity(2)),kron(pauli('Z'),np.identity(2))
builder += -1/4, kron(np.identity(2),pauli('Z').T),kron(np.identity(2),pauli('Z').T)

builder += 1/2, kron(pauli('X'),np.identity(2))
builder += -1/2, kron(np.identity(2),pauli('X').T)

H=builder.build_local_ham(L)

tebd = qtn.TEBD(psi0, H)

# Since entanglement will not grow too much, we can set quite
#     a small cutoff for splitting after each gate application
tebd.split_opts['cutoff'] = 1e-12

# times we are interested in
ts = np.linspace(0, 10, 101)

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
plt.plot(ts,be_t_b,ts,be_t_b2,ts,be_t_b0)


#%% M check
L=2
psir = qtn.MPS_computational_state('00')
# psir = qtn.MPS_computational_state('100001')
rhor = psir.ptr([0,1])
psi0=to_lindblad_space(rhor)

builder = qtn.SpinHam1D(S=3/2)
#hmmmmmm
builder += 1, kron(pauli('X'),np.identity(2)),kron(pauli('X'),np.identity(2))
builder += -1, kron(np.identity(2),pauli('X').T),kron(np.identity(2),pauli('X').T)
# builder += 1, kron(pauli('Z'),np.identity(2))
# builder += -1, kron(np.identity(2),pauli('Z').T)

H=builder.build_local_ham(L)


arr_vhk=H.get_gate([0,1])

#%%
L=2
psi0 = qtn.MPS_computational_state('0000')

builder = qtn.SpinHam1D(S=1/2)

# specify the interaction term (defaults to all sites)
builder += 1/2, 'X', 'X'
# builder += -1.0, 'Z'

# add random z-fields to each site
# np.random.seed(2)
# for i in range(L):
#     builder[i] += 2 * np.random.rand() - 1, 'Z'
    
H = builder.build_local_ham(L)
arr_hk=H.get_gate([0,1])
#%%
j=1
a=kron(kron(-j*pauli('X'),np.identity(2)),kron(pauli('X'),np.identity(2)))
b = kron(-kron(np.identity(2),-j*pauli('X').T),kron(np.identity(2),pauli('X').T))
c=(a+b).real

#%%
chk = qtn.MPO_ham_ising(2,j=1,bx=1)
chk = qtn.MPO_ham_heis(2,[-1,0,0],bz=-1)
#%%
#%% add a lindblad operator yolo
#build state
L=6
psir = qtn.MPS_computational_state('000100')
# psir = qtn.MPS_computational_state('100001')
rhor = psir.ptr([0,1,2,3,4,5])
psi0=to_lindblad_space(rhor)

builder = qtn.SpinHam1D(S=3/2)
#hmmmmmm
j=1
builder += 1/4, kron(pauli('Z'),np.identity(2)),kron(pauli('Z'),np.identity(2))
builder += -1/4, kron(np.identity(2),pauli('Z').T),kron(np.identity(2),pauli('Z').T)

builder += 1/2, kron(-pauli('X'),np.identity(2))
builder += -1/2, kron(np.identity(2),-pauli('X').T)

#dephase term
# gamma=0.5
# pZ = pauli('Z')
# builder += -1/2*gamma, kron(pZ,pZ.H.T)
# builder += 1/4*gamma, kron(pZ.H@pZ,np.identity(2))
# builder += 1/4*gamma, kron(np.identity(2),(pZ.H@pZ).T)


H=builder.build_local_ham(L)

tebd = qtn.TEBD(psi0, H)

# Since entanglement will not grow too much, we can set quite
#     a small cutoff for splitting after each gate application
tebd.split_opts['cutoff'] = 1e-12

# times we are interested in
ts = np.linspace(0, 100, 201)

ne_t_b = []  # block negativity
me_t_b = []

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
    ne_t_b.append(neg_sharp(psit,[2]*L))
    me_t_b.append(mut_sharp(psit,[2]*L))


    # be_t_b.append(psit.entropy(int(L/2)))

  
tebd.err  #  should be < tol=1e-
# plt.plot(ts,be_t_b,ts,be_t_b2,ts,be_t_b0,ts,ne_t_b,ts,me_t_b)
plt.plot(ts,be_t_b,ts,be_t_b2,ts,ne_t_b)
#%% M check
L=2
psir = qtn.MPS_computational_state('00')
# psir = qtn.MPS_computational_state('100001')
rhor = psir.ptr([0,1])
psi0=to_lindblad_space(rhor)

builder = qtn.SpinHam1D(S=3/2)
#hmmmmmm
builder += 1/4, kron(pauli('Z'),np.identity(2)),kron(pauli('Z'),np.identity(2))
builder += -1/4, kron(np.identity(2),pauli('Z').T),kron(np.identity(2),pauli('Z').T)

builder += 1/2, kron(-pauli('X'),np.identity(2))
builder += -1/2, kron(np.identity(2),-pauli('X').T)
gamma=0.5j
pZ = pauli('Z')
builder += 1/2*gamma, kron(pZ,pZ.H.T)
builder += -1/4*gamma, kron(pZ.H@pZ,np.identity(2))
builder += -1/4*gamma, kron(np.identity(2),(pZ.H@pZ).T)
H=builder.build_local_ham(L)


arr_vhk=H.get_gate([0,1])
