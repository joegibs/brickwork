#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 11:16:53 2023

@author: joeg
"""

import numpy as np
from quimb import *
import quimb.tensor as qtn
import matplotlib.pyplot as plt
from itertools import product

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
    return logneg(tst.to_dense(),dims,[i for i in range(int(L/2))])
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
a=product("01",repeat=2)
tst = [mps_to_mpo(qtn.MPS_computational_state("".join([str(j) for j in i]))) for i in a]
# tst1 = mps_to_mpo(qtn.MPS_computational_state('01'))
# tst2 = mps_to_mpo(qtn.MPS_computational_state('10'))
# tst3 = mps_to_mpo(qtn.MPS_computational_state('11'))
tst_mix=np.sum(tst)
tst_mix=tst_mix/tst_mix.trace()
#%%
tst = qtn.MPS_computational_state('00')
tst_bell = qtn.MatrixProductState.from_dense(bell_state(2),[2]*2)

print('00 entropy',tst.entropy(1))
print('bell entropy',tst_bell.entropy(1))
print('mix entropy ',entropy_subsys(tst_mix.to_dense(),[2]*2,[0]))
print('00 neg',tst.logneg_subsys([0],[1]))
print('bell neg',tst_bell.logneg_subsys([0],[1]))
print('mix neg ',logneg(tst_mix.to_dense(),[2]*2,[0]))
#%% ok now do this from vec

lin_mix = to_lindblad_space(tst_mix)
lin_00 = to_lindblad_space(mps_to_mpo(tst))
lin_bell = to_lindblad_space(mps_to_mpo(tst_bell))

print("lin 00 ent ", ent_sharp(lin_00,[2]*2))
print("lin_mix ent ",ent_sharp(lin_mix,[2]*2))
print("lin_bell ent ", ent_sharp(lin_bell,[2]*2))
print("lin 00 neg ", neg_sharp(lin_00,[2]*2))
print("lin_mix neg ",neg_sharp(lin_mix,[2]*2))
print("lin_bell neg ", neg_sharp(lin_bell,[2]*2))

#%% dephase get rid of bell state entropy?
L=2
mean_be=[]
# for i in np.linspace(0,1,1):
psir = qtn.MatrixProductState.from_dense(bell_state(2),[2]*2)
# psir= qtn.MPS_rand_computational_state(L)
rhor = psir.ptr([i for i in range(0,L)])
psi0=to_lindblad_space(rhor)
builder = qtn.SpinHam1D(S=3/2)

# j=1
# builder += 1/4, kron(pauli('I'),np.identity(2)),kron(pauli('I'),np.identity(2))
# builder += -1/4, kron(np.identity(2),pauli('I').T),kron(np.identity(2),pauli('I').T)
builder += 1, 'I', 'I'

# j=-1
# g=-1
# builder += 1/4, j*kron(pauli('z'),np.identity(2)),kron(pauli('z'),np.identity(2))
# builder += -1/4, j*kron(np.identity(2),pauli('z').T),kron(np.identity(2),pauli('z').T)

# builder += 1/2, g*kron(pauli('x'),np.identity(2))
# builder += -1/2, g*kron(np.identity(2),pauli('x').T)



# time for jump
#dephase term
gamma=np.sqrt(1)
pZ = pauli('Z',dim=2)

builder += 1/2*gamma, kron(pZ,pZ.H)
builder += -1/4*gamma,kron(pZ.H@pZ,np.identity(2))
builder += -1/4*gamma,kron(np.identity(2),(pZ.T@pZ.H))
# pZ = pauli('X',dim=2)
# builder += 1, kron(pZ,pZ.H.T)
# builder += -1/2, kron(pZ.H@pZ,np.identity(2))
# builder += 1/2, kron(np.identity(2),(pZ.H@pZ).T)
# H=builder.build_local_ham(L)
# pZ = pauli('Y',dim=2)
# builder += 1, kron(pZ,pZ.H.T)
# builder += -1/2, kron(pZ.H@pZ,np.identity(2))
# builder += 1/2, kron(np.identity(2),(pZ.H@pZ).T)
H=builder.build_local_ham(L)

# H = qtn.ham_1d_mbl(L=L,dh=0)
# H = qtn.ham_1d_heis(L=L)

tebd = qtn.TEBD(psi0, H,imag=False)

# Since entanglement will not grow too much, we can set quite
#     a small cutoff for splitting after each gate application
tebd.split_opts['cutoff'] = 1e-12

# times we are interested in
ts = np.linspace(0, 25, 201)

be_t_b0 = []  # block entropy
ne_t_b0 = []  # block entropy
hmm=[]
hmm_mag=[]
# generate the state at each time in ts
#     and target error 1e-3 for whole evolution
for psit in tebd.at_times(ts, tol=1e-3):
    
    # there is one more site than bond, so start with mag
    #     this also sets the orthog center to 0
    # be_t_b0.append(psit.entropy(int(L/2)))
    be_t_b0.append(ent_sharp(psit,[2]*L))
    ne_t_b0.append(neg_sharp(psit,[2]*L))
    hmm.append(from_lindblad_space(psit).to_dense()[0][-1])
    # hmm_mag.append(from_lindblad_space(psit).to_de"nse().round(2)[0][-1])


# mean_be.append(be_t_b0)
tebd.err  #  should be < tol=1e-3
# plt.plot(ts,be_t_b0)
# plt.plot(ts,ne_t_b0)

plt.plot(np.angle(hmm))
# plt.plot(ts,np.mean(mean_be,axis=0))
#%%
plt.plot(np.abs(hmm))

#%%
builder = qtn.SpinHam1D(S=3/2)

gamma=np.sqrt(1)*1j
pZ = pauli('Z',dim=2)

# builder+= 1,'z','z'
builder += 1,'I','I'
builder += 1,'z'
# builder += 1/2, kron(pZ,pZ.H)
# builder += 1/2,kron(pZ.H@pZ,np.identity(2))
# builder += -1/2,kron(np.identity(2),(pZ.T@pZ.H))

H=builder.build_local_ham(L)

print(H.terms)
print(H)
#%%
XX = pauli('X') & pauli('X')

YY = pauli('Y') & pauli('Y')

ham = qtn.LocalHam1D(L=100, H2=XX + YY)
print(H.terms)
h01=H.get_gate([0,1])/trace(H.get_gate([0,1]))
print(trace(h01@np.conjugate(h01)))
#%%

def compute(t, pt):
    """Perform computation at time ``t`` with state ``pt``.
    """
    dims = [2] * 2
    print(sum([np.abs(i[0])**2 for i in pt]))
    lns = np.abs(pt[5]/sum([np.abs(i[0])**2 for i in pt]))
    mis = np.abs(pt[0]/sum([np.abs(i[0])**2 for i in pt]))
    return t, lns, mis

h01=H.get_gate([0,1])
evo = Evolution(psi0.to_dense(), h01/trace(h01), method='integrate', compute=compute,progbar=True)
evo.update_to(200)

ts, lns, mis = zip(*evo.results)

# fig, axs = plt.subplots(2, 1, sharex=True)
# axs[0].plot(ts, lns, '-');
# axs[0].set_title("Logarithmic Negativity")
# axs[1].plot(ts, mis, '-');
# axs[1].set_title("Mutual Information")
plt.plot(ts,np.transpose(lns)[0],ts,np.transpose(mis)[0])
#%% base case
L=6
# for i in range(0,1):
# psi0 = qtn.MPS_rand_state(L,bond_dim=2,phys_dim=2)
psi0 = qtn.MPS_computational_state('000000')
# builder = qtn.SpinHam1D(S=1/2)

# # specify the interaction term (defaults to all sites)
# builder += 1, 'Z', 'Z'
# builder += 1.0, 'X'

# # add random z-fields to each site
# # np.random.seed(2)
# # for i in range(L):
# #     builder[i] += 2 * np.random.rand() - 1, 'Z'
    
# H = builder.build_local_ham(L)
builder = qtn.SpinHam1D(S=1/2)
builder += 1, 'I', 'I'
# builder += 1, 'Z'


H=builder.build_local_ham(L)

# H = qtn.ham_1d_mbl(L=L,dh=0)
# H = qtn.ham_1d_heis(L=L)

tebd = qtn.TEBD(psi0, H)

# Since entanglement will not grow too much, we can set quite
#     a small cutoff for splitting after each gate application
tebd.split_opts['cutoff'] = 1e-12

# times we are interested in
ts = np.linspace(0, 1, 2)

be_t_b0 = []  # block entropy
ne_t_b0 = []  # block entropy

# generate the state at each time in ts
#     and target error 1e-3 for whole evolution
for psit in tebd.at_times(ts, tol=1e-3):
    be_b = []
    
    # there is one more site than bond, so start with mag
    #     this also sets the orthog center to 0
    be_t_b0.append(psit.entropy(int(L/2)))
    ne_t_b0.append(psit.logneg_subsys([0,1,2],[3,4,5]))

  
tebd.err  #  should be < tol=1e-3
plt.plot(ts,be_t_b0)