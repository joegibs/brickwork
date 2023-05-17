# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 17:29:18 2023

@author: jogib
"""
import numpy as np
from quimb import *
import quimb.tensor as qtn
import matplotlib.pyplot as plt
#%%
psa = [[[0,1],[2,3]],[[1,2]]]

# def do_operation(state, gate, ps):
#     pairs = [tuple(i) for i in ps]
#     for p in pairs:
#         A = pkron(kron(gate),dims=[2]*4,inds=np.array(p).flatten())
#     return A@state
def do_operation(state, gate, ps):
    pairs = [tuple(i) for i in ps]
    
    ops=[gate for i in ps]
    A = pkron(kron(*ops),dims=[2]*4,inds=np.array(ps).flatten())
    return A@state

def measure_s(state, inds):
    sts=[]
    tst = np.diag(state.flatten())
    for i in range(4):
        if i in inds:
            ops = pauli("Z")
            states = ptr(tst,[2]*4,i)
            _,newi=measure(states,ops)
            sts.append(newi)
        else: 
            state_n = ptr(tst,[2]*4,i)
            sts.append(state_n)

    arr = np.diag(kron(*sts)).real
    return arr
#%%


M=np.array([[1,0.8,0.8,0],[0,0.1,0.1,0],[0,0.1,0.1,0],[0,0,0,1]])
s=computational_state("1110").real
s_history=[s]
              
i=0
steps=8
while i<steps/2:
    j=0
    for ps in psa:
        s = do_operation(s,M,ps)
        if i==1 and j ==0:
            print('hey')
            s = measure_s(s,[1])
        j+=1
        s_history.append(s)
    i+=1
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
        
#%%%
ent = [entropy_subsys(np.diag(np.transpose(i).flatten()),[2]*4,[0,1]) for i in s_history]

#%%
import brickwork.circuit_TN as bc
import brickwork.circuit_class as bc


import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

#%% Von nueman
sits =[4,6,8]
tot_vonq=[]
tot_bipq = []
for Sites in sits:
    arr_vonq=[]
    arr_bipq=[]

    
    interval =np.linspace(0.,1,10) 
    num_samples = 20
    eps=0.1
    gate="markovsing"
    
    start=time.time()
    for i in tqdm(interval):
        # print(i)
        numstep = 2*Sites
    
        
        circ = bc.circuit(Sites, numstep, meas_r=float(i), gate=gate, architecture="brick")
        circ.do_step(num=numstep, rec="vonbip")
        # vonq_avg = [(circ.rec_ent[i]+circ.rec_ent[i+1])/2 for i in np.arange(0,len(circ.rec_ent)-1,2)]
        # data_von = vonq_avg
        data_von=[circ.rec_ent]
        data_bip=[circ.bipent(int(Sites/2))]

        
        for j in range(1,num_samples):
            circ = bc.circuit(Sites, numstep, meas_r=float(i), gate=gate, architecture="brick")
            circ.do_step(num=numstep, rec="vonbip")
            vonq_avg = [circ.rec_ent]#[(circ.rec_ent[i]+circ.rec_ent[i+1])/2 for i in np.arange(0,len(circ.rec_ent)-1,2)]
            data_von = np.average(np.array([data_von, vonq_avg]), axis=0,weights=[j,1] )
            bipq_avg = [circ.bipent(int(Sites/2))]
            data_bip = np.average(np.array([data_bip, bipq_avg]), axis=0,weights=[j,1] )

        
        arr_vonq.append(data_von)
        arr_bipq.append(data_bip)

    end=time.time()
    tot_vonq.append([x[-1][-1] for x in arr_vonq])
    tot_bipq.append([x[0] for x in arr_bipq])

#%%
sits=[4,6,8,6,4,6,4,6]
fig, ax = plt.subplots()
for i in tot_bipq:
    ax.plot(interval,i)
ax.legend(title='Size',labels=np.round(sits,3))
plt.title(f"Bip mut inf, Gate:{gate}")

plt.show()
#%%
fig, ax = plt.subplots()
for i in arr_vonq:
    ax.plot(np.transpose(i))
ax.legend(title='meas_p',labels=np.round(interval,3))
plt.title(f"Entropy, Gate:{gate}, Sites:{Sites}, Samples:{num_samples}, Time={np.round(end-start,3)}")

plt.show()
#%%
sits=[4,6,4,6,4,6,4,6]
fig, ax = plt.subplots()
for i in tot_vonq:
    ax.plot(interval,i)
ax.legend(title='Size',labels=np.round(sits,3))
plt.title(f"Entropy, Gate:{gate}, Sites:{Sites}, Samples:{num_samples}, Time={np.round(end-start,3)}")

plt.show()

#%%
import brickwork.circuit_TN as bc
import brickwork.circuit_class as bc
#%% sep_mut
sits =[4,6]
# tot_vonq=[]
for Sites in sits:
    arr_mutq=[]

    
    interval =[0.1,0.4,0.6]#np.linspace(0.,1,10) 
    num_samples = 20
    eps=0.1
    gate="markov"
    
    start=time.time()
    for i in tqdm(interval):
        # print(i)
        numstep = 2*Sites
    
        
        circ = bc.circuit(Sites, numstep, meas_r=float(i), gate=gate, architecture="brick")
        circ.do_step(num=numstep, rec="sep_mut")
        # vonq_avg = [(circ.rec_ent[i]+circ.rec_ent[i+1])/2 for i in np.arange(0,len(circ.rec_ent)-1,2)]
        # data_von = vonq_avg
        data_mut=circ.rec_sep_mut

        
        for j in range(1,num_samples):
            circ = bc.circuit(Sites, numstep, meas_r=float(i), gate=gate, architecture="brick")
            circ.do_step(num=numstep, rec="sep_mut")
            mutq_avg = circ.rec_sep_mut#[(circ.rec_ent[i]+circ.rec_ent[i+1])/2 for i in np.arange(0,len(circ.rec_ent)-1,2)]
            data_mut = np.average(np.array([data_mut, mutq_avg]), axis=0,weights=[j,1] )

        
        arr_mutq.append(data_mut)

    end=time.time()
    tot_vonq.append([x[-1][-1] for x in arr_vonq])
    #%%
    fig, ax = plt.subplots()
    # ax.plot(np.transpose([[arr_mut[i][j] for j in range(0,np.shape(arr_mut)[1],2)] for i in range(np.shape(arr_mut)[0])]))
    # ax.plot(np.transpose([[arr_mutq[i][j] for j in range(0,np.shape(arr_mutq)[1],2)] for i in range(np.shape(arr_mutq)[0])]))
    ax.plot([2,3,4,5,6],np.transpose([arr.flatten() for arr in arr_mutq]))
    # ax.set_ybound(lower=10**-5)
    ax.legend(title='meas_r',labels=np.round(interval,3))
    plt.title(f"Mut_info, Gate:{gate}, Sites:{Sites}, Samples:{num_samples}, Time={np.round(end-start,3)}")

    plt.show()