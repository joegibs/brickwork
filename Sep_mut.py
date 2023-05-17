# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 17:45:28 2023

@author: jogib
"""

import brickwork.circuit_TN as bc
import brickwork.circuit_class as bc


import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

#%%
tot_sep_mut = []
#%%
sits =[6,8]
for Sites in sits:
    arr_sep_mut=[]
    
    interval =np.linspace(0.,0.8,10) 
    num_samples = 100
    eps=0.1
    gate="markov"
    
    start=time.time()
    for i in tqdm(interval):
        # print(i)
        numstep = 4*Sites
    
        
        circ = bc.circuit(Sites, numstep, meas_r=float(i), gate=gate, architecture="brick")
        circ.do_step(num=numstep, rec="sep_mut")
        data_sep_mut = circ.rec_sep_mut
        
        for j in range(1,num_samples):
            # print("j: ",j)
            circ = bc.circuit(Sites, numstep, meas_r=float(i), gate=gate, architecture="brick")
            circ.do_step(num=numstep, rec="sep_mut")
            data_sep_mut = np.average(np.array([data_sep_mut, circ.rec_sep_mut]),axis=0, weights=[j,1] )
    
        
        arr_vonq.append(data_von)
        arr_sep_mut.append(data_sep_mut)
    end=time.time()

    tot_sep_mut.append(arr_sep_mut)


#%% Mut inf at ends
x=np.linspace(1,8)
# x=interval
y = 1/(x**4)

fig, ax = plt.subplots()
for i in tot_sep_mut:
    ax.plot(np.arange(1,np.shape(i)[1]+1,1),np.transpose(i))
# ax.plot(x,y)

ax.legend(title='meas_r',labels=np.round(sits,3))
plt.title(f"Mut_info, Gate:{gate}, Sites:{Sites}, Samples:{num_samples}, Time={np.round(end-start,3)}")
ax.set_yscale('log')
# ax.set_ylim(bottom=1E-5)
# ax.set_xscale('log')


plt.show()
#%%
sits=[6,8,10,12,6,8,4,6,8]
# sits=[0.1,0.5,0.01,0.05,0.025,0.015,0.3,0.9,0.7]
fig, ax = plt.subplots()
for tri in tot_sep_mut:
    ax.plot(interval,tri)


ax.legend(title='length',labels=sits)
plt.title(f"Bip_ent, Gate:{gate}, Sites:{Sites}, Samples:{num_samples}")
# ax.set_yscale('log')

plt.show()

