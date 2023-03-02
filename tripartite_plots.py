import brickwork.circuit_TN as bc
# import brickwork.circuit_class as bc


import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

#%%
tot_tri = []
sits =[6,8,10]
for Sites in sits:
    arr_vonq=[]
    arr_sep_mut=[]
    arr_tri_mut = []
    
    interval =np.linspace(0.1,2,8) 
    num_samples = 20
    eps=0.1
    gate="2haar"
    
    start=time.time()
    for i in tqdm(interval):
        print(i)
        numstep = 4*Sites
    
        
        circ = bc.circuit(Sites, numstep, meas_r=float(i), gate=gate, architecture="brick")
        circ.do_step(num=numstep, rec="von tri_mut")
        data_von = circ.rec_ent
        data_sep_mut = circ.rec_sep_mut
        data_tri_mut = circ.rec_tri_mut
        
        for j in range(1,num_samples):
            # print("j: ",j)
            circ = bc.circuit(Sites, numstep, meas_r=float(i), gate=gate, architecture="brick")
            circ.do_step(num=numstep, rec="von tri_mut")
            data_von = np.average(np.array([data_von, circ.rec_ent]), axis=0,weights=[j,1] )
            data_sep_mut = np.average(np.array([data_sep_mut, circ.rec_sep_mut]),axis=0, weights=[j,1] )
            data_tri_mut = np.average(np.array([data_tri_mut, circ.rec_tri_mut]),axis=0, weights=[j,1] )
    
    
        arr_vonq.append(data_von)
        arr_sep_mut.append(data_sep_mut)
        arr_tri_mut.append(data_tri_mut)
    end=time.time()
    tot_tri.append(arr_tri_mut)
#%% Mut inf at ends
sits=[16,8,12]
sits=[4,6,8,10]
fig, ax = plt.subplots()
for tri in tot_tri:
    ax.plot(interval,tri)


ax.legend(title='Length',labels=sits)
plt.title(f"Mut_info, Gate:{gate}, Sites:{Sites}, Samples:{num_samples}, Time={np.round(end-start,3)}")
# ax.set_yscale('log')

plt.show()
#%%
fig, ax = plt.subplots()

ax.plot(np.transpose(arr_vonq))


ax.legend(title='meas_p',labels=np.round(interval,3))
plt.title(f"entropy, Gate:{gate}, Sites:{Sites}, Samples:{num_samples}, Time={np.round(end-start,3)}")
# ax.set_yscale('log')

plt.show()