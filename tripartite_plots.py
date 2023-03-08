import brickwork.circuit_TN as bc
# import brickwork.circuit_class as bc


import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

#%%
tot_tri = []
sits =[8]
for Sites in sits:
    arr_vonq=[]
    arr_sep_mut=[]
    arr_tri_mut = []

    
    interval =[0.2]#np.linspace(0.,1,6) 
    num_samples = 50
    eps=0.1
    gate="IX"
    
    start=time.time()
    for i in interval:
        print(i)
        numstep = 4*Sites
    
        
        circ = bc.circuit(Sites, numstep, meas_r=float(i), gate=gate, architecture="brick")
        circ.do_step(num=numstep, rec="von tri_mut sep_mut")
        data_von = circ.rec_ent
        data_sep_mut = circ.rec_sep_mut
        data_tri_mut = circ.rec_tri_mut
        
        for j in range(1,num_samples):
            # print("j: ",j)
            circ = bc.circuit(Sites, numstep, meas_r=float(i), gate=gate, architecture="brick")
            circ.do_step(num=numstep, rec="von tri_mut sep_mut")
            data_von = np.average(np.array([data_von, circ.rec_ent]), axis=0,weights=[j,1] )
            data_sep_mut = np.average(np.array([data_sep_mut, circ.rec_sep_mut]),axis=0, weights=[j,1] )
            data_tri_mut = np.average(np.array([data_tri_mut, circ.rec_tri_mut]),axis=0, weights=[j,1] )
    
    
        arr_vonq.append(data_von)
        arr_sep_mut.append(data_sep_mut)
        arr_tri_mut.append(data_tri_mut)
    end=time.time()
    tot_tri.append(arr_tri_mut)
#%% Mut inf at ends
sits=[8,16,8,12]
# sits=[0.1,0.5,0.01,0.05,0.025,0.015,0.3,0.9,0.7]
fig, ax = plt.subplots()
for tri in tot_tri:
    ax.plot(interval,tri)


ax.legend(title='length',labels=sits)
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
#%%
#%% Mut inf at ends
x=np.linspace(1,8)
y = 1/(x**4)

fig, ax = plt.subplots()
ax.plot(np.transpose(arr_sep_mut))
# ax.plot(x,y)

ax.legend(title='meas_r',labels=np.round(interval,3))
plt.title(f"Mut_info, Gate:{gate}, Sites:{Sites}, Samples:{num_samples}, Time={np.round(end-start,3)}")
ax.set_yscale('log')

plt.show()

