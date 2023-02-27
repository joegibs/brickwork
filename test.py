
import brickwork.circuit_TN as bc

import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

#%%

arr_vonq=[]
arr_sep_mut=[]

interval =np.linspace(0.0,0.5,20) 
Sites=6
num_samples = 1
eps=0.1
gate="2haar"

start=time.time()
for i in tqdm(interval):
    print(i)
    numstep = 50

    
    circ = bc.circuit(Sites, numstep, meas_r=float(i), gate=gate, architecture="brick")
    circ.do_step(num=numstep, rec="von sep_mut")
    data_von = circ.rec_ent
    data_sep_mut = circ.rec_sep_mut
    
    for j in range(1,num_samples):
        # print("j: ",j)
        circ = bc.circuit(Sites, numstep, meas_r=float(i), gate=gate, architecture="brick")
        circ.do_step(num=numstep, rec="von sep_mut")
        data_von = np.average(np.array([data_von, circ.rec_ent]), axis=0,weights=[j,1] )
        data_sep_mut = np.average(np.array([data_sep_mut, circ.rec_sep_mut]),axis=0, weights=[j,1] )


    arr_vonq.append(data_von)
    arr_sep_mut.append(data_sep_mut)
end=time.time()
#%% Mut inf at ends
fig, ax = plt.subplots()
ax.plot(np.transpose(arr_sep_mut))

ax.legend(title='meas_r',labels=np.round(interval,3))
plt.title(f"Mut_info, Gate:{gate}, Sites:{Sites}, Samples:{num_samples}, Time={np.round(end-start,3)}")
ax.set_yscale('log')

plt.show()
#%% bipartite entropy

fig, ax = plt.subplots()
ax.plot(np.arange(0,np.shape(arr_vonq)[1]-1,2),np.transpose([[(arr_vonq[i][j]+arr_vonq[i][j+1])/2 for j in range(0,np.shape(arr_vonq)[1]-1,2)] for i in range(np.shape(arr_vonq)[0])]))

ax.legend(title='meas_r',labels=np.round(interval,3))
plt.title(f"Mut_info, Gate:{gate}, Sites:{Sites}, Samples:{num_samples}, Time={np.round(end-start,3)}")
# ax.set_yscale('log')

plt.show()

#%%
