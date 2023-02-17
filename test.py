
import brickwork.circuit_TN as bc

import numpy as np
import matplotlib.pyplot as plt
import time

#%%

arr_vonq=[]

interval =[0]#np.linspace(0.,0.3,4) 
Sites=5
num_samples = 100
eps=0.1
gate="2haar"

start=time.time()
for i in interval:
    print(i)
    numstep = 50

    
    circ = bc.circuit(Sites, numstep, meas_r=float(i), gate=gate, architecture="brick")
    circ.do_step(num=numstep, rec="von")
    data_von = circ.rec_ent
    
    for j in range(1,num_samples):
        circ = bc.circuit(5, numstep, meas_r=float(i), gate=gate, architecture="brick")
        circ.do_step(num=numstep, rec="von")
        data_von = np.average(np.array([data_von, circ.rec_ent]), axis=0,weights=[j,1] )

    arr_vonq.append(data_von)
end=time.time()
#%%

fig, ax = plt.subplots()
ax.plot(np.transpose([[arr_vonq[i][j] for j in range(0,np.shape(arr_vonq)[1],2)] for i in range(np.shape(arr_vonq)[0])]))

ax.legend(title='meas_r',labels=np.round(interval,3))
plt.title(f"Mut_info, Gate:{gate}, Sites:{Sites}, Samples:{num_samples}, Time={np.round(end-start,3)}")
# ax.set_yscale('log')

plt.show()