# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 15:18:09 2023

@author: jogib
"""

#%%
arr_vonq=[]
arr_mutq=[]
arr_renq=[]
arr_negq=[]
interval =np.linspace(0.,1,5) 
Sites=6
num_samples = 100
eps=0.1
gate="2haar"

start=time.time()
for i in interval:
    print(i)
    numstep = 50

    
    circ = bc.circuit(Sites, numstep, init="up", meas_r=float(i), gate=gate, architecture="brick")
    circ.do_step(num=numstep, rec="von neg mut")
    data_von = circ.rec_ent
    data_neg = circ.rec_neg
    data_mut = circ.rec_mut_inf
    # data_ren= circ.rec_ren
    
    for j in range(1,num_samples):
        circ = bc.circuit(5, numstep, init="up", meas_r=float(i), gate=gate, architecture="brick")
        circ.do_step(num=numstep, rec="von neg mut")
        data_von = np.average(np.array([data_von, circ.rec_ent]), axis=0,weights=[j,1] )
        data_neg = np.average(np.array([data_neg, circ.rec_neg]), axis=0,weights=[j,1] )
        data_mut = np.average(np.array([data_mut, circ.rec_mut_inf]), axis=0,weights=[j,1] )

        # data_ren = np.average(np.array([data_ren, circ.rec_ren]), axis=0,weights=[j,1] )

    arr_vonq.append(data_von)
    arr_negq.append(data_neg)
    arr_mutq.append(data_mut)


    # arr_mutq.append(data_mut)
    # arr_renq.append(data_ren)
end=time.time()
#%%
fig, ax = plt.subplots(1,3,figsize=(9, 3),)
ax[0].plot(np.transpose(arr_vonq))

ax[0].legend(title='meas_r',labels=np.round(interval,3))
ax[0].set_title(f"Entropy")


ax[1].plot(np.transpose(arr_mutq))

# ax.legend(title='meas_r',labels=np.round(interval,3))
ax[1].set_title(f"MutInf")


ax[2].plot(np.transpose(arr_negq))

ax[2].set_title(f"LogNeg")

plt.show()