
#%%
#set current working directory
import brickwork.circuit_TN as bc
# import brickwork.circuit_class as bc
import time
import numpy as np
import matplotlib.pyplot as plt


#%%
data_mark = []
start=time.time()
for ii in np.linspace(0,0.5,1):
    arr_von=[]
    arr_mut=[]
    arr_ren=[]
    interval =np.linspace(0.,1,5) 
    Sites=5
    num_samples = 50
    eps=ii
    gate="2haar"
    

    for i in interval:
        print(i)
        numstep = 30
    
        
        circ = bc.circuit(Sites, numstep, init="up", meas_r=float(i), gate=gate, architecture="brick")
        circ.do_step(num=numstep, rec="von")
        data_von = circ.rec_ent
        
        for j in range(1,num_samples):
            circ = bc.circuit(5, numstep, init="up", meas_r=float(i), gate=gate, architecture="brick")
            circ.do_step(num=numstep, rec="von")
            data_von = np.average(np.array([data_von, circ.rec_ent]), axis=0,weights=[j,1] )
            
        arr_von.append(data_von)
    data_mark.append([arr_von,arr_mut,arr_ren])
end=time.time()
#%%
fig, ax = plt.subplots()
test_arr = [i[0] for i in data_mark]
for arr in test_arr:
    ax.plot(np.transpose([[arr[i][j] for j in range(0,np.shape(arr)[1],2)] for i in range(np.shape(arr)[0])]))
# ax.plot(np.transpose([[arr_mutq[i][j] for j in range(0,np.shape(arr_mutq)[1],2)] for i in range(np.shape(arr_mutq)[0])]))

    ax.legend(title='meas_r',labels=np.round(interval,3))
plt.title(f"Mut_info, Gate:{gate}, Sites:{Sites}, Samples:{num_samples}, Time={np.round(end-start,3)}")

plt.show()
#%%
test_arr=test_arr = [i[0] for i in data_mark]
fig, ax = plt.subplots()

for arr in test_arr:

    ax.plot(interval,[i[-1] for i in [[arr[i][j] for j in range(0,np.shape(arr)[1],2)] for i in range(np.shape(arr)[0])]])
    # ax.plot(interval,[i[-1] for i in [[arr_mutq[i][j] for j in range(0,np.shape(arr_mutq)[1],2)] for i in range(np.shape(arr_mutq)[0])]])
    
    # ax.set_yscale('log')
# ax.legend(title='meas_p',labels=np.round(interval,3))
plt.title(f"Mut_info, Sites:{Sites}, Samples:{num_samples}, Time={np.round(end-start,3)}")

plt.show()
#%%
fig, ax = plt.subplots()
ax.plot(np.transpose([[arr_mut[i][j] for j in range(0,np.shape(arr_mut)[1],2)] for i in range(np.shape(arr_mut)[0])]))
# ax.plot(np.transpose([[arr_mutq[i][j] for j in range(0,np.shape(arr_mutq)[1],2)] for i in range(np.shape(arr_mutq)[0])]))

ax.legend(title='meas_r',labels=np.round(interval,3))
plt.title(f"Mut_info, Gate:{gate}, Sites:{Sites}, Samples:{num_samples}, Time={np.round(end-start,3)}")

plt.show()

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
gate_holes=0.2

start=time.time()
for i in interval:
    print(i)
    numstep = 50

    
    circ = bc.circuit(Sites, numstep, init="up", meas_r=float(i), gate=gate, architecture="brick", gate_holes=gate_holes)
    circ.do_step(num=numstep, rec="von neg mut")
    data_von = circ.rec_ent
    data_neg = circ.rec_neg
    data_mut = circ.rec_mut_inf
    # data_ren= circ.rec_ren
    
    for j in range(1,num_samples):
        circ = bc.circuit(5, numstep, init="up", meas_r=float(i), gate=gate, architecture="brick", gate_holes=gate_holes)
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
fig, ax = plt.subplots()
ax.plot(np.transpose(arr_vonq))

ax.legend(title='meas_r',labels=np.round(interval,3))
plt.title(f"Entropy, Gate:{gate}, Sites:{Sites}, Samples:{num_samples}, Time={np.round(end-start,3)}")

plt.show()
#%%
fig, ax = plt.subplots()
ax.plot(np.transpose(arr_mutq))

ax.legend(title='meas_r',labels=np.round(interval,3))
plt.title(f"Mut, Gate:{gate}, Sites:{Sites}, Samples:{num_samples}, Time={np.round(end-start,3)}")

plt.show()
#%%
fig, ax = plt.subplots()
ax.plot(np.transpose(arr_negq))

ax.legend(title='meas_r',labels=np.round(interval,3))
plt.title(f"Negativity, Gate:{gate}, Sites:{Sites}, Samples:{num_samples}, Time={np.round(end-start,3)}")

plt.show()
#%%
fig, ax = plt.subplots()
# ax.plot(np.transpose([[arr_mut[i][j] for j in range(0,np.shape(arr_mut)[1],2)] for i in range(np.shape(arr_mut)[0])]))
ax.plot(np.transpose([[arr_mutq[i][j] for j in range(0,np.shape(arr_mutq)[1],2)] for i in range(np.shape(arr_mutq)[0])]))

ax.legend(title='meas_r',labels=np.round(interval,3))
plt.title(f"Mut_info, Gate:{gate}, Sites:{Sites}, Samples:{num_samples}, Time={np.round(end-start,3)}")

plt.show()
#%%fig, ax = plt.subplots()
fig, ax = plt.subplots()
ax.plot(interval,[i[-1] for i in [[arr_mut[i][j] for j in range(0,np.shape(arr_mut)[1],2)] for i in range(np.shape(arr_mut)[0])]])
ax.plot(interval,[i[-1] for i in [[arr_mutq[i][j] for j in range(0,np.shape(arr_mutq)[1],2)] for i in range(np.shape(arr_mutq)[0])]])

ax.set_yscale('log')
# ax.legend(title='meas_p',labels=np.round(interval,3))
plt.title(f"Mut_info, Sites:{Sites}, Samples:{num_samples}, Time={np.round(end-start,3)}")

plt.show()
#%%
fig, ax = plt.subplots()
ax.plot(np.transpose(arr_negq))

ax.legend(title='meas_r',labels=np.round(interval,3))
plt.title(f"Negativity, Gate:{gate}, Sites:{Sites}, Samples:{num_samples}, Time={np.round(end-start,3)}")

plt.show()
#%%fig, ax = plt.subplots()
fig, ax = plt.subplots()
ax.plot(interval,[i[-1] for i in [[arr_von[i][j] for j in range(0,np.shape(arr_mut)[1],2)] for i in range(np.shape(arr_mut)[0])]])
ax.plot(interval,[i[-1] for i in [[arr_vonq[i][j] for j in range(0,np.shape(arr_mutq)[1],2)] for i in range(np.shape(arr_mutq)[0])]])

ax.set_yscale('log')
ax.legend(title='meas_p',labels=np.round(interval,3))
plt.title(f"Entropy, Gate:{gate}, Sites:{Sites}, Samples:{num_samples}, Time={np.round(end-start,3)}")

plt.show()
#%%
fig, ax = plt.subplots()
ax.plot(np.transpose(arr_ren))
ax.legend(title='meas_r',labels=np.round(interval,3))
plt.title(f"Ren_Entropy, Gate:{gate}, Sites:{Sites}, Samples:{num_samples}, Time={np.round(end-start,3)}")

plt.show()
#%%fig, ax = plt.subplots()
fig, ax = plt.subplots()
ax.plot(interval,[i[-1] for i in [[arr_ren[i][j] for j in range(0,np.shape(arr_mut)[1],2)] for i in range(np.shape(arr_mut)[0])]])
ax.plot(interval,[i[-1] for i in [[arr_renq[i][j] for j in range(0,np.shape(arr_mut)[1],2)] for i in range(np.shape(arr_mut)[0])]])

ax.set_yscale('log')
ax.legend(title='meas_p',labels=np.round(interval,3))
plt.title(f"Ren_entropy, Gate:{gate}, Sites:{Sites}, Samples:{num_samples}, Time={np.round(end-start,3)}")

plt.show()
#%%
plt.plot(np.nansum(np.log(np.array(circ.rec_mut_inf)), 1))
