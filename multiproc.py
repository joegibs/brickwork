from multiprocessing.dummy import Process

import brickwork.circuit_TN as bc

import numpy as np
import matplotlib.pyplot as plt
import time

def Calculation(step,inter):
    Sites=6
    num_samples = 10
    eps=0.1
    gate="2haar"

    print(inter)
    numstep = 50

    
    circ = bc.circuit(Sites, numstep, meas_r=float(inter), gate=gate, architecture="brick")
    circ.do_step(num=numstep, rec="von sep_mut")
    data_von = circ.rec_ent
    data_sep_mut = circ.rec_sep_mut
    
    for j in range(1,num_samples):
        # print("j: ",j)
        circ = bc.circuit(Sites, numstep, meas_r=float(inter), gate=gate, architecture="brick")
        circ.do_step(num=numstep, rec="von sep_mut")
        data_von = np.average(np.array([data_von, circ.rec_ent]), axis=0,weights=[j,1] )
        data_sep_mut = np.average(np.array([data_sep_mut, circ.rec_sep_mut]),axis=0, weights=[j,1] )
    
    arr_vonq[step]=data_von
    arr_sep_mut[step]=data_sep_mut
    
        

if __name__ == "__main__":

    Ans = np.zeros((2,2))
    steps=20
    arr_vonq=[[]]*steps
    arr_sep_mut=[[]]*steps
    interval =np.linspace(0.0,0.5,steps) 
    start=time.time()

    Para_1 = [[5,4],[1,2]]
    Para_2 = [[1,2],[3,4]]
    Para_3 = [[4,4],[2,5]]

    processes = [Process(target=Calculation, args=(step,value)) for step,value in enumerate(interval)]

    for p in processes:
        p.start()

    for p in processes:
        p.join()
    
    end=time.time()

    Sites=6
    num_samples = 1
    eps=0.1
    gate="2haar"
    fig, ax = plt.subplots()
    ax.plot(np.arange(0,np.shape(arr_vonq)[1]-1,2),np.transpose([[(arr_vonq[i][j]+arr_vonq[i][j+1])/2 for j in range(0,np.shape(arr_vonq)[1]-1,2)] for i in range(np.shape(arr_vonq)[0])]))

    ax.legend(title='meas_r',labels=np.round(interval,3))
    plt.title(f"Ent, Gate:{gate}, Sites:{Sites}, Samples:{num_samples}, Time={np.round(end-start,3)}")
    # ax.set_yscale('log')

    plt.show()

#%% bipartite entropy

    
    
#make array approprite size at first dont append
#calculation will be the thing i need 
#loop over junk
