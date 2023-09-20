import brickwork.circuit_TN as bc
# import brickwork.circuit_class as bc

import brickwork.circuit_lindblad as bc


import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

#%%
tot_tri = []
tot_vonq = []
#%%
gate_holes = 0.
sits =[6]
for Sites in sits:
    arr_vonq=[]
    arr_sep_mut=[]
    arr_tri_mut = []

    
    interval =[0]#np.linspace(0.,1,2) 
    num_samples = 50
    eps=0.1
    gate="2haar#"
    rec='sharp'
    
    start=time.time()
    for i in tqdm(interval):
        # print(i)
        numstep = 8*Sites
    
        
        circ = bc.circuit(Sites, numstep, meas_r=float(i), gate=gate, architecture="brick",gate_holes=gate_holes)
        circ.do_step(num=numstep, rec=rec)
        vonq_avg = [(circ.rec_ent[i]+circ.rec_ent[i+1])/2 for i in np.arange(0,len(circ.rec_ent)-1,2)]
        data_von = vonq_avg
        # data_sep_mut = circ.rec_sep_mut
        # data_tri_mut = circ.rec_tri_mut
        
        for j in range(1,num_samples):
            # print("j: ",j)
            circ = bc.circuit(Sites, numstep, meas_r=float(i), gate=gate, architecture="brick",gate_holes=gate_holes)
            circ.do_step(num=numstep, rec=rec)
            vonq_avg = [(circ.rec_ent[i]+circ.rec_ent[i+1])/2 for i in np.arange(0,len(circ.rec_ent)-1,2)]
            data_von = np.average(np.array([data_von, vonq_avg]), axis=0,weights=[j,1] )
            # data_sep_mut = np.average(np.array([data_sep_mut, circ.rec_sep_mut]),axis=0, weights=[j,1] )
            # data_tri_mut = np.average(np.array([data_tri_mut, circ.rec_tri_mut]),axis=0, weights=[j,1] )
    
        
        arr_vonq.append(data_von)
        # arr_sep_mut.append(data_sep_mut)
        # arr_tri_mut.append(data_tri_mut)
    end=time.time()
    tot_vonq.append([x[-1] for x in arr_vonq])
    # tot_tri.append(arr_tri_mut)

#%%
sits=[4,6]
# sits=[0.1,0.5,0.01,0.05,0.025,0.015,0.3,0.9,0.7]
fig, ax = plt.subplots()
for tri in tot_vonq:
    ax.plot(interval,tri)


ax.legend(title='length',labels=sits)
plt.title(f"Bip_ent, Gate:{gate}, Sites:{Sites}, Samples:{num_samples}")
# ax.set_yscale('log')

plt.show()
#%%
from scipy.optimize import least_squares
#define objective function that should be minimized

#x = (p - pc)*L^(1/v)
L=[4,6]
# interval =np.linspace(0.,0.8,20)


#guess pc and say v =1 plot
#(p-pc)L
def xfunc(p,l,pc,v):
    return (p-pc)*l**(1/v)

def Spc(pc,l):
    return np.interp(pc,interval,tot_vonq[L.index(l)])

def yfunc(p,l,pc):
    return np.interp(p,interval,tot_vonq[L.index(l)]) - Spc(pc,l)

def mean_yfunc(p,pc):
    return np.mean([yfunc(p,l,pc) for l in L])





#%%
from scipy.optimize import minimize

def R(params):
    pc,v = params
    #sum over all the square differences
    x_vals = [[xfunc(p,l,pc,v) for p in interval] for l in L]
    y_vals = [[yfunc(p,l,pc) for p in interval] for l in L]
    min_x = np.max([x[0] for x in x_vals]) #max for smallest value st all overlap
    max_x = np.min([x[-1] for x in x_vals]) # min again to take overlap
    xi = np.linspace(min_x,max_x)
    mean_x_vals = np.mean(x_vals,axis=0)
    mean_y_vals = [mean_yfunc(p,pc) for p in interval]
    
    def mean_y(x):
        return np.interp(x,mean_x_vals,mean_y_vals)
    
    return np.sum([[(np.interp(x,x_vals[i],y_vals[i]) - mean_y(x))**2 for x in xi] for i in range(len(L))]) 
    
initial_guess = [0.0,0.1]
res = minimize(R, initial_guess)
#%%

ppc,vv=res.x
x_vals = [[xfunc(p,l,ppc,vv) for p in interval] for l in L]
y_vals = [[yfunc(p,l,ppc) for p in interval] for l in L]
# mean_y_vals = [mean_yfunc(p,0.26) for p in interval]

fig, ax = plt.subplots()
for i in range(len(L)):
    ax.plot(x_vals[i],y_vals[i])
# ax.plot(np.mean(x_vals,axis=0),mean_y_vals)

sits=[4,6,8]
ax.legend(title='length',labels=sits)
plt.title(f"Critical Behavior, pc:{np.round(ppc,3)}, V:{np.round(vv,3)}")
# ax.set_yscale('log')
#%%%
fig, ax = plt.subplots()
ax.plot(np.transpose(arr_vonq))
ax.legend(title='meas_r',labels=np.round(interval,3))
plt.title(f"Entropy, Gate:{gate}, Sites:{Sites}, Samples:{num_samples}, Time={np.round(end-start,3)}")

plt.show()