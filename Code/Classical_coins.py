# -*- coding: utf-8 -*-
"""
BAD BAD NOT GOOD
[[1,0,0,0],
 [0,1,0,0],
 [0,0,0,1],
 [0,0,1,0]]
breaks things
@author: jogib
"""

import numpy as np
import matplotlib.pyplot as plt
import time

#%%

markov = lambda e: np.array(
    [
        [1 - e, e / 3, e / 3, e / 3],
        [e / 3, 1 - e, e / 3, e / 3],
        [e / 3, e / 3, 1 - e, e / 3],
        [e / 3, e / 3, e / 3, 1 - e],
    ]
)

#%%


class coin:
    """
    contains all the functions for a singule coin
    """
    def __init__(self, h: float = None):
        if h is None:
            h = np.random.rand()
        self.prob = np.array([h, 1 - h])

    def measure_coin(self):
        temp = np.array([0, 0])
        temp[np.random.choice([0, 1], p=self.prob)] = 1
        self.prob = temp

    def vne(self):
        """
        vpn nueman entropy of one coin
        """
        if self.prob[0] * self.prob[1] == 0:
            return 0
        else:
            return sum([-i * np.log2(i) for i in self.prob])


#%%
# single coin to 2 coin state
# bi coins to coins
# sum entropy for all coins


class coin_list:
    """
    cointains a list of coin objects and functions to brickwork
    evolve the list and take measurements
    """
    def __init__(
        self, N: int, meas_prob: float, bc: str = "periodic", eps: float = 0.1
    ):
        self.N = N
        self.bc = bc
        self.meas_prob = [1 - meas_prob, meas_prob]
        self.coins = np.array([coin(h=0) for i in range(N)])
        self.step_num = 0
        self.pairs, self.offset_pairs = self.gen_pairs()
        self.all_meas = []
    def reset_coins(self):
        self.coins = np.array([coin() for i in range(self.N)])
        
    def CoinsToBicoin(self, c1, c2):
        """
        Combine two prob vectors from coins

        Parameters
        ----------
        c1 : 
            Index for a coin.
        c2 : TYPE
            Index for a coin.

        Returns
        -------
        list
            Combined Prob vector.

        """
        vec = np.kron(self.coins[c1].prob, self.coins[c2].prob)
        return vec / sum(vec)  # chop off umerical errors

    def BicoinToCoins(self, b):
        """
        Super broke
        bad
        no good
        """
        return coin(np.round(b[0] + b[1], 10)), coin(np.round(b[0] + b[2], 10))

    def gen_pairs(self):
        pairs = [[i, i + 1] for i in range(0, self.N - 1, 2)]
        offset_pairs = [[i, i + 1] for i in range(1, self.N - 1, 2)]
        if self.bc == "periodic":
            if self.N % 2 == 0:
                offset_pairs.append([self.N - 1, 0])
            else:
                pairs.append([self.N - 1, 0])
        return pairs, offset_pairs

    def single_evo(self, a, b, matrix):
        bicoin = self.CoinsToBicoin(a, b)
        post_markov = np.dot(matrix, bicoin)
        self.coins[a], self.coins[b] = self.BicoinToCoins(post_markov)

    def step(self, matrix):

        # check step num to get pairs
        if self.step_num % 2 == 0:
            pairs = self.pairs
        else:
            pairs = self.offset_pairs
        # print(pairs)
        # single evolve all the pairs
        for i in pairs:
            self.single_evo(i[0], i[1], matrix)
        # random meas
        for i in self.coins:
            if np.random.choice([False, True], p=self.meas_prob):
                i.measure_coin()
        # get meas
        meas_vne = [i.vne() for i in self.coins]

        # +1 step_num
        self.step_num += 1

        return meas_vne

    def steps(self, N, matrix):
        for i in range(N):
            self.all_meas.append(sum(self.step(matrix)))
    def get_avg(self,num,curr,new):
        return np.average(np.array([curr, new]), axis=0,weights=[1,1] )
    
    def repeat_steps(self, N, matrix,num_repeats):
        #will need to refactor probabily
        self.reset_coins()
        old_data=[]
        
        for i in range(N):
            old_data.append(sum(self.step(matrix)))
           
        counter=1
        for trial in range(num_repeats-1):
            self.reset_coins()
            data = []
            for i in range(N):
                data.append(sum(self.step(matrix)))
            # print(data)
            # print(old_data)
            old_data=self.get_avg(counter,old_data,data)
            counter+=1
        self.all_meas = old_data
            
            


#bootstrap i need to
#1. create the coin list
#2. run the exp multiple times
#3. avg results
#4.return same data
#%%
test = []
num_steps=1
num_coins=1000
num_samples=1
eps = 0.1
start=time.time()
for i in np.linspace(0,0.3,8):
    test.append(coin_list(num_coins,i))

for i in test:
    i.repeat_steps(num_steps,markov(eps),num_samples)
    plt.plot(np.linspace(1, num_steps, num_steps), i.all_meas)
end=time.time()

plt.title(f"Entropy, Coins:{num_coins}, Samples:{num_samples},Eps={eps}, Time={np.round(end-start,3)}")
#%%
test = []
for i in np.linspace(0,0.3,5):
    test.append(coin_list(1000,i))

for i in test:
    i.steps(30,markov(0.1))
    plt.plot(np.linspace(1, 30, 30), i.all_meas)

#%% always zero? i mean it makes sense
def Sentropy(vec):
    return sum([0 if i == 0 else -i * np.log2(i) for i in vec])
def mut_info(vec1,vec2):
    return Sentropy(vec1)+Sentropy(vec2)-Sentropy(np.kron(vec1,vec2))
#mut info
vec=np.kron(c.coins[0].prob,c.coins[1].prob)
vec1= np.kron(c.coins[0].prob,c.coins[1].prob)
vec2= np.kron(c.coins[2].prob,c.coins[3].prob)
#
#for i in c.coins[2:5]:
#    vec=np.kron(vec,i.prob)
