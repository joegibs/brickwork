# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 15:07:50 2023

@author: jogib
"""

import numpy as np
import matplotlib.pyplot as plt

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
            return sum([i * np.log(i) for i in self.prob])


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
        self.coins = np.array([coin() for i in range(N)])
        self.step_num = 0
        self.pairs, self.offset_pairs = self.gen_pairs()
        self.all_meas = []

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


#%%
c = coin_list(1000, 0.0)
c1 = coin_list(1000, 0.1)
c2 = coin_list(1000, 0.5)

c.steps(30, markov(0.1))
c1.steps(30, markov(0.1))
c2.steps(30, markov(0.1))

plt.plot(np.linspace(1, 30, 30), c.all_meas)
plt.plot(np.linspace(1, 30, 30), c1.all_meas)
plt.plot(np.linspace(1, 30, 30), c2.all_meas)
