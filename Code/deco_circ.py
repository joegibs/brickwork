#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Decoherence
---------------------------------------------------------------------
Simulating MIPT with general quantum channels

TODO
-----
    would be nice to do this with MPOs... good way to shortcut the 
    decoherence with this structure?

    by Andrew Projansky - last modified February 21st
"""

import numpy as np
from numpy import linalg as LA
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import qiskit.quantum_info as qis
#%%
error_t = 10**(-8)
H = 1/np.sqrt(2) * np.array([[1,1],[1,-1]])
S = np.array([[1,0],[0,1j]])
T = np.array([[1,0],[0,np.exp(1j*np.pi/4)]])
X = np.array([[0,1],[1,0]])
Y = np.array([[0,-1j],[1j,0]])
Z = np.array([[1,0],[0,-1]])
CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])

def cc(mat):
    
    return np.conj(mat).T

def make_sim(dim, simmean, simwidth):
    
    RR = np.random.normal(simmean, simwidth, (dim,dim))
    SM = RR + 1j * np.random.normal(simmean, simwidth, (dim,dim))
    return SM

def make_unitary(dim, simmean, simwidth):
    
    sim = make_sim(dim, simmean, simwidth)
    Q, R = np.linalg.qr(sim)
    Etta = np.zeros((dim,dim))
    for j in range(dim):
        Etta[j,j] = R[j,j]/LA.norm(R[j,j])
    U = Q @ Etta
    return U

class Density_Circ:
    
    def __init__(self, N, layers, m_rate, init_state = 'zero'):
        
        self.N = N
        self.layers = layers
        self.m_rate = m_rate
        self.rho = self.init_density(init_state)
        self.gates = self.make_gates()
        self.ms = self.make_measures()
        
    def init_density(self, init_state):
        
        if init_state == 'zero':
            rho = np.zero((2**self.N,2**self.N), dtype='complex')
            rho[0,0] = 1
        if init_state == 'rand product':
            for j in range(self.N):
                rand_psi = np.random.rand(2) + 1j*np.random.rand(2)
                rand_psi = rand_psi / LA.norm(rand_psi)
                rand_rho = np.outer(cc(rand_psi), rand_psi)
                if j == 0:
                    rho = rand_rho
                else:
                    rho = np.kron(rho, rand_rho)

        return rho
    
    def make_gates(self):
        
        gates = [0 for j in range(self.layers)]
        for k in range(self.layers):
            g_l = []
            if k % 2 == 0:
                for l in range(self.N//2):
                    g_l.append(make_unitary(4, 0, 1))
                if self.N % 2 != 0:
                    g_l.append(make_unitary(2,0,1))
            else:
                g_l.append(make_unitary(2,0,1))
                for l in range((self.N-1)//2):
                    g_l.append(make_unitary(4, 0, 1))
                if self.N % 2 == 0:
                    g_l.append(make_unitary(2,0,1))
            gates[k] = g_l
        return gates
                    
    def make_measures(self):
        
        ms = [0 for j in range(self.layers)]
        for k in range(self.layers):
            m_l = []
            for s in range(self.N):
                if self.m_rate > np.random.rand():
                    m_l.append(s)
            ms[k] = m_l
        return ms