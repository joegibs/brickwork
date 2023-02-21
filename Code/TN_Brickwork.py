#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TN_Brickwork
---------------------------------------------------------------------
Simulating MIPT through tensor networks

    by Andrew Projansky - last modified February 16th
"""

import numpy as np
from numpy import linalg as LA
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from quimb import *

error_t = 10**(-8)
H = 1/np.sqrt(2) * np.array([[1,1],[1,-1]])
S = np.array([[1,0],[0,1j]])
T = np.array([[1,0],[0,np.exp(1j*np.pi/4)]])
X = np.array([[0,1],[1,0]])
Y = np.array([[0,-1j],[1j,0]])
Z = np.array([[1,0],[0,-1]])
CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])

class Circuit:
    """
    Class for circuits composed of matrix product operators acting on 
    matrix product states
    
    Parameters
    ----------
    N : int
        number of qubits
    max_chi : int
        max bond dimension in MPS
    init_state: string, array
        accepts string arguements 'zero','rand_loc', and 'random' for 
        generating initial states. Also accepts manually written array 
        as initial state
    c_center: int
        canonization center for MPS 
    meas_rate : float
        float [0,1] measurement rate for system
        
    Attributes
    ----------
    Psi: array
        empty array of tensors, to be filled after initializing circuit
    """
    
    def __init__(self, N=10, max_chi = None,
                init_state = 'zero', c_center=None):
        self.N = N
        if max_chi == None: self.max_chi = 2**(N//2)
        else: self.max_chi = max_chi
        self.init_state = init_state
        self.Psi = [0 for x in range(N)]
        if c_center==None: self.c_center = N-1
        else: self.c_center = c_center
        
    def init_Circuit(self):
        """
        Initializes circuit by defining Psi dependant on initial state
        input, then canonizes the state while normalizing all tensors

        """
        ### Initialize Psi
        if self.init_state == 'zero':
            for i in range(self.N):
                self.Psi[i] = np.array([[[1],[0]]])
        if self.init_state == 'rand_loc':
            for i in range(self.N):
                a = random.random()
                self.Psi[i] = np.array([[[a],[1-a]]])
        if self.init_state == 'random':
            self.Psi[0] = np.random.rand(1,2,min(self.max_chi, 2))
            for k in range(1,self.N):
                self.Psi[k] = np.random.rand(self.Psi[k-1].shape[1],2,
                          min(min(self.max_chi,self.Psi[k-1].shape[2]*2),
                              2**(self.N-k-1)))
        if type(self.init_state) == type(np.array([])):
            self.Psi = self.init_state
           
        ### Canonize Psi
        self.canonize_psi()
            
    def left_canonize(self, site):
        """
        Left canonoize sites by performing SVD on re-shaped sites

        Parameters
        ----------
        site : int
            site to be left normalized

        """
        d1 = self.Psi[site].shape[0]; d2 = self.Psi[site].shape[1]
        d3 = self.Psi[site].shape[2]; d1p = self.Psi[site+1].shape[0] 
        d2p = self.Psi[site+1].shape[1]; d3p = self.Psi[site+1].shape[2]; 
        psi_m = self.Psi[site].reshape(d1*d2, d3)
        u, d, vh = LA.svd(psi_m)
        d = self.trun_d(d)
        self.Psi[site] = u[:,np.arange(0,len(d),1)].reshape(d1,d2,len(d))
        #self.Psi[site] = u[:,np.arange(0,len(d),1)].reshape(d1,d2,d3)
        #psi_mp = np.diag(d) @ vh @ self.Psi[site+1].reshape(d1p, d2p*d3p) / LA.norm(d)
        psi_mp = np.diag(d) @ vh[np.arange(0,len(d),1)] @ self.Psi[site+1].reshape(d1p, d2p*d3p) / LA.norm(d)
        #self.Psi[site+1] = psi_mp.reshape(d1p, d2p, d3p)
        self.Psi[site+1] = psi_mp.reshape(len(d), d2p, d3p)
        
    def right_canonize(self, site):
        """
        Right canonoize sites by performing SVD on re-shaped sites

        Parameters
        ----------
        site : int
            site to be right normalized

        """
        d1 = self.Psi[site].shape[0]; d2 = self.Psi[site].shape[1]
        d3 = self.Psi[site].shape[2]; d1p = self.Psi[site-1].shape[0] 
        d2p = self.Psi[site-1].shape[1]; d3p = self.Psi[site-1].shape[2]; 
        psi_m = self.Psi[site].reshape(d1, d2* d3)
        u, d, vh = LA.svd(psi_m)
        d = self.trun_d(d)
        self.Psi[site] = vh[np.arange(0,len(d),1),:].reshape(d1,d2,d3)
        psi_mp = (self.Psi[site-1].reshape(d1p*d2p, d3p) @ u @ np.diag(d)) / LA.norm(d)
        self.Psi[site+1] = psi_mp.reshape(d1p, d2p, d3p)
        
    def contract_to_dense(self):
        """
        Contracts MPS into dense tensor of rank n, each dimension 2

        Returns
        -------
        contracted : array
            dense contracted state-tensor from MPS
        """
        
        t1 = np.array([1])
        contracted = np.tensordot(t1, self.Psi[0], axes=((0),(0)))
        for i in np.arange(1,self.N, 1):
            contracted = np.tensordot(contracted, self.Psi[i], axes=((i), (0)))
        contracted=np.tensordot(contracted, t1, axes=((self.N), (0)))
        return contracted
    
    def sqgate(self, site, gate):
        """
        applies single qubit gate by contracting with physical index, 

        Parameters
        ----------
        site : TYPE
            DESCRIPTION.
        gate : TYPE
            DESCRIPTION.


        """
        self.Psi[site] = np.tensordot(self.Psi[site], gate, axes=((1),(0)))
        self.Psi[site] = np.swapaxes(self.Psi[site],1,2)
        
    def twoqgate(self, control, gate):
        """
        applies two qubit gate by re-shaping MPS on control and target into 
        one site, contracting with 2 qubit gate on dim 4 physical index
        before re-shaping back into two nodes using the SVD

        Parameters
        ----------
        control : TYPE
            DESCRIPTION.
        gate : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        left_one = self.Psi[control].shape[0]
        right_two = self.Psi[control+1].shape[2]
        two_site = np.tensordot(self.Psi[control], self.Psi[control+1], axes=((2),(0)))
        two_site = np.reshape(two_site, (two_site.shape[0], 4, two_site.shape[3]))
        two_site = np.tensordot(two_site, gate, axes=((1),(0)))
        two_site = np.swapaxes(two_site, 1, 2)
        two_site = np.reshape(two_site, (two_site.shape[0]*2, 2*two_site.shape[2]))
        u, d, vh = LA.svd(two_site)
        d = self.trun_d(d)
        u = u[:,np.arange(0,len(d),1)]
        self.Psi[control] = np.reshape(u, (left_one, 2, u.shape[1]))
        rmat = np.diag(d) @ vh[np.arange(0,len(d),1),:]
        self.Psi[control+1] = np.reshape(rmat, (len(d), 2, right_two) )
        
    def canonize_psi(self):
        """
        canonizes psi by SVD on each site respective to center of canonization

        """
        for i in range(self.c_center):
            self.left_canonize(i)     
        for i in range(self.N-1, self.c_center, -1):
            self.right_canonize(i)
        ### Normalize center tensor
        self.Psi[self.c_center] = self.Psi[self.c_center]/LA.norm(self.Psi[self.c_center])
        
    def trun_d(self, d):
        """
        Truncates singular values based on error threshold 
        ... need to implement max_chi truncation as well

        Parameters
        ----------
        d : array
            array of non truncated singular values

        Returns
        -------
        d : array
            Dtruncated vector of singular values

        """
        
        for i in range(len(d)-1,-1,-1):
            if d[i] > error_t:
                d = d[:i+1]
                break
        return d
    
    def meas_qbit(self, q):
        """
        at site, projects onto one qubit or another based on probability
        of measuring that outcome (relative to other)

        Parameters
        ----------
        q : int
            site to be measured

        """
        
        mv = np.zeros(2)
        sv = np.zeros(2, dtype='complex')
        for m in range(2):
            fl = self.Psi[q][:,m,:].flatten()
            sv[m] = sum(fl); fl = np.real(fl*np.conj(fl))
            mv[m] = sum(fl)
        mv = mv / sum(mv)
        if np.random.rand() < mv[1]:
            (self.Psi[q][:,1,:]).fill(0.)
            self.Psi[q] = self.Psi[q]/sv[0] 
        else:
            (self.Psi[q][:,0,:]).fill(0.)
            self.Psi[q] = self.Psi[q]/sv[1] 
            
def get_layers(j, N):
    """
    Logic for geenrating list of sites for single and two qubit 
    gates to be applied on

    Parameters
    ----------
    j : int
        even or odd layer of circuit 
    N : int
        number of sites

    Returns
    -------
    dqg : list
        list of contorl qubits two qubit gates are going to be applied on
    sqg : list
        list of qubits single qubit gates are going to be applied on

    """
    dqg = []; sqg = []
    if j % 2 == 0:
        for i in range(N//2):
            dqg.append(2*i)
        if N % 2 != 0:
            sqg.append(N-1)
    if j % 2 == 1:
        sqg.append(0)
        for i in range((N-1)//2):
            dqg.append(2*i + 1)
        if N % 2 == 0:
            sqg.append(N-1)
    return dqg, sqg
                 
        
def exp(N, m_rate, layers, init_state='zero'):
    """
    Initializes and runs hybrid circuit of determined size, with 
    a measurement rate and set amount of layers

    Parameters
    ----------
    N : int
        number of sites
    m_rate : float
        measurement rate in [0,1]
    layers : int
        number of brickwork layers

    Returns
    -------
    circ : Circiut
        returns circ object

    """
    
    circ = Circuit(N, init_state)
    circ.init_Circuit()
    ev_layer_d, ev_layer_s = get_layers(0, N)
    odd_layer_d, odd_layer_s = get_layers(1, N)
    bd_l = np.arange(0, layers, 1)
    for j in range(layers):
        if j % 2 == 0:
            sl, dl = ev_layer_s, ev_layer_d
        else:
            sl, dl = odd_layer_s, odd_layer_d
        for q in sl:
            circ.sqgate(q, rand_uni(2))
        for q in dl:
            circ.twoqgate(q, rand_uni(4))
        if m_rate > 0:
            for q in range(N):
                if np.random.rand() < m_rate:
                    circ.meas_qbit(q)
        circ.canonize_psi()
        m = 0
        for t in circ.Psi:
            if t.shape[2] > m:
                m = t.shape[2]
        bd_l[j] = m
        
                    
    return circ, bd_l
#%%
#Testing time
layers=6
bdl = np.arange(0, layers, 1)
for mr in np.array([0, 0.05, 0.1, 0.17, 0.22, 0.3]):
    for j in range(1000):
        circ, bd_l = exp(10,mr,layers); #bd_l = np.log(bd_l)
        bdl = bdl + bd_l
    bdl = bdl / 100
    plt.plot(np.arange(0,layers,1), np.log(bdl))
#plt.show()
#%%
#Testing of Markov Matrix
p = np.kron([0.4,0.6],[0.3,0.7]); p = np.kron(p, [0.1,0.9])
