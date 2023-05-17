# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 11:51:11 2023

@author: jogib
"""
import numpy as np
import matplotlib.pyplot as plt
from quimb import *

#%%
def markov(eps):
    # need to check this currently a left matrix....
    M = np.random.randint(0,high=9000,size=(4, 4))
    for i in range(15):
        M = M / np.sum(M, axis=0, keepdims=True)
        M = M / np.sum(M, axis=1, keepdims=True)
    if np.isnan(np.min(M)):
        M=markov(eps)
    return M

def markov_alt(eps):
    # need to check this currently a left matrix....
    M=[]
    for i in range(4):
        arr=[]
        tot = 1
        for j in range(3):
            samp = np.random.uniform(0,tot)
            tot -=samp
            arr.append(samp)
        arr.append(tot)
        M.append(arr)
    return M

def markov(eps):
    # need to check this currently a left matrix....
    M = np.random.poisson(size=(4,4))
    for i in range(15):
        M = M / np.sum(M, axis=0, keepdims=True)
        M = M / np.sum(M, axis=1, keepdims=True)
    if np.isnan(np.min(M)):
        M=markov(eps)
    return M
def markove(eps):
    # need to check this currently a left matrix....
    M = np.array(
        [
            [1-eps, 0, 0, eps],
            [0, 1-eps, eps, 0],
            [0, eps, 1-eps, 0],
            [eps, 0, 0, 1-eps],
        ])
    return M
def markov_mix(eps):
    # need to check this currently a left matrix....
    M = np.array(
        [
            [1/4, 1/4, 1/4, 1/4],
            [1/4, 1/4, 1/4, 1/4],
            [1/4, 1/4, 1/4, 1/4],
            [1/4, 1/4, 1/4, 1/4],
        ])
    return M
def rand_rxx():
    arr = 2 * np.pi * np.random.rand(1)
    return RXX(arr[0])
def RXX(theta):
    return np.array(
        [
            [np.cos(theta/2), 0, 0, -1j*np.sin(theta/2)],
            [0, np.cos(theta/2), -1j*np.sin(theta/2), 0],
            [0, -1j*np.sin(theta/2), np.cos(theta/2), 0],
            [-1j*np.sin(theta/2), 0, 0, np.cos(theta/2)],
        ]
    )
def rand_match():
    arr = 2 * np.pi * np.random.rand(2)
    return match(arr[0]/2, arr[1]/2)
def match(theta, phi):
    return np.array(
        [
            [np.cos(theta), 0, 0, -np.sin(theta)],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [np.sin(theta), 0, 0, np.cos(theta)],
        ]
    )
def IX_q(eps,theta):
    M = np.exp(1j*theta)*(np.sqrt((1-eps))*np.identity(4) + 1j*np.sqrt(eps)*kron(pauli("X"),pauli("X")))
    return M

def IX_class(eps):
    M = (1-eps)*np.identity(4) + eps*kron(pauli("X"),pauli("X"))
    return M
def rand_IX_q():
    return IX_q(np.random.rand(),2*np.pi*np.random.rand())
def rand_IX_c():
    return IX_class(np.random.rand())

def oz():
    return np.array([[0,1,0,0,0],[0.25,0,0.75,0,0],[0,0.5,0,0.5,0],[0,0,0.75,0,0.25],[0,0,0,1,0]])
#%%
thetas = np.array([])
eigens = np.array([])
for i in range(10000):
    mat = markov_alt(0.1)
    d, u = np.linalg.eig(mat)
    sort_d = np.sort(d)
    theta_d = np.arctan2(sort_d.real,sort_d.imag)
    # thetas =np.concatenate((thetas,theta_d))
    eigens = np.concatenate((eigens,sort_d))
# plt.plot(thetas)
plt.hist(eigens,bins=50,density=True)
# plt.xlim(0,1.2)
# plt.hist(eigens,density=True)
