# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
from copy import copy

def dvdt(q,p):
    gauss_rand = np.random.normal(0.0,1.0)
    return q #+ gamma*p -(2*beta*gamma)**0.5*gauss_rand

def potential(q):
    return 0.5*q**2




deltat = np.pi/1000
N = 1
dimension = 1
gamma = 1.0
beta = 1.0



def velocity_verlet_step(pspace,dvdt,deltat,etherm):
    q = pspace[0]
    p = pspace[1]
    
    c1 = np.exp(-(deltat/2.0)*gamma)
    c2 = (1-c1**2)**0.5
    
    
    etherm+=p**2/2.0
    gauss_rand = np.random.normal(0.0,1.0)
    p = c1*p + (1/beta)**0.5*c2*gauss_rand   # Thermostat step
    etherm-=p**2/2.0
    acc = -dvdt(q,p)
    p = p + (deltat/2.0)*acc                 # Momentum update step
    q = q + (deltat)*p                       # Position update step
    acc = -dvdt(q,p)
    p = p + (deltat/2.0)*acc                 # Momentum update step
    
    etherm+=p**2/2.0
    gauss_rand = np.random.normal(0.0,1.0)
    p = c1*p + (1/beta)**0.5*c2*gauss_rand   # Thermostat step
    etherm-=p**2/2.0
    
    
    #evol_q = q + p*deltat + 0.5*acc*deltat**2
    
    #evol_acc = -dvdt(evol_q)
    #evol_p = p + 0.5*(acc+evol_acc)*deltat
    return [q,p]
    
    

def velocity_verlet(q,p,dvdt,deltat,Tarr):
    t=0.0
    T = Tarr[len(Tarr)-1]
    pspace = [q,p]
    etherm = np.zeros_like(p)
    etherm = np.array(etherm)
    tarr.append(t)
    qarr.append(pspace[0])
    parr.append(pspace[1])
    ethermarr.append(etherm)
    count=0
    while (t<=T):
        pspace = velocity_verlet_step(pspace,dvdt,deltat,etherm)
        t+=deltat
        count+=1
        #print(pspace)
        #print(t)
        if(count%10 ==0):
            tarr.append(t)
            qarr.append(pspace[0])
            parr.append(pspace[1])
            ethermarr.append(etherm.copy())
            #print(ethermarr)
            #print('here')
            #print(etherm.copy)
        
        
q =  np.zeros((N,dimension))
p =  np.zeros((N,dimension))
qarr = []
parr = []
tarr = []
ethermarr= []

q = [[1.0]]
p = [[0.0]]

q = np.array(q)
p = np.array(p)

T = 2000*np.pi
Tarr = np.arange(0,T+0.01,np.pi/20)

velocity_verlet(q,p,dvdt,deltat,Tarr)

qarr = np.array(qarr)
parr = np.array(parr)

energyarr = parr**2/2 + potential(qarr)
ethermarr = np.array(ethermarr)

kinarr = parr[:,0,0]**2/2

kinmean = np.zeros_like(kinarr)

for i in range(len(kinmean)):
    kinmean[i] = np.sum(kinarr[:i])/(i+1)
    
plt.plot(tarr,kinmean)


#plt.plot(tarr[1:],ethermarr[1:,0,0]+energyarr[1:,0,0])
