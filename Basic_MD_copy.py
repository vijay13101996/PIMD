#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 14:11:10 2019

@author: vgs23
"""

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from numpy import linalg as LA
import cProfile


deltat = np.pi/1000
N = 1
dimension = 1
gamma = 1.0
beta = 1.0

def dpotential(q):
    #print('here',np.ones_like(q)*(LA.norm(q))**3)
    return np.ones_like(q)*(q[0])

def potential(q):
    return 0.25*(LA.norm(q))**4

def pos_op(q,p):
    return q

class particle:
    
    def __init__(self):
        self.q = [1.0]
        self.p = [0.0]
        self.m = 1.0
        
    def setpq(self,p,q):
        self.q = q
        self.p = p
        
def sys_kin(swarm):     # Working alright
    kinen = 0.0
    for i in range(len(swarm)):
        kinen += LA.norm(np.array(swarm[i].p))**2/(2*swarm[i].m)
        
    return kinen

def sys_pot(swarm):    # Working alright
    poten = 0.0
    for i in range(len(swarm)):
        poten += potential(np.array(swarm[i].q))
        
    return poten
 
def sys_pq_assign(swarm,P,Q):
    for i in range(N):
        swarm[i].setpq(P[i],Q[i]) 
    
        
def vv_step_thermostat(swarm,dpotential,deltat,etherm):
    Q = np.array([swarm[i].q for i in range(N)])
    P = np.array([swarm[i].p for i in range(N)])
    
    c1 = np.exp(-(deltat/2.0)*gamma)
    c2 = (1-c1**2)**0.5
    
    sys_pq_assign(swarm,P,Q)
    etherm[0] += sys_kin(swarm)
    #print(etherm)
    
    gauss_rand = np.random.normal(0.0,1.0)
    P = c1*P + (1/beta)**0.5*c2*gauss_rand   # Thermostat step
    
    sys_pq_assign(swarm,P,Q)
    etherm[0] -= sys_kin(swarm)
    
    acc = -dpotential(Q)
    P = P + (deltat/2.0)*acc                 # Momentum update step
    Q = Q + (deltat)*P                       # Position update step
    acc = -dpotential(Q)
    P = P + (deltat/2.0)*acc                 # Momentum update step
    
    sys_pq_assign(swarm,P,Q)
    etherm[0] += sys_kin(swarm)
    
    gauss_rand = np.random.normal(0.0,1.0)
    P = c1*P + (1/beta)**0.5*c2*gauss_rand   # Thermostat step, remember to add mass
    
    sys_pq_assign(swarm,P,Q)
    etherm[0] -= sys_kin(swarm)
    
def vv_step(swarm,dpotential,deltat):
    Q = np.array([swarm[i].q for i in range(N)])
    P = np.array([swarm[i].p for i in range(N)])
    
    acc = -dpotential(Q)
    P = P + (deltat/2.0)*acc                 # Momentum update step
    Q = Q + (deltat)*P                       # Position update step
    acc = -dpotential(Q)
    P = P + (deltat/2.0)*acc                 # Momentum update step
    
    sys_pq_assign(swarm,P,Q)
    
def time_evolve(swarm,dpotential,deltat,T):
    t=0.0
    tarr.append(t)
    swarmarr.append(deepcopy(swarm))
    count=0
    while (t<=T):
        vv_step(swarm,dpotential,deltat)
        t+=deltat
        count+=1
        #print(pspace)
        #print(t)
        if(count%10 ==0):
            tarr.append(t)
            swarmarr.append(deepcopy(swarm))
            #ethermarr.append(etherm.copy())
            
def thermalize(swarm,dpotential,deltat,thermtime): # Thermalize is working well.
    t=0.0
    etherm = np.zeros(1)
    tarr.append(t)
    #print(swarmarr)
    swarmarr.append(deepcopy(swarm))
    ethermarr.append(etherm[0])
    count=0
    while (t<=thermtime):
        vv_step_thermostat(swarm,dpotential,deltat,etherm)
        t+=deltat
        count+=1
        #print(pspace)
        #print(t)
        if(count%10 ==0):
            tarr.append(t)
            
            swarmarr.append(deepcopy(swarm))
            #print()
            ethermarr.append(etherm[0].copy())               

def corr_function(swarm,A,B,time_corr):
    tcf_tarr = np.arange(0,time_corr+0.0001,time_corr/500.0)
    tcf = np.zeros_like(tcf_tarr)
    n_approx = 100
    thermalize(swarm,dpotential,deltat,10*np.pi)
    for j in range(n_approx):
        A_0 = A(swarm[0].q,swarm[0].p)
        tcf[0]+= np.dot(A_0,A_0)
        for i in range(1,len(tcf)):
            time_evolve(swarm, dpotential, deltat, tcf_tarr[i]-tcf_tarr[i-1])
            B_t = B(swarm[0].q,swarm[0].p)
            tcf[i] += np.dot(A_0,B_t)
        thermalize(swarm,dpotential,deltat,2*np.pi)
    tcf/=n_approx
    
    return tcf
        
        
            
swarmarr =[]
tarr = []
ethermarr= []

T = 2*np.pi       # 200*pi is long enough to thermalize.

swarm = [particle() for i in range(N)]

#print(sys_kin(swarm))

#thermalize(swarm,dpotential,deltat,T)
#time_evolve(swarm,dpotential,deltat,T)

tcf = corr_function(swarm, pos_op,pos_op,20*np.pi)
tcf_tarr = np.arange(0,20*np.pi+0.0001,20*np.pi/500.0)
print(tcf)
plt.plot(tcf_tarr,tcf)

kinarr = np.array([sys_kin(swarmarr[i]) for i in range(len(tarr))]) 
potarr = np.array([sys_pot(swarmarr[i]) for i in range(len(tarr))])

energyarr = kinarr +potarr
kinmean = np.zeros_like(kinarr)

for i in range(len(kinmean)):
    kinmean[i] = np.sum(kinarr[:i])/(i+1)
 
cProfile.run("corr_function(swarm, pos_op,pos_op,20*np.pi)")
#print(energyarr)
#plt.plot(tarr,energyarr)
#print(ethermarr)
#print(swarmarr[1][0].q, swarmarr[100][0].q)
#plt.plot(tarr[1:],energyarr[1:]+ethermarr[1:])
