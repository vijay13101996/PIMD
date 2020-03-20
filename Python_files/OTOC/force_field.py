#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 12:04:25 2020

@author: vgs23
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp,odeint
import scipy 
n_particles = 15
n_dim = 2

rng = np.random.RandomState(1)
q = rng.rand(n_particles,n_dim)#np.zeros((n_particles,n_dim)) #
p = np.zeros((n_particles,n_dim))
dt = 1e-3
m=1.0

print('shape',np.shape(q))

#print('q',q)
#print('flatten q',q.flatten())
#print('normal q',q.reshape(n_particles,2))

N = n_particles*n_dim
print('N',N)
def func(y,t):
    
    q = y[:N].reshape(n_particles,n_dim)
    p = y[N:2*N].reshape(n_particles,n_dim)
    
    dydt = np.concatenate((p.flatten(),-force_dpotential(q).flatten()))
    return dydt

if(0):
    q[0][0] = 1.0/2
    q[1][0] = -1.0/2
    p[0][1] = 1.0
    p[1][1] = -1.0

def scalar_dpotential(R):
    return 1/R**2#2*R*np.exp(-R**2)#

def scalar_ddpotential(R):
    return -2/R**3

def force_dpotential(q):
    dpotential = np.zeros_like(q)
    for i in range(len(q)):
        for j in range(len(q)):
            if(i!=j):
                R = np.sum((q[i]-q[j])**2)**0.5
                unit = (q[i]-q[j])/R
                dpotential[i]+=scalar_dpotential(R)*unit
       
    return dpotential         
 
def ddpotential(q):
    
               
def vv_step(p,q):
    p[:]-=(dt/2)*force_dpotential(q)
    q[:]+=(dt/m)*p
    p[:]-=(dt/2)*force_dpotential(q)
    
T = 0.5
t=0.0
x_arr = []
y_arr = []
n=0

if(1):
    y0=np.concatenate((q.flatten(), p.flatten()))
    #print(y0)
    sol = scipy.integrate.odeint(func,y0,[0,T])
    
    q = sol[1,:N]
    q = q.reshape(n_particles,n_dim)
    p = sol[1,N:2*N]
    p = p.reshape(n_particles,n_dim)
    print(p)

if(0):
    while(t<=T):
        vv_step(p,q)
        #print('hi',np.shape(x_arr))
        x_arr.extend(q[:,0])
        y_arr.extend(q[:,1])
        
        if(0):#n%20==0):
            print('hi',np.shape(x_arr))
            plt.scatter(x_arr,y_arr,s=1)#q[:,0],q[:,1])
            plt.show()
        t+=dt
        n+=1
                
    print(p)