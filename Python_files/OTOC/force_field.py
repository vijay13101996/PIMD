#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 12:04:25 2020

@author: vgs23
"""

import numpy as np
from matplotlib import pyplot as plt


n_particles = 2
n_dim = 2
q = np.zeros((n_particles,n_dim)) #np.random.rand(n_particles,n_dim)#
p = np.zeros((n_particles,n_dim))
dt = 1e-2
m=1.0

if(1):
    q[0][0] = 1.0/3
    q[1][0] = -1.0/3
    p[0][1] = 1.0
    p[1][1] = -1.0

def scalar_dpotential(R):
    return 1/R**2#2*R*np.exp(-R**2)#

def force_dpotential(q):
    dpotential = np.zeros_like(q)
    for i in range(len(q)):
        for j in range(len(q)):
            if(i!=j):
                R = np.sum((q[i]-q[j])**2)**0.5
                unit = (q[i]-q[j])/R
                dpotential[i]+=scalar_dpotential(R)*unit
       
    return dpotential         
                
def vv_step(p,q):
    p[:]-=(dt/2)*force_dpotential(q)
    q[:]+=(dt/m)*p
    p[:]-=(dt/2)*force_dpotential(q)
    
T = 10.0
t=0.0
x_arr = []
y_arr = []
n=0
#fig = plt.figure()
while(t<=T):
    vv_step(p,q)
    #print('hi',np.shape(x_arr))
    x_arr.extend(q[:,0])
    y_arr.extend(q[:,1])
    
    if(n%20==0):
        print('hi',np.shape(x_arr))
        plt.scatter(x_arr,y_arr,s=1)#q[:,0],q[:,1])
        plt.show()
    t+=dt
    n+=1
                