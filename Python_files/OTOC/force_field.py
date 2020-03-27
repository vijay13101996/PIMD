
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 12:04:25 2020

@author: vgs23
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint
import scipy 
from matplotlib.animation import FuncAnimation

n_particles = 20
n_dim = 2

rng = np.random.RandomState(1)
q = rng.rand(n_particles,n_dim)#np.zeros((n_particles,n_dim)) #
p = np.zeros((n_particles,n_dim))
dt = 1e-2
m=1.0

print('shape',np.shape(q))

#print('q',q)
#print('flatten q',q.flatten())
#print('normal q',q.reshape(n_particles,2))

N = n_particles*n_dim
print('N',N)


def func(y,t): 
    global m
    q = y[:N].reshape(n_particles,n_dim)
    p = y[N:2*N].reshape(n_particles,n_dim)
    n_mmat = n_particles*n_dim*n_dim
    Mpp = y[2*N:2*N+n_mmat].reshape(n_particles,n_dim,n_dim)
    Mpq = y[2*N+n_mmat:2*N+2*n_mmat].reshape(n_particles,n_dim,n_dim)
    Mqp = y[2*N+2*n_mmat:2*N+3*n_mmat].reshape(n_particles,n_dim,n_dim)
    Mqq = y[2*N+3*n_mmat:2*N+4*n_mmat].reshape(n_particles,n_dim,n_dim)
    dd_mpp = -np.matmul(ddpotential(q),Mqp)
    dd_mpq = -np.matmul(ddpotential(q),Mqq)
    #print(q,p)
    dydt = np.concatenate((p.flatten(),-force_dpotential(q).flatten(),dd_mpp.flatten(),dd_mpq.flatten(),Mpp.flatten()/m, Mpq.flatten()/m  ))
    return dydt

if(0):
    q[0][0] = 1.0/2
    q[1][0] = -1.0/2
    p[0][1] = 1.0
    p[1][1] = -1.0

def scalar_dpotential(R):
    return 2*R*np.exp(-R**2)# 1/R**2#2*R*np.exp(-R**2)#

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

def vector_ddpotential(R_vector,R):
    ret = R**2*np.identity(n_dim) - np.outer(R_vector,R_vector)
    ret*=(1/R**3)
    return ret 

def ddpotential(q):
    ddpotential = np.zeros((n_particles,n_dim,n_dim))
    for i in range(n_particles):
        for j in range(n_particles):
            if(i!=j): 
                R_vector = q[i]-q[j]
                R = np.sum((R_vector)**2)**0.5
                unit = R_vector/R
                ddpotential[i]+= scalar_ddpotential(R)*np.outer(unit,unit) + scalar_dpotential(R)*vector_ddpotential(R_vector,R)
    return ddpotential

def vv_step(p,q):
    p[:]-=(dt/2)*force_dpotential(q)
    q[:]+=(dt/m)*p
    p[:]-=(dt/2)*force_dpotential(q)
    
T = 10.0
t=0.0
x_arr = []
y_arr = []
n=0

if(1):
    Mpp= np.zeros((n_particles,n_dim,n_dim))
    Mpq= np.zeros_like(Mpp)
    Mqp= np.zeros_like(Mpp)
    Mqq= np.zeros_like(Mpp)

    Mpp = np.array([np.identity(n_dim) for i in range(n_particles)])
    Mqq = Mpp.copy()
    y0=np.concatenate((q.flatten(), p.flatten(),Mpp.flatten(),Mpq.flatten(),Mqp.flatten(),Mqq.flatten() ))
    #print(y0)
    sol = scipy.integrate.odeint(func,y0,np.linspace(0,T,20))
    
    q = sol[1,:N]
    q = q.reshape(n_particles,n_dim)
    p = sol[1,N:2*N]
    p = p.reshape(n_particles,n_dim)
    

    print(p)

if(0):
    
    fig,ax = plt.subplots()
    xdata,ydata= [],[]
    ln, = plt.plot([],[],'ro')

    def init():
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        return ln,

    def propagate():
        global t,n
        vv_step(p,q)
        #print('hi',np.shape(x_arr))
        x_arr.extend(q[:,0])
        y_arr.extend(q[:,1])
        t+=dt 
        n+=1

    def update(frame):
        propagate()
        xdata = x_arr
        ydata = y_arr
        ln.set_data(xdata,ydata)
        return ln,

    ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 100),
                    init_func=init, blit=True)
    plt.show()
if(0):
        while(t<=T):
            vv_step(p,q)
            #print('hi',np.shape(x_arr))
            x_arr.extend(q[:,0])
            y_arr.extend(q[:,1])
            
            if(0):#(n%20==0):
                print('hi',np.shape(x_arr))
                plt.scatter(x_arr,y_arr,s=1)#q[:,0],q[:,1])
                plt.show()
            t+=dt
            n+=1
                
        print('p',p)
        print('q',q)
