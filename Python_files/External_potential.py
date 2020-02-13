#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 12:03:52 2019

@author: vgs23
"""
import numpy as np
from matplotlib import pyplot as plt

def potential(q):
    """
    The quartic potential 0.25 x^4
    
    Args:
        q (1d-ndarray): particle coordinates
        
    Returns
        pot (1d-ndarray): the potential
    """
    return 0.25*q**4

def dpotential(q):
    """
    The derivative of the potential w.r.t coordinates.
    
    Args:
        q (1d-ndarray): particle coordinates
        
    Returns
        dpot (1d-ndarray): the gradient of the potential
    """
    return q**3

def VV(p,q,dt):
   
    p-= (dt/2.0)*dpotential(q)
    #print(p,q)
    q+= p*dt
    p-= (dt/2.0)*dpotential(q)

    

dt = 1e-2
T=10.0
p0=-1.74976547
q0=0.0
p=np.array([p0])
q=np.array([q0])

parr = []
qarr=[]
qarr.append(q0)
parr.append(p0)
tarr = []
tarr.append(0.0)

def time_propagate(T):
    t=0.0
    while(abs(t-T)>1e-3):
        VV(p,q,dt)
        parr.append(p[0])
        qarr.append(q[0])
        
        #print(p,q,t)
        #print(q,'analytic',(q0*np.cos(t) + p0*np.sin(t)-q)/q)
        t+=dt
        tarr.append(t)
        #print(t,'t')

print(T)
N=200
#for i in range(N):
    #time_propagate(2*T/N)
#print('gap')
#time_propagate(T)
#print('gap')
#time_propagate(T)
#print('gap')
#time_propagate(T)
#print('here',q)  
    
    
#print(tarr,qarr)
#plt.plot(tarr,qarr)
#plt.plot(tarr,q0*np.cos(tarr)+p0*np.sin(tarr),color='g')
#plt.show()
    
#-----------------------------------------------------------
if(0):
    n_beads = 4
    Transf = np.zeros((n_beads,n_beads))
    
    for l in range(n_beads):
        for n in range(int(n_beads/2)):
            if(n==0):
                Transf[l,n] = 1/n_beads**0.5
            else:
                Transf[l,2*n-1] =(2/n_beads)**0.5\
                        *np.cos(-2*np.pi*n*(l+1)/n_beads) 
                #print(-2*np.pi*n*(l-int(n_beads/2))/n_beads)
                Transf[l,2*n] = (2/n_beads)**0.5\
                        *np.sin(2*np.pi*n*(l+1)/n_beads)# 
    
    print(Transf)
    A = np.array([0,1.8,2,-3])
    B= np.matmul(Transf.T,A)
    print(B)
    C = np.matmul(Transf,B)
    print(C)
    import scipy
    D= scipy.fftpack.rfft(A)
    rearrr = np.ones(n_beads)*(-(2/n_beads)**0.5)
    print(rearrr)
    rearrr[0] = 1/n_beads**0.5
    rearrr[n_beads-1] = 1/n_beads**0.5
    E = D*rearrr
    F = E*np.array(1/rearrr)#np.array([1/n_beads**(-0.5),-(2/n_beads)**(-0.5),-(2/n_beads)**(-0.5),-(2/n_beads)**(-0.5),-(2/n_beads)**(-0.5)])
    G = scipy.fftpack.irfft(F)
    print(G)
#print(np.fft.rfft(A))