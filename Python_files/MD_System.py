#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 14:51:47 2019

@author: vgs23
"""

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy,copy
from numpy import linalg as LA
from numpy.fft import fft,ifft
import multiprocessing as mp
from functools import partial


global dimension,gamma,beta,n_beads,w_n,w_arr,derpotential,w_arr_scaled,omega,beta_n,mass_factor, cosw_arr, sinw_arr
dimension = 1
gamma = 1.0
beta = 1
n_beads = 1
beta_n = beta/n_beads  
w_n = 1/beta_n
w_arr_scaled = np.ones(n_beads)
w_arr_scaled[0] =0.0
omega = 1.0
centroid_force = None
mass_factor = 1.0*np.ones(n_beads)
cosw_arr = np.cos(w_arr_scaled)
sinw_arr = np.sin(w_arr_scaled)

arr = np.array(range(-int(n_beads/2),int(n_beads/2+1)))*np.pi/n_beads
    
w_arr = 2*w_n*np.sin(arr)

def rearrange(w_arr):
    rearr = np.zeros(n_beads,dtype= int)
    
    if(n_beads%2!=0):
        for i in range(1,int(n_beads/2)+1):
            rearr[2*i-1]=-i
            rearr[2*i]=i
        rearr+=int(n_beads/2)
    else:
        for i in range(1,int(n_beads/2)):
            rearr[2*i-1]=-i
            rearr[2*i]=i
        rearr[len(rearr)-1] = -int(n_beads/2)
        rearr+=int(n_beads/2)

    return w_arr[rearr]

def system_definition(beta_g,n_b,dim,derivpotential):
    # Ensure that cos_warr and sin_warr values are changed as per the simulation requirements 
    global dimension,gamma,beta,beta_n,n_beads,w_n,w_arr,omega,derpotential,w_arr_scaled,mass_factor, cosw_arr, sinw_arr
    dimension = dim
    
    friction_param = 32.0

    gamma = 2*friction_param
    beta = beta_g
    n_beads = n_b
    beta_n = beta/n_beads  
    w_n = 1/beta_n
    #print('beta_n',beta_n)
    derpotential = derivpotential
    arr = np.array(range(-int(n_beads/2),int(n_beads/2)+1))*np.pi/n_beads
    
    w_arr = 2*w_n*np.sin(arr)
    
    w_arr = rearrange(w_arr)
    
    omega = friction_param
    mass_factor = (beta_n*w_arr[1:]/omega)**2 #
    print('System definition: beta, n_beads, omega',beta,n_beads,omega)
    w_arr_scaled = np.ones(n_beads)*(omega/beta_n)#**2
    w_arr_scaled[0]=0.0
    w_arr_scaled = w_arr_scaled[1:]
    
    cosw_arr = np.cos(w_arr_scaled)
    sinw_arr = np.sin(w_arr_scaled)

    print('shape', cosw_arr)
def potential(q):
    """
    The quartic potential 0.25 x^4
    
    Args:
        q (1d-ndarray): particle coordinates
        
    Returns
        pot (1d-ndarray): the potential
    """
    return 0.5*q**2

def dpotential(q):
    """
    The derivative of the potential w.r.t coordinates.
    
    Args:
        q (1d-ndarray): particle coordinates
        
    Returns
        dpot (1d-ndarray): the gradient of the potential
    """
    return q**3

def dpotential_ring(swarmobj):
    #print(w_n**2)
    return swarmobj.m*w_n**2*(2*swarmobj.q-np.roll(swarmobj.q,1,axis=2)
                    - np.roll(swarmobj.q,-1,axis=2))



def potential_ring(swarmobj):
    return 0.5*swarmobj.m*w_n**2*\
        ((swarmobj.q-np.roll(swarmobj.q,1,axis=2))**2)

def pos_op(q,p):
    """ 
    Position operator
    
    Args:
        q (1d-ndarray): particle coordinates
        
    Returns
        q (1d-ndarray): particle coordinates
    """
    return q.copy()

def mom_op(q,p):
    return p.copy()/1741.1

class swarm:
    """
    A class of particles and their associated dynamical variables
    
    """
    
    def __init__(self,N):
        self.q = None
        self.p = None
        self.N = N
        self.m = 1741.1#
        self.sm = np.sqrt(self.m)
        self.kinen = 0.0
        self.poten = 0.0 
        
    def setpq(self,p,q):
        self.q = q
        self.p = p
        
    def sys_kin(self):
        """
        Computes and returns system kinetic energy
        
        """
        
        sp = self.p.flatten()/self.sm
        self.kinen = 0.5*np.dot(sp,sp)
        return self.kinen
        
    def sys_pot(self):
        
        """
        Computes and returns system potential energy
        
        """
        
        self.poten =  np.sum(potential(self.q)) + np.sum(potential_ring(self))
        return self.poten
    
    def sys_grad_pot(self):
        
        """
        Computes and returns gradient of the potential (a vector)
        
        """

        grad_pot = derpotential(self.q) #+ dpotential_ring(self)
        #print(grad_pot)
        return grad_pot
    
    def sys_grad_ring_pot_coord(self):
        return dpotential_ring(self)
    
    def sys_grad_ring_pot(self):
        grad_ring_pot = self.m*w_arr**2*self.q
        return grad_ring_pot

    
    
#------------------------------------------------------STRAY CODE

