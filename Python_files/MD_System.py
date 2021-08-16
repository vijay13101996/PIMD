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

class swarm:
    """
    A class of particles and their associated dynamical variables
    
    """
    
    def __init__(self,N, n_b, beta_g,dim):
        self.q = None
        self.p = None
        self.N = N
        self.m = 0.5#1741.1#
        self.sm = np.sqrt(self.m)
        self.kinen = 0.0
        self.poten = 0.0 
        
        self.dimension = dim
        self.friction_param = 20.0
        self.gamma = 2*self.friction_param
        self.beta = beta_g
        self.n_beads = n_b
        self.beta_n = self.beta/self.n_beads  
        self.w_n = 1/self.beta_n
         
        arr = np.array(range(-int(self.n_beads/2),int(self.n_beads/2)+1))*np.pi/self.n_beads
        self.w_arr = 2*self.w_n*np.sin(arr)
        self.w_arr = self.rearrange()[1:]
    
        self.omega = self.friction_param
        self.mass_factor = (self.beta_n*self.w_arr[1:]/self.omega)**2 
        #print('System definition: beta, n_beads, omega',self.beta,self.n_beads,self.omega)
        self.w_arr_scaled = np.ones(self.n_beads)#*(self.omega/self.beta_n)#**2
        self.w_arr_scaled[0]=0.0
        self.w_arr_scaled = self.w_arr_scaled[1:]
  
        self.M=self.n_beads 
        self.w_marr = 2*np.arange(-int((self.M-1)/2),int((self.M-1)/2)+1)*np.pi\
            /(self.beta)
 
        self.cosw_arr = np.cos(self.w_arr)
        self.sinw_arr = np.sin(self.w_arr)

        self.ft_rearrange= np.ones(self.n_beads)*(-(2.0/self.n_beads)**0.5)
        self.ft_rearrange[0] = 1/self.n_beads**0.5

        #print('ft_rearrange',self.ft_rearrange)

        if(self.n_beads%2 == 0):
            self.ft_rearrange[self.n_beads-1]= 1/self.n_beads**0.5

    def rearrange(self):
        rearr = np.zeros(self.n_beads,dtype= int)
        
        if(self.n_beads%2!=0):
            for i in range(1,int(self.n_beads/2)+1):
                rearr[2*i-1]=-i
                rearr[2*i]=i
            rearr+=int(self.n_beads/2)
        else:
            for i in range(1,int(self.n_beads/2)):
                rearr[2*i-1]=-i
                rearr[2*i]=i
            rearr[len(rearr)-1] = -int(self.n_beads/2)
            rearr+=int(self.n_beads/2)

        return self.w_arr[rearr]
   
    def setpq(self,p,q):
        self.q = q
        self.p = p
    
    def ddpotential_ring(self):
        if(self.n_beads==1):
            return np.zeros((n_particles,n_beads,n_dim,n_dim))
        else:
            id_mat =  np.array([[np.identity(self.dimension) for i in range(self.n_beads)] for j in range(self.N)])
            return 2*self.m*self.w_n**2*id_mat


    def dpotential_ring(self):
        return self.m*self.w_n**2*(2*self.q-np.roll(self.q,1,axis=2) - np.roll(self.q,-1,axis=2))

    def potential_ring(self):
        return 0.5*self.m*self.w_n**2*((self.q-np.roll(self.q,1,axis=2))**2)

    def pos_op(self):
        """ 
        Position operator
        
        Args:
            q (1d-ndarray): particle coordinates
            
        Returns
            q (1d-ndarray): particle coordinates
        """
        
        return self.q.copy()

    def mom_op(self):
        return self.p.copy()

    def vel_op(self):
        return self.p.copy()/self.m
    
    def sys_kin(self):
        """
        Computes and returns system kinetic energy
        
        """
        
        sp = self.p.flatten()/self.sm
        self.kinen = 0.5*np.dot(sp,sp)
        return self.kinen
        
    def sys_pot(self):   # Needs change!
        
        """
        Computes and returns system potential energy
        
        """
        
        self.poten =  np.sum(potential(self.q)) + np.sum(potential_ring(self))
        return self.poten
    
    def sys_grad_pot(self): # Needs change!
        
        """
        Computes and returns gradient of the potential (a vector)
        
        """

        grad_pot = derpotential(self.q) #+ dpotential_ring(self)
        #print(grad_pot)
        return grad_pot
    
    def sys_grad_ring_pot_coord(self):
        return self.dpotential_ring(self)
    
    def sys_grad_ring_pot(self):
        grad_ring_pot = self.m*self.w_arr**2*self.q
        return grad_ring_pot


