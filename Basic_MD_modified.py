#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 16:27:39 2019

@author: vgs23
"""

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy,copy
from numpy import linalg as LA
from numpy.fft import fft,ifft
from multiprocessing import Process
import cProfile
import multiprocessing as mp
from functools import partial


deltat = np.pi/1000
N = 100
dimension = 1
gamma = 1.0
beta = 1.0

def dpotential(q):
    """
    The derivative of the potential w.r.t coordinates.
    
    Args:
        q (1d-ndarray): particle coordinates
        
    Returns
        dpot (1d-ndarray): the gradient of the potential
    """
    return q**3

def potential(q):
    """
    The quartic potential 0.25 x^4
    
    Args:
        q (1d-ndarray): particle coordinates
        
    Returns
        pot (1d-ndarray): the potential
    """
    return 0.25*q**4

def pos_op(q,p):
    """ 
    Position operator
    
    Args:
        q (1d-ndarray): particle coordinates
        
    Returns
        q (1d-ndarray): particle coordinates
    """
    return q.copy()

class swarm:
    """
    A class of particles and their associated dynamical variables
    
    """
    
    def __init__(self):
        self.q = None
        self.p = None
        self.N = N
        self.m = 1.0
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
        
        self.poten = np.sum(potential(self.q))
        return self.poten
    
    def sys_grad_pot(self):
        
        """
        Computes and returns gradient of the potential (a vector)
        
        """

        grad_pot = dpotential(self.q)
        return grad_pot
        
def vv_step(swarmobj,dpotential,deltat):
    
    """
    
    This function implements one step of the Velocity Verlet algorithm
    
    Arguments:
        Swarmobj : Swarm class object
        dpotential: Derivative of the system potential, to compute acceleration
        deltat : Time step size
    """
    swarmobj.p -= (deltat/2.0)*swarmobj.sys_grad_pot()
    swarmobj.q += (deltat/swarmobj.m)*swarmobj.p              # Position update step
    swarmobj.p -= (deltat/2.0)*swarmobj.sys_grad_pot()

c1 = np.exp(-(deltat/2.0)*gamma)
c2 = (1-c1**2)**0.5
    

def vv_step_thermostat(swarmobj,dpotential,deltat,etherm,rng):
    
    """ 
    
    This function implements one step of the Velocity verlet, along with the 
    Langevin thermostat
    
    Arguments:
        Swarmobj : Swarm class object
        dpotential: Derivative of the system potential, to compute acceleration
        deltat : Time step size
        etherm : This variable is auxiliary, helps take care of correct
                 implementation of the thermostat.
        rng: A randomstate object, used to ensure that the code is seeded
             consistently

    """    
    etherm[0] += swarmobj.sys_kin()
         # Can be passed as an argument if reqd.
    
    gauss_rand = rng.normal(0.0,1.0,np.shape(swarmobj.q))
    swarmobj.p = c1*swarmobj.p + swarmobj.sm*(1/beta)**0.5*c2*gauss_rand 
    
    # Done! Taken care of!: 
       #  Be extra careful in the above step when you vectorize the code.
    etherm[0] -= swarmobj.sys_kin()
   
    vv_step(swarmobj,dpotential,deltat)
   
    etherm[0] += swarmobj.sys_kin()
    gauss_rand = rng.normal(0.0,1.0,np.shape(swarmobj.q))
    swarmobj.p = c1*swarmobj.p + swarmobj.sm*(1/beta)**0.5*c2*gauss_rand
    etherm[0] -= swarmobj.sys_kin()
    
def time_evolve(swarmobj,dpotential,deltat,T):
    
    """
    This function time evolves the given dynamical quantities of the system
    
    Arguments:
        Swarmobj : Swarm class object
        dpotential: Derivative of the system potential, to compute acceleration
        deltat : Time step size
        T : Duration of time evolution
        
    """
    
    t=0.0
    tarr.append(t)
    qarr.append(copy(swarmobj.q))
    parr.append(copy(swarmobj.p))
    energyarr.append(copy(swarmobj.sys_kin() + swarmobj.sys_pot() ))
    count=0
    while (t<=T):
        vv_step(swarmobj,dpotential,deltat)
        t+=deltat
        count+=1
        #print(pspace)
        #print(t)
        if(count%10 ==0):
            tarr.append(t)
            qarr.append(copy(swarmobj.q))
            parr.append(copy(swarmobj.p))
            energyarr.append(copy(swarmobj.sys_kin() + swarmobj.sys_pot() ))
            #ethermarr.append(etherm.copy())
    
def thermalize(swarmobj,dpotential,deltat,thermtime,rng): # Thermalize is working well.
    
    """
    This function 'thermalizes' the given system using Langevin thermostat
    
    Arguments:
        Swarmobj : Swarm class object
        dpotential: Derivative of the system potential, to compute acceleration
                    A function that takes position as argument.
        deltat : Time step size (float)
        thermtime : Duration of time evolution (float)
        rng: A randomstate object, used to ensure that the code is seeded
             consistently. It is passed to the function vv_step_thermostat
             
    """
    
    t=0.0
    etherm = np.zeros(1)
    tarr.append(t)
    #print(swarmarr)
    qarr.append(copy(swarmobj.q))
    parr.append(copy(swarmobj.p))
    energyarr.append(copy(swarmobj.sys_kin() + swarmobj.sys_pot() ))
    ethermarr.append(etherm[0])
    count=0
    while (t<=thermtime):
        vv_step_thermostat(swarmobj,dpotential,deltat,etherm,rng)
        t+=deltat
        count+=1
        #print(pspace)
        #print(t)
        if(count%10 ==0):
            tarr.append(t)
            
            qarr.append(swarmobj.q.copy())
            parr.append(copy(swarmobj.p))
            energyarr.append(swarmobj.sys_kin() + swarmobj.sys_pot())
            #print()
            ethermarr.append(etherm[0])
    #return swarmobj

#def corr_instance(swarmobj,A,B,tcf_tarr):
#    thermalize(swarmobj,dpotential,deltat,2*np.pi)
#    print(swarmobj.q[0])
#    A_0 = A(swarmobj.q,swarmobj.p)
#    tcf_instance = np.zeros_like(tcf_tarr)
#    tcf_instance[0]= A_0*A_0
#    # The above line have to be treat on a case to case basis
#    for i in range(1,len(tcf_tarr)):
#        time_evolve(swarmobj, dpotential, deltat, tcf_tarr[i]-tcf_tarr[i-1])
#        B_t = B(swarmobj.q,swarmobj.p)
#        tcf_instance[i] = np.dot(A_0,B_t)
#    return tcf_instance
#    #thermalize(swarmobj,dpotential,deltat,2*np.pi)

            
def corr_function(swarmobj,A,B,time_corr,rng_ind):
    
    """ 
    This function computes the time correlation function for one instance
    of the system prepared by seeding with rng_ind
    
    Arguments:
        swarmobj: Swarm class object
        A: First operator, to compute correlation (A function of p,q)
        B: Second operator, to compute correlation (A function of p,q)
        time_corr : Total duration of correlation (float)
        rng_ind : Integer, used to seed the Randomstate object
        
    """
    rng = np.random.RandomState(rng_ind)
    tcf_tarr = np.arange(0,time_corr+0.0001,time_corr/500.0)
    tcf = np.zeros_like(tcf_tarr)
    n_approx = 10
    thermalize(swarmobj,dpotential,deltat,10*np.pi,rng)
      
    for j in range(n_approx):
        A_0 = A(swarmobj.q,swarmobj.p)
        tcf[0]+= np.sum(A_0*A_0)
        # The above line have to be treated on a case to case basis
        for i in range(1,len(tcf)):
            time_evolve(swarmobj, dpotential, deltat, tcf_tarr[i]-tcf_tarr[i-1])
            B_t = B(swarmobj.q,swarmobj.p)
            tcf[i] += np.sum(A_0*B_t)
        thermalize(swarmobj,dpotential,deltat,2*np.pi,rng)
    tcf/=(n_approx*swarmobj.N)  #Change when you increase dimension
    
    return tcf

def corr_function_upgrade(swarmobj,A,B,time_corr,rng_ind):
    rng = np.random.RandomState(rng_ind)
    tcf_tarr = np.arange(0,time_corr+0.0001,time_corr/500.0)
    tcf = np.zeros_like(tcf_tarr)
    
    tcf_taux = np.arange(0,2*time_corr+0.0001,time_corr/500.0)
    #A_arr = np.array([np.zeros_like(swarmobj.q) for i in range(len(tcf_taux))])
    A_arr = np.zeros((len(tcf_taux),)+swarmobj.q.shape)
    B_arr = np.zeros_like(A_arr)
    print(np.shape(A_arr))
    
    n_approx = 10
     #The above line have to be treated on a case to case basis
    for j in range(n_approx):
        A_arr[0] = A(swarmobj.q,swarmobj.p)
        B_arr[0] = B(swarmobj.q,swarmobj.p)
        for i in range(1,len(tcf_taux)):
            time_evolve(swarmobj, dpotential, deltat, tcf_taux[i]-tcf_taux[i-1])
            A_arr[i] = A(swarmobj.q,swarmobj.p)
            B_arr[i] = B(swarmobj.q,swarmobj.p)
        
        #print(A_arr)
        A_arr[len(tcf):] = 0.0*A_arr[len(tcf):]
        #A_arr = A_arr.T
        #B_arr = B_arr.T
        #print('here')
        #print(A_arr)
        
        A_tilde = np.fft.fft(A_arr,axis=0)
        B_tilde = np.fft.fft(B_arr,axis=0)
        
        tcf_tilde = np.conj(A_tilde)*B_tilde
        #print(tcf_tilde)
        #print('there')
        tcf_cr = np.fft.ifft(tcf_tilde,axis=0)
        #print(np.imag(tcf_cr))
        #tcf_cr = tcf_cr.T
        tcf_cr = np.sum(tcf_cr,axis=1)
        #print(np.imag(tcf_cr))
        tcf+= np.real(tcf_cr[:len(tcf)])
        #A_arr = A_arr.T
        #B_arr = B_arr.T
        #print(tcf)
        
#        for i in range(len(tcf)):
#            for k in range(len(tcf)):
#                #print(A_arr[i+j]*B_arr[j] + B_arr[i+j]*A_arr[j])
#                tcf[i] += np.sum(A_arr[i+k]*B_arr[k] + B_arr[i+k]*A_arr[k]) 
    #tcf[i] += np.sum(A_0*B_t)
        thermalize(swarmobj,dpotential,deltat,2*np.pi,rng)
    tcf/=(n_approx*swarmobj.N*len(tcf))
    return tcf
    
    
        
tarr = []
ethermarr= []
qarr =[]
parr = []
energyarr =[]

T = 2*np.pi

swarmobject = swarm()
swarmobject.q = np.zeros(swarmobject.N)
rand_boltz = np.random.RandomState(1000)
swarmobject.p = rand_boltz.normal(0.0,swarmobject.m/beta,np.shape(swarmobject.q))

pool = mp.Pool(mp.cpu_count())
func = partial(corr_function_upgrade,swarmobject,pos_op,pos_op,20*np.pi)

n_inst = 10
results = pool.map(func, range(n_inst))
pool.close()
pool.join()
tcf = sum(results,0)/n_inst


#tcf = corr_function(swarmobject, pos_op,pos_op,20*np.pi,1)
tcf_tarr = np.arange(0,20*np.pi+0.0001,20*np.pi/500.0)

plt.plot(tcf_tarr,tcf)#, tcf1)

cProfile.run("corr_function_upgrade(swarmobject, pos_op,pos_op,20*np.pi,1)")