#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 14:37:17 2019

@author: vgs23
"""
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy,copy
from numpy import linalg as LA
from numpy.fft import fft,ifft
from multiprocessing import Process
from Velocity_verlet import vv_step, vv_step_thermostat, vv_step_qcmd_thermostat, vv_step_baqcmd_thermostat
from Theta_constrained_sampling import theta_constrained_randomize, centrifugal_term
import MD_System
import Velocity_verlet
import psutil
import sys
from Animation import animate_2D
import scipy
import Matsubara_potential
tarr = []
ethermarr= []
qarr =[]
parr = []
kenergyarr =[]
penergyarr =[]
#renergyarr =[]


def baqcmd_thermalize(BAQCMD,swarmobj,BAQC_q,BAQC_p,U,lambda_curr,dpotential,deltat,thermtime,rng):
    t=0.0
    etherm = np.zeros(1)
    tarr.append(t)
    Velocity_verlet.set_therm_param(deltat,swarmobj.gamma)
    count=0
    x_ar= np.arange(1.0,1.5,0.01)
    y_ar = np.arange(1,1.5,0.01)
    X,Y = np.meshgrid(x_ar,y_ar)
   
    #print(np.sum(swarmobj.q**2,axis=1)**0.5)
    while (t<=thermtime):
        vv_step_baqcmd_thermostat(BAQCMD,swarmobj,BAQC_q,BAQC_p,U,lambda_curr,dpotential,deltat,rng)
        #qarr.append(np.mean(swarmobj.q,axis=2))
        t+=deltat 
        #print('t',t, np.sum(swarmobj.q[0,2,:]))
        count+=1

    #print('thermdone', np.shape(qarr))
    #animate(qarr) 

def qcmd_thermalize(QCMD,swarmobj,QC_q,QC_p,lambda_curr,dpotential,deltat,thermtime,rng):
    t=0.0
    etherm = np.zeros(1)
    tarr.append(t)
    Velocity_verlet.set_therm_param(deltat,MD_System.gamma)
    count=0
    x_ar= np.arange(1.0,1.5,0.01)
    y_ar = np.arange(1,1.5,0.01)
    X,Y = np.meshgrid(x_ar,y_ar)
    
    
    while (t<=thermtime):
        #print('qbefore',swarmobj.q[10])
        vv_step_qcmd_thermostat(QCMD,swarmobj,QC_q,QC_p,lambda_curr,dpotential,deltat,rng)
        t+=deltat
        if(0):#count%10==0):
            print('t',t)
            print('q',swarmobj.q[0])
            
            axes = plt.gca()
            #axes.set_xlim([0.0,1.5])
            #axes.set_ylim([0.0,1.5])
            
            plt.plot(swarmobj.q[0][0],swarmobj.q[0][1])
            plt.scatter(swarmobj.q[0][0],swarmobj.q[0][1])
            #plt.scatter(QC_q[62][0],QC_q[62][1])
            
            #print('QC_q',QC_q)
            #plt.imshow(potential(X,Y))#,origin='lower')
            #plt.plot(QC_q[:,0],QC_q[:,1])
            #plt.scatter(QC_q[:,0],QC_q[:,1])
            plt.show()
        count+=1

def thermalize(ACMD,CMD,Matsubara,M,swarmobj,dpotential,deltat,thermtime,rng): # Thermalize is working fine.
    
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
    
    #---------------------------------------------------------------------------------------
    #tarr.append(t)
    #print(swarmarr)
    #qarr.append(copy(swarmobj.q))
    #parr.append(copy(swarmobj.p))
    #kenergyarr.append(swarmobj.sys_kin()/(swarmobj.N*MD_System.n_beads))# + swarmobj.sys_pot())
    #penergyarr.append(swarmobj.sys_pot())
    #ethermarr.append(etherm[0])
    #parr.append(swarmobj.p.copy())
    #count=0
    #--------------------------------------------------------------------------------------- Beware of memory leaks when using any of the tools above!!!!!
    tarr = []
    Earr = []
    Velocity_verlet.set_therm_param(deltat,swarmobj.gamma)
    
    while (t<=thermtime):
        if(psutil.virtual_memory()[2] > 60.0):
                    print('Memory limit exceeded, exiting...', psutil.virtual_memory()[2])
                    sys.exit()
        vv_step_thermostat(ACMD,CMD,Matsubara,M,swarmobj,dpotential,deltat,etherm,rng)
        t+=deltat
        
        #tarr.append(t)
        #Earr.append(np.sum(swarmobj.p**2/(2*swarmobj.m))/swarmobj.N)
    
    #plt.plot(tarr,Earr)
    #plt.show()
                
def vel_randomize(swarmobj, Matsubara,rng):
    #swarmobj.p = scipy.fftpack.rfft(swarmobj.p,axis=2)*swarmobj.ft_rearrange
    if(Matsubara==1):
        swarmobj.p = rng.normal(0, (swarmobj.m/swarmobj.beta)**0.5,swarmobj.q.shape) 
    else:
        swarmobj.p = rng.normal(0, (swarmobj.m/swarmobj.beta_n)**0.5,swarmobj.q.shape) 
    
    #swarmobj.p = scipy.fftpack.irfft(swarmobj.p,axis=2)*swarmobj.ft_rearrange

def Andersen_thermalize(ACMD,CMD, Matsubara, M, swarmobj, niter, thermtime, dpotential, deltat, rng):
    for i in range(niter):
        vel_randomize(swarmobj,Matsubara,rng)
        t=0.0
        while(t<=thermtime):
            vv_step(ACMD,CMD,Matsubara,M,swarmobj,dpotential,deltat)
            t+=deltat

def Theta_constrained_thermalize(M,swarmobj,theta,thermtime,dpotential,deltat,rng):
        #Sampling the q distribution
        def dpot_theta(Q):
               ret1 = dpotential(Q) 
               ret2 = centrifugal_term(swarmobj,theta)
               #Rsq = np.sum(swarmobj.w_marr**2*swarmobj.q[...,::-1]**2,axis=2) 
               #print('ret1',ret1[397], ret2[397])#Matsubara_potential.dpot_inv_harmonic_M3(Q[397]), theta**2/Rsq[397]**2)
               return ret1+ret2
        #Rsq = np.sum(swarmobj.w_marr**2*swarmobj.q[...,::-1]**2,axis=2)
               
        thermalize(0,0,1,M,swarmobj,dpot_theta,deltat,thermtime,rng)
        
        theta_constrained_randomize(swarmobj,theta,rng)
        
        t=0.0
        while(t<=thermtime):
            vv_step(0,0,1,M,swarmobj,dpotential,deltat)
            t+=deltat
        
