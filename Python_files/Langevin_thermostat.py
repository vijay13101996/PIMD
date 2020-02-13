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
from Velocity_verlet import vv_step, vv_step_thermostat, vv_step_qcmd_thermostat
#from MD_System import gamma
import MD_System
import Velocity_verlet

tarr = []
ethermarr= []
qarr =[]
parr = []
kenergyarr =[]
penergyarr =[]
#renergyarr =[]
def potential(x,y):
    K=0.49
    r_c = 1.89
    r= (x**2 + y**2)**0.5
    V_rh = (K/2)*(r-r_c)**2
    
    D0 = 0.18748
    alpha = 1.1605
    r_c = 1.8324
    V_cb = D0*(1 - np.exp(-alpha*(r-r_c)))**2
    return V_cb

def qcmd_thermalize(swarmobj,QC_q,QC_p,lambda_curr,dpotential,deltat,thermtime,rng):
    t=0.0
    etherm = np.zeros(1)
    tarr.append(t)
    Velocity_verlet.set_therm_param(deltat,MD_System.gamma)
    count=0
    x_ar= np.arange(1,1.5,0.01)
    y_ar = np.arange(1,1.5,0.01)
    X,Y = np.meshgrid(x_ar,y_ar)
    
    
    while (t<=thermtime):
        #print('qbefore',swarmobj.q[10])
        vv_step_qcmd_thermostat(swarmobj,QC_q,QC_p,lambda_curr,dpotential,deltat,rng)
        t+=deltat
        if(0):#count%10==0):
            print('t',t)
            print('q',swarmobj.q[17])
            
            axes = plt.gca()
            axes.set_xlim([1.2,1.4])
            axes.set_ylim([1.2,1.4])
            
            #plt.plot(swarmobj.q[62][0],swarmobj.q[62][1])
            #plt.scatter(swarmobj.q[62][0],swarmobj.q[62][1])
            #plt.scatter(QC_q[62][0],QC_q[62][1])
            
            
            #plt.imshow(potential(X,Y))#,origin='lower')
            #plt.plot(QC_q[:,0],QC_q[:,1])
            plt.scatter(QC_q[:,0],QC_q[:,1])
            plt.show()
        count+=1

def thermalize(CMD,Matsubara,M,swarmobj,dpotential,deltat,thermtime,rng): # Thermalize is working well.
    
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
    #qarr.append(copy(swarmobj.q))
    #parr.append(copy(swarmobj.p))
    #kenergyarr.append(swarmobj.sys_kin()/(swarmobj.N*MD_System.n_beads))# + swarmobj.sys_pot())
    #penergyarr.append(swarmobj.sys_pot())
    #ethermarr.append(etherm[0])
    #parr.append(swarmobj.p.copy())
    #count=0
    Velocity_verlet.set_therm_param(deltat,MD_System.gamma)
    
    while (t<=thermtime):
        vv_step_thermostat(CMD,Matsubara,M,swarmobj,dpotential,deltat,etherm,rng)
        t+=deltat
        #count+=1
        #print(etherm)
        #print(pspace)
        #print(t)
            #tarr.append(t)
            
            #qarr.append(swarmobj.q.copy())
            #parr.append(copy(swarmobj.p))
            #kenergyarr.append(swarmobj.sys_kin()/(swarmobj.N*MD_System.n_beads)) #+ swarmobj.sys_pot())
            #penergyarr.append(swarmobj.sys_pot())
            #print()
            #ethermarr.append(etherm[0])
            #parr.append(swarmobj.p.copy())
    #print('hereh')
    #print(np.shape(energyarr),np.shape(ethermarr))
    #plt.plot(tarr,np.array(ethermarr)+np.array(kenergyarr))
    #plt.plot(tarr,kenergyarr)#+ethermarr)
    #plt.show()
    #print('potential')
    #print(penergyarr)
    #print('rspa')
    #print(parr)