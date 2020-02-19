#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 14:40:08 2019

@author: vgs23
"""
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy,copy
from numpy import linalg as LA
from numpy.fft import fft,ifft
from multiprocessing import Process
from Velocity_verlet import vv_step,vv_step_nc_thermostat,vv_step_constrained,vv_step_qcmd_thermostat,vv_step_qcmd_adiabatize
import Velocity_verlet
import MD_System

tarr = []
ethermarr= []
qarr =[]
parr = []
energyarr =[]
global count
count=0
def time_evolve_nc(CMD,Matsubara,M,swarmobj,dpotential,deltat,T,rng):
    t=0.0
    Velocity_verlet.set_therm_param(deltat,2*MD_System.omega)
    while (abs(t-T)>2e-3):
        vv_step_nc_thermostat(CMD,Matsubara,M,swarmobj,dpotential,deltat,rng)
        t+=deltat
 
def time_evolve_qcmd(swarmobj,QC_q,QC_p,lambda_curr,dpotential,deltat,T,rng):
    global count
    t=0.0
    Velocity_verlet.set_therm_param(deltat,2*MD_System.omega)
    while (abs(t-T)>1e-4):
        #vv_step_constrained(swarmobj,QC_q,QC_p,lambda_curr,dpotential,deltat,1)
        vv_step_qcmd_adiabatize(swarmobj,QC_q,QC_p,lambda_curr,dpotential,deltat,rng)
        t+=deltat
        count+=1
        if(0):#count%5==0):
            print('t',t)
            print('q',swarmobj.q[10])
            plt.plot(swarmobj.q[10][0],swarmobj.q[10][1])
            plt.scatter(swarmobj.q[10][0],swarmobj.q[10][1])
            #plt.scatter(QC_q[0][0],QC_q[0][1])
            plt.show()

def time_evolve_qcmd_therm(QCMD,swarmobj,QC_q,QC_p,lambda_curr,dpotential,deltat,T,rng):
    t=0.0
    while (abs(t-T)>1e-4):
        #vv_step_constrained(swarmobj,QC_q,QC_p,lambda_curr,dpotential,deltat)
        vv_step_qcmd_thermostat(QCMD,swarmobj,QC_q,QC_p,lambda_curr,dpotential,deltat,rng)
        t+=deltat
        
def time_evolve(CMD,Matsubara,M,swarmobj,dpotential,deltat,T):
    
    """
    This function time evolves the given dynamical quantities of the system
    
    Arguments:
        Swarmobj : Swarm class object
        dpotential: Derivative of the system potential, to compute acceleration
        deltat : Time step size
        T : Duration of time evolution
        
    """
    
    t=0.0
    #tarr.append(t)
    #qarr.append(copy(swarmobj.q))
    #parr.append(copy(swarmobj.p))
    #energyarr.append(copy(swarmobj.sys_kin() + swarmobj.sys_pot() ))
    count=0
    
    while (abs(t-T)>2e-3):
        vv_step(CMD,Matsubara,M,swarmobj,dpotential,deltat)
        #print(swarmobj.q)
        t+=deltat
        count+=1
        
        #if(count%10 ==0):
#            tarr.append(t)
#            qarr.append(copy(swarmobj.q))
#            parr.append(copy(swarmobj.p))
#            energyarr.append(copy(swarmobj.sys_kin() + swarmobj.sys_pot() ))