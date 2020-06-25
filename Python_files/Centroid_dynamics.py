#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 14:47:18 2019

@author: vgs23
"""

import numpy as np
from Correlation_function import corr_function_upgrade
import MD_System
import multiprocessing as mp
from functools import partial
import Velocity_verlet
import importlib
import scipy
import Centroid_mean_force
import pickle
import time
from matplotlib import pyplot as plt
import Ring_polymer_dynamics
start_time =time.time()

def CMD_instance(beads,dpotential,beta): 
    
    MD_System.system_definition(beta,beads,2,dpotential)
    importlib.reload(Velocity_verlet)
    
    N = 100
    deltat = 5.0
    NQ = 20
    Q = np.linspace(0.5,3,NQ)
    n_sample = 5
    Centroid_mean_force.mean_force_calculation(dpotential,N,deltat,Q,n_sample)
     
    f = open("/home/vgs23/Pickle_files/CMD_Force_B_{}_NB_{}_N_{}_nsample_{}_deltat_{}_NQ_{}.dat".format(MD_System.beta,MD_System.n_beads,N,n_sample,deltat,NQ),'rb') 
      
    data = np.loadtxt(f)
    Q = data[:,0]
    CMD_force = data[:,1]
   
    #plt.plot(Q,CMD_force)
    #plt.show()
    xforce = scipy.interpolate.interp1d(Q,CMD_force,kind='cubic')
    return xforce   ### BE CAREFUL!!! This is wrong convention followed!

def ACMD_instance(rng,N,beads,dpotential,beta,T,deltat):
    
    swarmobject = MD_System.swarm(N)
    MD_System.system_definition(beta,beads,2,dpotential)
    importlib.reload(Velocity_verlet)
    Velocity_verlet.set_therm_param(deltat,MD_System.gamma) 
    
    swarmobject.q = np.zeros((swarmobject.N,MD_System.dimension,MD_System.n_beads)) + 1.0
    rand_boltz = np.random.RandomState(rng)
    swarmobject.p = rand_boltz.normal(0.0,swarmobject.m/(MD_System.beta),np.shape(swarmobject.q))
    
    pool = mp.Pool(mp.cpu_count()-6)    ### Using 6 cores is the safest!
    print('Using', mp.cpu_count()-6, 'cores')
    Matsubara = 0
    CMD=0
    ACMD=1
    
    n_inst = 6
    func = partial(corr_function_upgrade,ACMD,CMD,Matsubara,beads,swarmobject,dpotential,MD_System.pos_op,MD_System.pos_op,T,deltat)
    results = pool.map(func, range(n_inst)) 
    pool.close()
    pool.join()
    return np.sum(results,0)/n_inst

def compute_tcf_ACMD(n_ACMD_instance,N,beads,dpotential,beta,T,deltat):
    
    for i in range(n_ACMD_instance):
        print('ACMD initiated')
        tcf =  ACMD_instance((i+1)*100,N,beads,dpotential,beta,T,deltat)
        f = open('/home/vgs23/Pickle_files/ACMD_vv_tcf_N_{}_B_{}_inst_{}_T_{}_dt_{}_NB_{}_SAMP3.dat'.format(N*100,MD_System.beta,i,T,deltat,MD_System.n_beads),'wb')
        pickle.dump(tcf,f)
        #plt.plot(tcf_tarr,tcf,color='r')
        #plt.plot(tcf_tarr,np.cos(tcf_tarr),color='g')
        f.close()
        print(i,'completed')
        print('time',time.time() - start_time)


def compute_tcf_CMD(n_instance,N,dpotential,beta,T,deltat):
    
    for i in range(n_instance):
        tcf =  Ring_polymer_dynamics.RPMD_instance((i+1)*100,N,1,dpotential,beta,T,deltat)
        print('beta',MD_System.beta,'n_beads',MD_System.n_beads,'T',T)
        f = open('/home/vgs23/Pickle_files/CMD_tcf_N_{}_B_{}_inst_{}_T_{}_dt_{}_SAMP3.dat'.format(N*100,beta,i,T,deltat),'wb')
        pickle.dump(tcf,f)
        #plt.plot(tcf_tarr,tcf,color='g')
        #plt.plot(tcf_tarr,np.cos(tcf_tarr),color='g')
        f.close()
        print(i,'completed')
        print('time',time.time() - start_time)
