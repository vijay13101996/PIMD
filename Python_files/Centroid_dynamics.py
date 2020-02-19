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
start_time =time.time()

def CMD_instance(beads,dpotential,beta): 
    
    MD_System.system_definition(beta,beads,1,dpotential)
    importlib.reload(Velocity_verlet)
    Centroid_mean_force.mean_force_calculation(dpotential)
     
    f = open("/home/vgs23/Pickle_files/CMD_Force_Morse_B_{}_NB_{}_100_10.dat".format(MD_System.beta,MD_System.n_beads),'rb')       
    data = np.loadtxt(f)
    Q = data[:,0]
    CMD_force = data[:,1]
    
    xforce = scipy.interpolate.interp1d(Q,CMD_force,kind='cubic')
    return xforce

def ACMD_instance(rng,N,beads,dpotential,beta,T,deltat):
    
    swarmobject = MD_System.swarm(N)
    MD_System.system_definition(beta,beads,1,MD_System.dpotential)
    #gamma = 1.0
    #w_arr = beads**0.5/(beta*gamma)*np.ones(beads)
    #MD_System.w_arr = MD_System.rearrange(w_arr)
    #MD_System.w_arr[0] = 0.0
    
    importlib.reload(Velocity_verlet)
    
    swarmobject.q = np.zeros((swarmobject.N,MD_System.dimension,MD_System.n_beads))
    rand_boltz = np.random.RandomState(rng)
    swarmobject.p = rand_boltz.normal(0.0,swarmobject.m/(MD_System.beta),np.shape(swarmobject.q))
    
    pool = mp.Pool(mp.cpu_count())
    Matsubara = 0
    CMD=0
    
    
    n_inst = 10
    func = partial(corr_function_upgrade,CMD,Matsubara,beads,swarmobject,dpotential,MD_System.pos_op,MD_System.pos_op,T,deltat)
    results = pool.map(func, range(n_inst))
    
    pool.close()
    pool.join()
    return np.sum(results,0)/n_inst

def compute_tcf(n_ACMD_instance,N,beads,dpotential,beta,T,deltat):
    
    for i in range(n_ACMD_instance):
        tcf =  ACMD_instance((i+1)*100,N,beads,dpotential,beta,T,deltat)
        f = open('/home/vgs23/Pickle_files/ACMD_tcf_N_{}_B_{}_inst_{}_dt_{}_NB_{}.dat'.format(N*100,MD_System.beta*MD_System.n_beads,i,deltat,MD_System.n_beads),'wb')
        pickle.dump(tcf,f)
        #plt.plot(tcf_tarr,tcf,color='r')
        #plt.plot(tcf_tarr,np.cos(tcf_tarr),color='g')
        f.close()
        print(i,'completed')
        print('time',time.time() - start_time)