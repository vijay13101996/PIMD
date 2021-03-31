#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 14:43:35 2019

@author: vgs23
"""
import numpy as np
from Correlation_function import corr_function_upgrade, corr_function_Andersen, static_average
import MD_System
import multiprocessing as mp
from functools import partial
import Velocity_verlet
import importlib
import time
import pickle
start_time = time.time()


def RPMD_instance(rng,N,beads,dpotential,beta,T,n_tp,deltat):
    swarmobject = MD_System.swarm(N,beads,beta,1)
    print(swarmobject.dimension, 'dim')
    
    swarmobject.q = np.zeros((swarmobject.N,swarmobject.dimension,swarmobject.n_beads))+1.0 
    print('RP', swarmobject.n_beads,swarmobject.beta_n)
    rand_boltz = np.random.RandomState(rng)
    swarmobject.p = np.zeros_like(swarmobject.q)#rand_boltz.normal(0.0,swarmobject.m/(swarmobject.beta),np.shape(swarmobject.q))
    
    pool = mp.Pool(mp.cpu_count())
    Matsubara = 0
    CMD=0
    ACMD = 0

    n_inst = 10
    func = partial(corr_function_upgrade,ACMD,CMD,Matsubara,beads,swarmobject,dpotential,swarmobject.pos_op,swarmobject.pos_op,T,n_tp,deltat)
    results =  pool.map(func, range(n_inst)) #corr_function_upgrade(ACMD,CMD,Matsubara,beads,swarmobject,dpotential,swarmobject.pos_op,swarmobject.pos_op,T,n_tp,deltat,0) #     
    
    pool.close()
    pool.join()
    return np.sum(results,0)/n_inst

def RPMD_Andersen_instance(rng,N,beads,dpotential,beta,T,n_tp,deltat):
    swarmobject = MD_System.swarm(N,beads,beta,1)
    importlib.reload(Velocity_verlet)
    
    swarmobject.q = np.zeros((swarmobject.N,swarmobject.dimension,swarmobject.n_beads)) 
    print('RP', swarmobject.n_beads,swarmobject.beta_n)
    rand_boltz = np.random.RandomState(rng)
    swarmobject.p = rand_boltz.normal(0.0,swarmobject.m/(swarmobject.beta),np.shape(swarmobject.q))
    
    pool = mp.Pool(mp.cpu_count())
    Matsubara = 0
    CMD=0
    ACMD = 0

    n_inst = 10
    func = partial(corr_function_Andersen,ACMD,CMD,Matsubara,beads,swarmobject,dpotential,swarmobject.pos_op,swarmobject.pos_op,T,n_tp,deltat)
    results =  pool.map(func, range(n_inst)) #corr_function_upgrade(ACMD,CMD,Matsubara,beads,swarmobject,dpotential,swarmobject.pos_op,swarmobject.pos_op,T,n_tp,deltat,0) #     
    
    pool.close()
    pool.join()
    return np.sum(results,0)/n_inst


def compute_tcf(n_RPMD_instance,N,beads,dpotential,beta,T,n_tp,deltat):
    
    for i in range(n_RPMD_instance):
        tcf =  RPMD_instance((i+1)*100,N,beads,dpotential,beta,T,n_tp,deltat)
        f = open('/home/vgs23/Pickle_files/RPMD_Andersen_tcf_N_{}_B_{}_inst_{}_dt_{}_NB_{}_S1.dat'.format(N*100,beta,i,deltat,beads),'wb')
        pickle.dump(tcf,f)
        #plt.plot(tcf_tarr,tcf,color='r')
        #plt.plot(tcf_tarr,np.cos(tcf_tarr),color='g')
        f.close()
        print(i,'completed')
        print('time',time.time() - start_time)
        
    #return tcf

def compute_static_avg(N,beads,dpotential,beta,deltat):
    swarmobject = MD_System.swarm(N,beads,beta,1)
    importlib.reload(Velocity_verlet)
    
    swarmobject.q = np.zeros((swarmobject.N,swarmobject.dimension,swarmobject.n_beads)) 
    print('RP', swarmobject.n_beads,swarmobject.beta_n)
    rand_boltz = np.random.RandomState(0)
    swarmobject.p = rand_boltz.normal(0.0,swarmobject.m/(swarmobject.beta),np.shape(swarmobject.q))
    
    ACMD = 0
    CMD = 0 
    Matsubara = 0
    M=0
    static_average(ACMD,CMD,Matsubara,M,swarmobject,dpotential,swarmobject.pos_op,swarmobject.pos_op,deltat,0)
