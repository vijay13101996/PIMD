#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 14:37:29 2019

@author: vgs23
"""
import numpy as np
from Correlation_function import compute_phase_histogram,corr_function_matsubara, corr_function_phaseless_Matsubara, OTOC_phase_dep_Matsubara, corr_function_phase_dep_Matsubara
import MD_System
import multiprocessing as mp
from functools import partial
import importlib
import pickle
import time

def Matsubara_instance(rng,N,M,dpotential,beta,T,n_tp,deltat):
    """ 
    This function computes the TCF from Matsubara dynamics.
    The time propagation is done along the Matsubara coordinates.
    """
    
    swarmobject = MD_System.swarm(N,M,beta,1)

    swarmobject.q = np.zeros((swarmobject.N,swarmobject.dimension,swarmobject.n_beads)) 
    rand_boltz = np.random.RandomState(rng)
    swarmobject.p = rand_boltz.normal(0.0,swarmobject.m/swarmobject.beta,np.shape(swarmobject.q))
    
    pool = mp.Pool(mp.cpu_count())
    Matsubara = 1
    CMD=0
    ACMD=0
     
    n_inst = 1
    func = partial(corr_function_matsubara,ACMD,CMD,Matsubara,M,swarmobject,dpotential,swarmobject.pos_op,swarmobject.pos_op,T,n_tp,deltat)
    results = pool.map(func, range(n_inst))
    pool.close()
    pool.join()
    return np.sum(results,0)/n_inst

def Phaseless_Matsubara_instance(rng,N,M,dpotential,beta,T,n_tp,ntheta, deltat):
    swarmobject = MD_System.swarm(N,M,beta,1)

    swarmobject.q = np.zeros((swarmobject.N,swarmobject.dimension,swarmobject.n_beads)) 
    rand_boltz = np.random.RandomState(rng)
    swarmobject.p = rand_boltz.normal(0.0,swarmobject.m/swarmobject.beta,np.shape(swarmobject.q))
    

    extent = np.linspace(-10.0,10.0,ntheta)
    theta_arr = extent.copy()#np.sort(rand_boltz.normal(0.0,1.0,len(extent)))
    print('theta',theta_arr, theta_arr.dtype)
    pool = mp.Pool(mp.cpu_count()-2)
     
    n_inst = 1
    func = partial(corr_function_phaseless_Matsubara,swarmobject,dpotential,swarmobject.pos_op,swarmobject.pos_op,T,n_tp,theta_arr,deltat)
    results = pool.map(func, range(n_inst))
    pool.close()
    pool.join()
    return np.sum(results,0)/n_inst

def Phase_dep_tcf_instance(rng,N,M,dpotential,beta,T,n_tp, deltat, theta_arr):
    swarmobject = MD_System.swarm(N,M,beta,1)

    swarmobject.q = np.zeros((swarmobject.N,swarmobject.dimension,swarmobject.n_beads)) 
    rand_boltz = np.random.RandomState(rng)
    swarmobject.p = rand_boltz.normal(0.0,swarmobject.m/swarmobject.beta,np.shape(swarmobject.q))
    
    print('rng', rng, swarmobject.p[0])
    print('theta',theta_arr[-1], theta_arr.dtype)

    if(1):
        pool = mp.Pool(6)
        n_inst = 6
        func = partial(corr_function_phase_dep_Matsubara,swarmobject,dpotential,swarmobject.pos_op,swarmobject.pos_op,T,n_tp,theta_arr,deltat)
        results = pool.map(func, range(n_inst))
        pool.close()
        pool.join()
        return np.sum(results,0)/n_inst

    if(0):
        tcf_thetat = corr_function_phase_dep_Matsubara(swarmobject,dpotential,swarmobject.pos_op,swarmobject.pos_op,T,n_tp,theta_arr,deltat,rng)
        return tcf_thetat
        
def Phase_dep_OTOC_instance(rng,N,M,dpotential,ddpotential,beta,T,n_tp, deltat, theta_arr):
    swarmobject = MD_System.swarm(N,M,beta,1)

    swarmobject.q = np.zeros((swarmobject.N,swarmobject.dimension,swarmobject.n_beads)) 
    rand_boltz = np.random.RandomState(rng)
    swarmobject.p = rand_boltz.normal(0.0,swarmobject.m/swarmobject.beta,np.shape(swarmobject.q))
    
    #print('rng', rng, swarmobject.p[0])
    #print('theta',theta_arr, theta_arr.dtype)
    
    if(0):
        B_hist,C = compute_phase_histogram(10,swarmobject,dpotential,beta,deltat,theta_arr,rand_boltz)
        f = open('/home/vgs23/Pickle_files/Matsubara_histogram_B_{}_inst_{}_M_{}_deltat_{}_ntheta_{}.dat'.format(N,beta,rng,M,deltat,len(theta_arr)),'wb')
        pickle.dump(B_hist,f)
        pickle.dump(C,f)
        f.close()
    
    if(1):
        pool = mp.Pool(5)
        n_inst = 5
        func = partial(OTOC_phase_dep_Matsubara,swarmobject,dpotential,ddpotential,T,n_tp,theta_arr,deltat)
        results = pool.map(func, range(n_inst))
        pool.close()
        pool.join()
        return np.sum(results,0)/n_inst

    if(0):
        tcf_thetat = OTOC_phase_dep_Matsubara(swarmobject,dpotential,ddpotential,T,n_tp,theta_arr,deltat,rng)
        return tcf_thetat
        
def compute_phase_dep_OTOC(Matsubara_instance,N,M,dpotential,ddpotential,beta,T,n_tp,deltat, theta_arr):
    start_time = time.time()    
    for i in Matsubara_instance:
        ntheta= 1
        tcf = Phase_dep_OTOC_instance(i,N,M,dpotential,ddpotential,beta,T,n_tp, deltat, theta_arr)
        f = open('/home/vgs23/Pickle_files/Matsubara_phase_dep_OTOC_N_{}_B_{}_inst_{}_dt_{}_M_{}_ntheta_{}.dat'.format(N,beta,i,deltat,M,len(theta_arr)),'wb')
        pickle.dump(tcf,f)
        f.close()
        print(i,'completed')
        print('time',time.time() - start_time)
        
    return tcf

def compute_phase_dep_tcf(Matsubara_instance,N,M,dpotential,beta,T,n_tp,deltat, theta_arr):
    start_time = time.time()    
    for i in Matsubara_instance:
        tcf = Phase_dep_tcf_instance(i,N,M,dpotential,beta,T,n_tp, deltat, theta_arr)
        f = open('/home/vgs23/Pickle_files/Matsubara_phase_dep_tcf_N_{}_B_{}_inst_{}_dt_{}_M_{}_ntheta_{}.dat'.format(N,beta,i,deltat,M,len(theta_arr)),'wb')
        pickle.dump(tcf,f)
        f.close()
        print(i,'completed')
        print('time',time.time() - start_time)
        
    return tcf

def compute_tcf(n_Matsubara_instance,N,M,dpotential,beta,T,n_tp,deltat):
    start_time = time.time()    
    for i in range(n_Matsubara_instance):
        tcf = Matsubara_instance((i+1)*100,N,M,dpotential,beta,T,n_tp,deltat)
        f = open('/home/vgs23/Pickle_files/Matsubara_tcf_N_{}_B_{}_inst_{}_dt_{}_M_{}_S1.dat'.format(N*100,beta,i,deltat,M),'wb')
        pickle.dump(tcf,f)
        f.close()
        print(i,'completed')
        print('time',time.time() - start_time)
        
    return tcf

def compute_phaseless_tcf(n_Matsubara_instance,N,M,dpotential,beta,T,n_tp,deltat):
    start_time = time.time()    
    for i in range(n_Matsubara_instance):
        ntheta= 51
        tcf = Phaseless_Matsubara_instance((i+1)*100,N,M,dpotential,beta,T,n_tp,ntheta, deltat)
        f = open('/home/vgs23/Pickle_files/Matsubara_phaseless_tcf_N_{}_B_{}_inst_{}_dt_{}_M_{}_ntheta_{}.dat'.format(N*100,beta,i,deltat,M,ntheta),'wb')
        pickle.dump(tcf,f)
        f.close()
        print(i,'completed')
        print('time',time.time() - start_time)
        
    return tcf


