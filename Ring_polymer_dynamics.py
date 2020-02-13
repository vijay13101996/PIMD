#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 14:43:35 2019

@author: vgs23
"""
import numpy as np
from Correlation_function import corr_function_upgrade
import MD_System
import multiprocessing as mp
from functools import partial
import Velocity_verlet
import importlib
import time
import pickle
start_time = time.time()


def RPMD_instance(rng,N,beads,dpotential,beta,T,deltat):
    swarmobject = MD_System.swarm(N)
    MD_System.system_definition(beta,beads,2,MD_System.dpotential)
    importlib.reload(Velocity_verlet)
    
    swarmobject.q = np.zeros((swarmobject.N,MD_System.dimension,MD_System.n_beads)) +1.0
    rand_boltz = np.random.RandomState(rng)
    swarmobject.p = rand_boltz.normal(0.0,swarmobject.m/(MD_System.beta),np.shape(swarmobject.q))
    
    pool = mp.Pool(mp.cpu_count())
    Matsubara = 0
    CMD=0
    
    n_inst = 10
    func = partial(corr_function_upgrade,CMD,Matsubara,beads,swarmobject,dpotential,MD_System.mom_op,MD_System.mom_op,T,deltat)
    results = pool.map(func, range(n_inst))
    
    pool.close()
    pool.join()
    return np.sum(results,0)/n_inst

def compute_tcf(n_RPMD_instance,N,beads,dpotential,beta,T,deltat):
    
    for i in range(n_RPMD_instance):
        tcf =  RPMD_instance((i+1)*100,N,beads,dpotential,beta,T,deltat)
        f = open('RPMD_tcf_N_{}_B_{}_inst_{}_dt_{}_NB_{}.dat'.format(N*100,MD_System.beta*MD_System.n_beads,i,deltat,MD_System.n_beads),'wb')
        pickle.dump(tcf,f)
        #plt.plot(tcf_tarr,tcf,color='r')
        #plt.plot(tcf_tarr,np.cos(tcf_tarr),color='g')
        f.close()
        print(i,'completed')
        print('time',time.time() - start_time)
        
    #return tcf