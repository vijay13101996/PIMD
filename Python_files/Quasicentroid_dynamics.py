#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 12:05:13 2020

@author: vgs23
"""

import numpy as np
from Correlation_function import corr_function_QCMD
import MD_System
import multiprocessing as mp
from functools import partial
import Velocity_verlet
import importlib
import time
import pickle
start_time = time.time()


def QCMD_instance(rng,N,beads,dpotential,beta,T,deltat):
    swarmobject = MD_System.swarm(N)
    
    MD_System.system_definition(beta,beads,2,MD_System.dpotential)
    importlib.reload(Velocity_verlet)
    
    rand_boltz = np.random.RandomState(2)
    print('beta',MD_System.beta,MD_System.beta_n)
    swarmobject.q = np.zeros((swarmobject.N,MD_System.dimension,MD_System.n_beads)) + 1.0 #np.random.rand(swarmobject.N,MD_System.dimension,MD_System.n_beads) 
    swarmobject.p = np.zeros_like(swarmobject.q)#rand_boltz.normal(0.0,swarmobject.m/(MD_System.beta_n),np.shape(swarmobject.q))
    
    # In the line above, 1.0 is should be replaced with beta, when required
    QC_q = np.zeros((swarmobject.N,MD_System.dimension)) + 1.0 
    QC_p = np.zeros_like(QC_q)#rand_boltz.normal(0.0,swarmobject.m/(MD_System.beta),np.shape(QC_q))#np.zeros((swarmobject.N,MD_System.dimension))
    lambda_curr = np.zeros_like(swarmobject.q)
    
    pool = mp.Pool(mp.cpu_count())
    n_inst = 1 
    func = partial(corr_function_QCMD,swarmobject,QC_q,QC_p,lambda_curr,MD_System.mom_op,MD_System.mom_op,T,deltat,dpotential)
    results = pool.map(func, range(n_inst))
    #print('results',np.sum(results,0)/n_inst)
    pool.close()
    pool.join()
    return np.sum(results,0)/n_inst

def compute_tcf(n_QCMD_instance,N,beads,dpotential,beta,T,deltat):
    print('beta',beta)
    for i in range(n_QCMD_instance):
        tcf =  QCMD_instance((i+1)*100,N,beads,dpotential,beta,T,deltat)
        f = open('QCMD_vvtcf_N_{}_B_{}_inst_{}_dt_{}_NB_{}.dat'.format(N*100,MD_System.beta*MD_System.n_beads,i,deltat,MD_System.n_beads),'wb')
        pickle.dump(tcf,f)
        #plt.plot(tcf_tarr,tcf,color='r')
        #plt.plot(tcf_tarr,np.cos(tcf_tarr),color='g')
        f.close()
        print(i,'completed')
        print('time',time.time() - start_time)
        
    return tcf
        
