#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 14:37:29 2019

@author: vgs23
"""
import numpy as np
from Correlation_function import corr_function_matsubara
#from MD_System import swarm,pos_op,beta,dimension,n_beads,w_n
import MD_System
import multiprocessing as mp
from functools import partial
import Velocity_verlet
import importlib



def Matsubara_instance(rng,N,M,dpotential,T,deltat):
    swarmobject = MD_System.swarm(N)
    MD_System.system_definition(8,M,2,MD_System.dpotential)
    importlib.reload(Velocity_verlet)

    swarmobject.q = np.zeros((swarmobject.N,MD_System.dimension,MD_System.n_beads))
    
    rand_boltz = np.random.RandomState(rng)
    swarmobject.p = rand_boltz.normal(0.0,swarmobject.m/MD_System.beta,np.shape(swarmobject.q))
    
    pool = mp.Pool(mp.cpu_count())
    Matsubara = 1
    CMD=0
    
    
    n_inst = 10
    func = partial(corr_function_matsubara,CMD,Matsubara,M,swarmobject,dpotential,MD_System.pos_op,MD_System.pos_op,T,deltat)
    results = pool.map(func, range(n_inst))
    pool.close()
    pool.join()
    return np.sum(results,0)/n_inst
