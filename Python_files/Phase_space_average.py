#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 16:38:22 2019

@author: vgs23
"""

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy,copy
from numpy import linalg as LA
from numpy.fft import fft,ifft
from multiprocessing import Process
from Correlation_function import corr_function,corr_function_upgrade,corr_function_matsubara
#from MD_System import swarm,pos_op,beta,dimension,n_beads,w_n
import MD_System
import multiprocessing as mp
from functools import partial
import scipy
import Velocity_verlet
import importlib
import time
import pickle
start_time = time.time()
from MD_Evolution import time_evolve
from Langevin_thermostat import thermalize
import Monodromy
import cProfile
import imp

def phase_space_average(CMD,Matsubara,M,swarmobj,derpotential,tim,deltat,rng_ind):
    rng = np.random.RandomState(rng_ind)
    tarr = np.arange(0,tim+0.0001,tim/100.0)

    thermalize(CMD,Matsubara,M,swarmobj,derpotential,deltat,100,rng)
    n_approx = 1
    Quan_arr = np.zeros_like(tarr)
    print('thermalized')
    #plt.scatter(swarmobj.q[:,0],swarmobj.q[:,1])
    #plt.show()
    if(0):
        for j in range(n_approx):
            traj_data = Monodromy.Integration_instance(swarmobj.q,swarmobj.p)
            pool = mp.Pool(mp.cpu_count()-2)
            for i in range(1,len(tarr)):
                time_evolve(CMD,Matsubara,M,swarmobj, derpotential, deltat, tarr[i]-tarr[i-1])
                traj_data = Monodromy.vector_integrate(pool,tarr[i]-tarr[i-1],traj_data)
                Quan_arr[i]+= np.sum(Monodromy.vector_mqq(pool,traj_data))
            pool.close()
            pool.join()
            #thermalize(CMD,Matsubara,M,swarmobj,derpotential,deltat,4,rng)
            print(time.time()-start_time)
        Quan_arr/=(n_approx*swarmobj.N)

    for j in range(n_approx):
        traj_arr = Monodromy.ode_instance(swarmobj.q,swarmobj.p,tarr)
        Quan_arr+= np.sum(Monodromy.detmqq(traj_arr),axis=1)
        thermalize(CMD,Matsubara,M,swarmobj,derpotential,deltat,10,rng)
    
    return Quan_arr

N=20
dim = 3
deltat=1e-2
swarmobject = MD_System.swarm(N)
MD_System.system_definition(78943.7562025,1,dim,MD_System.dpotential)
imp.reload(Velocity_verlet)
Monodromy.system_definition(N,dim)


rand_boltz = np.random.RandomState(1)
swarmobject.q = rand_boltz.rand(swarmobject.N,MD_System.dimension,MD_System.n_beads)
swarmobject.p = rand_boltz.normal(0.0,swarmobject.m/(MD_System.beta),np.shape(swarmobject.q))

Matsubara = 0
CMD=0
M=0
T =10.0    
Marr =phase_space_average(CMD,Matsubara,M,swarmobject,Monodromy.dpotential_He_aux,T,deltat,1)
#cProfile.run('phase_space_average(CMD,Matsubara,M,swarmobject,Monodromy.force_dpotential,T,deltat,1)')
tarr = np.arange(0,T+0.0001,T/100.0)
print('Time taken:',time.time()-start_time)
fig = plt.figure(1)
plt.plot(tarr,2*np.log(abs(Marr)))
plt.show()
