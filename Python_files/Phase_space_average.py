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
import Correlation_function 
import matplotlib.pyplot as plt
from matplotlib import rc

rc('text', usetex=True)
rc('font', family='serif')


def phase_space_average(CMD,Matsubara,M,swarmobj,derpotential,tim,deltat,rng_ind):
    rng = np.random.RandomState(rng_ind)
    tarr = np.arange(0,tim+0.0001,tim/100.0)
    
    #taux = np.arange(0,2*tim+0.001,tim/100.0)
    
    thermalize(CMD,Matsubara,M,swarmobj,derpotential,deltat,200,rng)
    print('Time 1', time.time()-start_time)
    n_approx = 1
    Quan_arr = np.zeros_like(tarr)
    print('thermalized', swarmobj.sys_kin()/(N*n_beads), swarmobj.m, swarmobj.sm, np.sum(swarmobj.p**2))
    #plt.plot(np.mean(swarmobj.q,axis=2)[:,0],np.mean(swarmobj.q,axis=2)[:,1])
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
        

    for j in range(n_approx):
        temp_q = swarmobj.q.transpose(0,2,1)
        temp_p = swarmobj.p.transpose(0,2,1)
        print('temp_q',np.shape(temp_q))
        traj_arr = Monodromy.ode_instance_RP(temp_q,temp_p,tarr) ##
        if(0): 
            q_arr = np.mean(Monodromy.q_RP(traj_arr),axis=2)
            A_arr = q_arr.copy()
            A_arr[len(tarr):,...] = 0.0
            B_arr = q_arr.copy()
            print('A_arr', np.shape(A_arr)) 
            tcf_cr = Correlation_function.convolution(A_arr,B_arr,0)
            tcf_cr = np.sum(tcf_cr,axis=2) #Summing up over dimensions
            tcf_cr = np.sum(tcf_cr,axis=1) #Summing up over the number of particles
            print('tcf',np.shape(tcf_cr))
        print('Time 2', time.time()-start_time)
        Quan_arr+= np.sum(Monodromy.detmqq_RP(traj_arr),axis=1)#tcf_cr[:len(tarr)]#
        thermalize(CMD,Matsubara,M,swarmobj,derpotential,deltat,100,rng)
    Quan_arr/=(n_approx*swarmobj.N) 
    #Quan_arr/=(n_approx*swarmobj.N*MD_System.dimension*len(tarr))
    return Quan_arr

N=100
dim = 2
n_beads = 4
beta = 1.0/10
deltat=1e-2
T = 15.0

tarr = np.arange(0,T+0.0001,T/100.0)
fig = plt.figure(1)
plt.suptitle('Thermal averaged stability matrix elements: Time evolution')
plt.title(r'$<M_{qq}> \;\; vs \;\; t$')
plt.xlabel(r'$t$')
plt.ylabel(r'$<M_{qq}>$')
def set_sim(N,dim,n_beads,beta,deltat,col):
    swarmobject = MD_System.swarm(N)
    MD_System.system_definition(beta,n_beads,dim,MD_System.dpotential)   #78943.7562025: Liquid Helium temperature
    swarmobject.m = 0.5
    swarmobject.sm = 0.5**0.5
    MD_System.gamma = 10.0
    imp.reload(Velocity_verlet)
    Monodromy.system_definition_RP(N,dim,n_beads,n_beads/beta)


    print('Beta',MD_System.beta)
    rand_boltz = np.random.RandomState(1)
    swarmobject.q = rand_boltz.rand(swarmobject.N,MD_System.dimension,MD_System.n_beads)
    swarmobject.p = rand_boltz.normal(0.0,swarmobject.m/(MD_System.beta),np.shape(swarmobject.q))

    Matsubara = 0
    CMD=0
    M=0

    Marr =phase_space_average(CMD,Matsubara,M,swarmobject,Monodromy.dpotential_cq_vv,T,deltat,1)
   
    #cProfile.run('phase_space_average(CMD,Matsubara,M,swarmobject,Monodromy.force_dpotential,T,deltat,1)')
    print('Time taken:',time.time()-start_time)
    #fig = plt.figure(1)
    #plt.plot(tarr[:50], Marr[:50])
    plt.plot(tarr,np.log(abs(Marr)),color=col,label=r'$N_b={}$'.format(n_beads))
    #plt.plot(tarr,0.5*tarr)
    plt.legend()
    #plt.plot(tarr,abs(Marr)**2)
    #plt.plot(tarr,Marr)


#set_sim(N,dim,n_beads,beta,deltat,'g')
beadarr = [n_beads]#,4,10]#,20,30]
colorarr = ['r','g','b','m']
for (n_bead,col) in zip(beadarr,colorarr):
    set_sim(N,dim,n_bead,beta,deltat,col)
plt.show()



