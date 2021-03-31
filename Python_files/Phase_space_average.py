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
from matplotlib.ticker import ScalarFormatter
import matplotlib

rc('text', usetex=True)
rc('font', family='serif')


def phase_space_average(CMD,Matsubara,M,swarmobj,derpotential,tim,deltat,rng_ind):
    rng = np.random.RandomState(rng_ind)
    tarr = np.arange(0,tim+0.0001,tim/100.0)
    
    #taux = np.arange(0,2*tim+0.001,tim/100.0)
    
    thermalize(CMD,Matsubara,M,swarmobj,derpotential,deltat,200,rng)
    print('Time 1', time.time()-start_time)
    n_approx = 10
    Quan_arr = np.zeros_like(tarr)
    print('thermalized', swarmobj.sys_kin()/(N*n_beads), swarmobj.m, swarmobj.sm, np.sum(swarmobj.p**2))
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


if(0):
    N=100
    dim = 2
    n_beads = 36
    beta = 1.0/5
    deltat=1e-2
    T = 15.0

    tarr = np.arange(0,T+0.0001,T/100.0)

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
        f =  open("/home/vgs23/Pickle_files/RPMD_OTOC_{}_beta_{}_n_bead_{}_tfinal_{}.dat".format('Coupled_harmonic',beta,n_beads,T),'wb')
        pickle.dump(tarr,f)
        pickle.dump(Marr,f)
        f.close()

        print('Time taken:',time.time()-start_time)
        
    #set_sim(N,dim,n_beads,beta,deltat,'g')
    beadarr = [1,12,24,36]#,20,30]
    colorarr = ['r','g','b','m']

    if(0):
        for (n_bead,col) in zip(beadarr,colorarr):
            set_sim(N,dim,n_bead,beta,deltat,col)
        plt.show()



    fig = plt.figure(1)
    ax = fig.add_subplot(1,1,1)
    matplotlib.rcParams.update({'font.size': 50})
    #plt.suptitle('Thermal averaged stability matrix elements: Time evolution')
    #plt.title(r'$<M_{qq}> \;\; vs \;\; t$')
    plt.xlabel(r'$t$',fontsize=15)
    plt.ylabel(r'$C^{RP}_N(t)$',fontsize=15)
    plt.gcf().subplots_adjust(bottom=0.12,left=0.12)
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_formatter(ScalarFormatter())
    #ax.legend(loc=1)
    for (bead,col) in zip(beadarr,colorarr):
        f =  open("/home/vgs23/Pickle_files/RPMD_OTOC_{}_beta_{}_n_bead_{}_tfinal_{}.dat".format('Coupled_harmonic',beta,bead,T),'rb')
        tarr = pickle.load(f)
        Marr = pickle.load(f)
        f.close()

        print('Time taken:',time.time()-start_time)
        
        plt.plot(tarr,(abs(Marr)),color=col,label=r'$N_b={}$'.format(bead))

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
              fancybox=False, shadow=False, ncol=4,prop={'size': 10})

    f =  open("/home/vgs23/Pickle_files/Kubo_OTOC_{}_beta_{}_basis_{}_n_eigen_{}_tfinal_{}.dat".format('Coupled_harmonic',beta,250,150,T),'rb')
    tarr = pickle.load(f)
    OTOC_arr = pickle.load(f)
    f.close()
    matplotlib.rcParams.update({'font.size': 13})
    ax.annotate('Quantum OTOC',xy=(2.5,1.2),xytext=(0.64,0.42),textcoords = 'axes fraction')
    plt.plot(tarr,(OTOC_arr))
    #plt.legend()
    plt.yscale('log')

    plt.savefig('/home/vgs23/Images/RPMD_OTOC.eps', format='eps',dpi=96)
    plt.show()
