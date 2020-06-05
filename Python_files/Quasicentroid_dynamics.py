#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 12:05:13 2020

@author: vgs23
"""

import numpy as np
import scipy
from scipy import interpolate
from Correlation_function import corr_function_QCMD
import MD_System
import multiprocessing as mp
from functools import partial
import Velocity_verlet
import importlib
import time
import pickle
import Langevin_thermostat
from matplotlib import pyplot as plt
import Ring_polymer_dynamics
start_time = time.time()

def QC_MF_calculation(dpotential,N,n_sample,NQ,deltat):
    print(MD_System.n_beads,MD_System.w_arr)
    Q = np.linspace(0.5,3.0,NQ)
    QC_MF = np.zeros((len(Q),MD_System.dimension))
    swarmobject = MD_System.swarm(N)
    rand_boltz = np.random.RandomState(1)
    swarmobject.p = rand_boltz.normal(0.0,swarmobject.m/MD_System.beta,(swarmobject.N,MD_System.dimension,MD_System.n_beads))
    rng=np.random.RandomState(1)
    
    for j in range(len(Q)):
        swarmobject.q = np.zeros((swarmobject.N,MD_System.dimension,MD_System.n_beads)) + 1.0 
        print('j and time',j, time.time()-start_time)   
        QC_q = np.zeros((swarmobject.N,MD_System.dimension)) 
        QC_q[:,0] = Q[j] 
        QC_p = np.zeros_like(QC_q)
        lambda_curr = np.zeros_like(swarmobject.q)
        
        Langevin_thermostat.qcmd_thermalize(1,swarmobject,QC_q,QC_p,lambda_curr,dpotential,deltat,10000,rng)
        
        for i in range(n_sample):
            QC_MF[j] += np.mean(Velocity_verlet.dpotential_qcentroid(QC_q,swarmobject,dpotential),axis=0)
            Langevin_thermostat.qcmd_thermalize(1,swarmobject,QC_q,QC_p,lambda_curr,dpotential,deltat,5000,rng)
            print('time',time.time()-start_time)
    QC_MF/=n_sample
    ### BE CAREFUL WITH THE SIGN OF THE POTENTIAL HERE. You have used
    #   force and gradient of potential interchangeably
    f = open("/home/vgs23/Pickle_files/QCMD_Force_B_{}_NB_{}_N_{}_nsample_{}_deltat_{}_NQ_{}.dat".format(MD_System.beta,MD_System.n_beads,N,n_sample,deltat,len(Q)),'w+')
    QCMD_force = QC_MF[:,0]
    for x in zip(Q,QCMD_force):
        f.write("{}\t{}\n".format(x[0],x[1]))
    f.close()
   
    if(0):
        plt.plot(Q,QC_MF[:,0],color='r')
        plt.plot(Q,QC_MF[:,1],color='b')
        plt.show()
    

def QCMD_instance(beads,dpotential,beta):
    MD_System.system_definition(beta,beads,2,dpotential)
    importlib.reload(Velocity_verlet)
    N=100
    n_sample=1
    NQ = 50
    deltat = 1.0
    #QC_MF_calculation(dpotential,N,n_sample,NQ,deltat)
    
    f = open("/home/vgs23/Pickle_files/QCMD_Force_B_{}_NB_{}_N_{}_nsample_{}_deltat_{}_NQ_{}.dat".format(MD_System.beta,MD_System.n_beads,N,n_sample,deltat,NQ),'rb')       
    data = np.loadtxt(f)
    Q = data[:,0]
    QCMD_force = data[:,1]
    
    rforce = scipy.interpolate.interp1d(Q,QCMD_force,kind='cubic')
    if(0):
        plt.plot(Q,QCMD_force)
        plt.plot(Q,rforce(Q))
        plt.show()
    return rforce
    
    
def AQCMD_instance(rng,N,beads,dpotential,beta,T,deltat):
    swarmobject = MD_System.swarm(N)
    
    MD_System.system_definition(beta,beads,2,MD_System.dpotential)
    importlib.reload(Velocity_verlet)
    
    rand_boltz = np.random.RandomState(2)
    print('beta',MD_System.beta,MD_System.beta_n)
    swarmobject.q = np.zeros((swarmobject.N,MD_System.dimension,MD_System.n_beads)) + 1.0 #np.random.rand(swarmobject.N,MD_System.dimension,MD_System.n_beads) 
    swarmobject.p = rand_boltz.normal(0.0,swarmobject.m/(MD_System.beta_n),np.shape(swarmobject.q))
    
    # In the line above, 1.0 is should be replaced with beta, when required
    # Update: Done!!!
    QC_q = np.zeros((swarmobject.N,MD_System.dimension)) + 1.0
    QC_p = rand_boltz.normal(0.0,swarmobject.m/(MD_System.beta),np.shape(QC_q))#np.zeros((swarmobject.N,MD_System.dimension))
    lambda_curr = np.zeros_like(swarmobject.q)
    
    pool = mp.Pool(mp.cpu_count())
    n_inst = 1 
    func = partial(corr_function_QCMD,swarmobject,QC_q,QC_p,lambda_curr,MD_System.pos_op,MD_System.pos_op,T,deltat,dpotential)
    results = pool.map(func, range(n_inst))
    #print('results',np.sum(results,0)/n_inst)
    pool.close()
    pool.join()
    return np.sum(results,0)/n_inst

def compute_tcf_AQCMD(n_QCMD_instance,N,beads,dpotential,beta,T,deltat):
    print('beta',beta)
    for i in range(n_QCMD_instance):
        tcf =  AQCMD_instance((i+1)*100,N,beads,dpotential,beta,T,deltat)
        f = open('/home/vgs23/Pickle_files/AQCMD_tcf_N_{}_B_{}_inst_{}_T_{}_dt_{}_NB_{}.dat'.format(N*100,MD_System.beta*MD_System.n_beads,i,T,deltat,MD_System.n_beads),'wb')
        pickle.dump(tcf,f)
        #plt.plot(tcf_tarr,tcf,color='r')
        #plt.plot(tcf_tarr,np.cos(tcf_tarr),color='g')
        f.close()
        print(i,'completed')
        print('time',time.time() - start_time)
        
    return tcf
        
def compute_tcf_QCMD(n_instance,N,dpotential,beta,T,deltat):
    
    for i in range(n_instance):
        tcf =  Ring_polymer_dynamics.RPMD_instance((i+1)*100,N,1,dpotential,beta,T,deltat)
        print('beta',MD_System.beta,'n_beads',MD_System.n_beads,'T',T)
        f = open('/home/vgs23/Pickle_files/QCMD_tcf_N_{}_B_{}_inst_{}_T_{}_dt_{}_SAMP1.dat'.format(N*100,MD_System.beta,i,T,deltat),'wb')
        pickle.dump(tcf,f)
        #plt.plot(tcf_tarr,tcf,color='g')
        #plt.plot(tcf_tarr,np.cos(tcf_tarr),color='g')
        f.close()
        print(i,'completed')
        print('time',time.time() - start_time)
