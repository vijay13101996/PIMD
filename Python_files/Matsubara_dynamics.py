#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 14:37:29 2019

@author: vgs23
"""
import numpy as np
import Correlation_function
from Correlation_function import compute_phase_histogram,corr_function_matsubara, corr_function_phaseless_Matsubara, OTOC_phase_dep_Matsubara, corr_function_phase_dep_Matsubara
import MD_System
import multiprocessing as mp        
from functools import partial
import importlib
import pickle
import time
import utils
import Langevin_thermostat
from matplotlib import pyplot as plt


def Matsubara_theta_OTOC(N,M,beta,thermtime,deltat,dpotential,ddpotential,tcf_tarr, fprefix, theta,rngSeed):
        
        swarmobject = MD_System.swarm(N,M,beta,1) #Change dimension when required
        swarmobject.q = np.zeros((swarmobject.N,swarmobject.dimension,swarmobject.n_beads)) + 1.0
        rng = np.random.RandomState(rngSeed)
        swarmobject.p = rng.normal(0.0,swarmobject.m/swarmobject.beta,np.shape(swarmobject.q))
      
        Langevin_thermostat.Theta_constrained_thermalize(M,swarmobject,theta,thermtime,dpotential,deltat,rng)
        tcf = Correlation_function.OTOC_theta(swarmobject,dpotential,ddpotential,theta,tcf_tarr,rng)        
        fname = '{}_theta_{}_S_{}'.format(fprefix,theta,rngSeed)
        utils.store_1D_plotdata(tcf_tarr,tcf,fname)
        print('theta = {} Seed = {}, done'.format(theta,rngSeed))

        #plt.plot(tcf_tarr,np.log(abs(tcf)))
        #plt.show() 
        
def Matsubara_phase_coverage(N,M,beta,n_sample,thermtime,deltat,dpotential,fprefix,ntheta,rngSeed):
        swarmobj = MD_System.swarm(N,M,beta,1) #Change dimension when required
        swarmobj.q = np.zeros((swarmobj.N,swarmobj.dimension,swarmobj.n_beads)) 
        rng = np.random.RandomState(rngSeed)
        swarmobj.p = rng.normal(0.0,swarmobj.m/swarmobj.beta,np.shape(swarmobj.q))
      
        denom = 0.0j
        B_hist = np.zeros(ntheta)
        thetagrid = np.zeros(ntheta)
        rearr = Correlation_function.rearrangearr(swarmobj.n_beads)
        
        Langevin_thermostat.thermalize(0,0,1,swarmobj.n_beads,swarmobj,dpotential,deltat,thermtime,rng)
        theta = Correlation_function.matsubara_phase_factor(swarmobj.n_beads,swarmobj,rearr)
        maxtheta = max(theta) + 0.1*max(theta)
        mintheta = min(theta) - 0.1*min(theta)
        
        for i in range(n_sample):
            Langevin_thermostat.thermalize(0,0,1,swarmobj.n_beads,swarmobj,dpotential,deltat,thermtime,rng)
            theta = Correlation_function.matsubara_phase_factor(swarmobj.n_beads,swarmobj,rearr)
            hist, bin_edges = np.histogram(theta, bins=ntheta, range=(mintheta,maxtheta),density=True)
            B_hist+=hist/hist.sum()
            thetagrid+=bin_edges[1:]
            #print('hist',bin_edges)
            #plt.hist(theta,bins = ntheta, range=(-100.0,100.0),density=True)
            
            #plt.plot(bin_edges[1:],hist)
            #plt.show()
            exponent = 1j*(swarmobj.beta)*theta
            denom+= np.sum(np.exp(exponent))
            #print('denom', np.sum(np.exp(exponent))/swarmobj.N)
         
        denom/=(swarmobj.N*n_sample)
        B_hist/=n_sample
        thetagrid/=n_sample
        plt.plot(thetagrid,B_hist)
        plt.show()
        fname = '{}_theta_histogram_S_{}'.format(fprefix,rngSeed)
        utils.store_1D_plotdata(thetagrid,B_hist,fname)

        #print('denom final, B_hist ', denom, B_hist.sum())

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
        pool = mp.Pool(6)
        n_inst = 6
        func = partial(corr_function_phase_dep_Matsubara,swarmobject,dpotential,swarmobject.pos_op,swarmobject.pos_op,T,n_tp,theta_arr,deltat)
        results = pool.map(func, range(n_inst))
        pool.close()
        pool.join()
        return np.sum(results,0)/n_inst

        tcf_thetat = corr_function_phase_dep_Matsubara(swarmobject,dpotential,swarmobject.pos_op,swarmobject.pos_op,T,n_tp,theta_arr,deltat,rng)
        return tcf_thetat
        
def Phase_dep_OTOC_instance(N,M,beta,thermtime,deltat,dpotential,ddpotential,tcf_tarr, fprefix, theta,inst,ctx):
        ### FOR M=3, timeout = 300.0(ET: 120s), M = 5, timeout = 1000.0(ET: 330s) 
        func = partial(Matsubara_theta_OTOC,N,M,beta,thermtime,deltat,dpotential,ddpotential,tcf_tarr, fprefix, theta)
        inst_split = utils.chunks(inst,100)
        #print('inst_split', inst_split)
        for inst_batch in inst_split:
                procs = []
                for i in inst_batch:
                        p = mp.Process(target=func, args=(i,))
                        procs.append(p) 
                        p.start()
                        #print('start', p.name)
                for p in procs:
                        p.join(300.0)
                        if p.is_alive():
                                p.terminate()
                                print('end', p.name)
 
        #pool = ctx.Pool(min(10,len(inst)))
        #func = partial(Matsubara_theta_OTOC,N,M,beta,thermtime,deltat,dpotential,ddpotential,tcf_tarr, fprefix, theta)
        #pool_map = pool.map(func, inst)
        #pool.close()
                
def compute_phase_dep_OTOC(Matsubara_instance,N,M,dpotential,ddpotential,beta,T,n_tp,deltat, theta_arr):
    start_time = time.time()    
    for i in Matsubara_instance:
        ntheta= 1
        tcf = Phase_dep_OTOC_instance(i,N,M,dpotential,ddpotential,beta,T,n_tp, deltat, theta_arr)
        f = open('/home/vgs23/Pickle_files/Matsubara_phase_dep_OTOC_N_{}_B_{}_inst_{}_dt_{}_M_{}_ntheta_{}.dat'.format(N,beta,i,deltat,M,len(theta_arr)),'wb')
        pickle.dump(tcf,f)
        f.close()
        print(i,'completed',tcf[0])
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


