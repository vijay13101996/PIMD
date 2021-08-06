import numpy as np
from Correlation_function import OTOC_Classical, corr_function_upgrade
import MD_System
import multiprocessing as mp
from functools import partial
import importlib
import pickle
import time

def Classical_OTOC_instance(rng,N,dpotential,ddpotential,beta,T,n_tp, deltat):
    swarmobject = MD_System.swarm(N,1,beta,1)
    swarmobject.q = np.zeros((swarmobject.N,swarmobject.dimension,swarmobject.n_beads)) 
    rand_boltz = np.random.RandomState(rng)
    swarmobject.p = rand_boltz.normal(0.0,swarmobject.m/swarmobject.beta,np.shape(swarmobject.q))
     
    if(1):
        pool = mp.Pool(5)
        n_inst = 5
        func = partial(OTOC_Classical,swarmobject,dpotential,ddpotential,T,n_tp,deltat)
        results = pool.map(func, range(n_inst))
        pool.close()
        pool.join()
        return np.sum(results,0)/n_inst

    if(0):
        tcf = OTOC_Classical(swarmobject,dpotential,ddpotential,T,n_tp,deltat,rng)
        return tcf

def Classical_tcf_instance(rng,N,dpotential,beta,T,n_tp, deltat):
    swarmobject = MD_System.swarm(N,1,beta,1)
    swarmobject.q = np.zeros((swarmobject.N,swarmobject.dimension,swarmobject.n_beads)) 
    rand_boltz = np.random.RandomState(rng)
    swarmobject.p = rand_boltz.normal(0.0,swarmobject.m/swarmobject.beta,np.shape(swarmobject.q))
     
    if(1):
        pool = mp.Pool(5)
        n_inst = 5
        func =partial(corr_function_upgrade,0,0,0,1,swarmobject,dpotential,swarmobject.pos_op,swarmobject.pos_op,T,n_tp,deltat)
        results = pool.map(func, range(n_inst))
        pool.close()
        pool.join()
        return np.sum(results,0)/n_inst

    if(0):
        tcf = corr_function_upgrade(0,0,0,1,swarmobject,dpotential,swarmobject.pos_op,swarmobject.pos_op,T,n_tp,deltat,rng)
        return tcf
  
def compute_classical_tcf(classical_instance,N,dpotential,beta,T,n_tp,deltat): 
    start_time = time.time()    
    for i in classical_instance:
        tcf = Classical_tcf_instance(i,N,dpotential,beta,T,n_tp, deltat)
        f = open('/home/vgs23/Pickle_files/Classical_tcf_N_{}_B_{}_inst_{}_dt_{}.dat'.format(N,beta,i,deltat),'wb')
        pickle.dump(tcf,f)
        f.close()
        print(i,'completed')
        print('time',time.time() - start_time)        
    return tcf

def compute_classical_OTOC(classical_instance,N,dpotential,ddpotential,beta,T,n_tp,deltat):
    start_time = time.time()    
    for i in classical_instance:
        tcf = Classical_OTOC_instance(i,N,dpotential,ddpotential,beta,T,n_tp, deltat)
        f = open('/home/vgs23/Pickle_files/Classical_OTOC_N_{}_B_{}_inst_{}_dt_{}.dat'.format(N,beta,i,deltat),'wb')
        pickle.dump(tcf,f)
        f.close()
        print(i,'completed')
        print('time',time.time() - start_time)
        
    return tcf


