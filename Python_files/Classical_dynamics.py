import numpy as np
import Correlation_function
from Correlation_function import OTOC_Classical, corr_function_upgrade
import MD_System
import multiprocessing as mp
from functools import partial
import importlib
import pickle
import time
import utils
import Langevin_thermostat
from matplotlib import pyplot as plt

def Classical_OTOC(N,beta,thermtime,deltat,dpotential,ddpotential,tcf_tarr,fprefix,rngSeed):
        swarmobject = MD_System.swarm(N,1,beta,1) #Change dimension when required
        swarmobject.q = np.zeros((swarmobject.N,swarmobject.dimension,swarmobject.n_beads)) + 1.0
        rng = np.random.RandomState(rngSeed)
        swarmobject.p = rng.normal(0.0,swarmobject.m/swarmobject.beta,np.shape(swarmobject.q))
      
        Langevin_thermostat.thermalize(0,0,0,1,swarmobject,dpotential,deltat,thermtime,rng)
        print('KE', swarmobject.p**2)
        tcf = Correlation_function.OTOC_Classical(swarmobject,dpotential,ddpotential,tcf_tarr,rng)        
        fname = '{}_S_{}'.format(fprefix,rngSeed)
        utils.store_1D_plotdata(tcf_tarr,tcf,fname)
        print('Seed = {}, done'.format(rngSeed))

        plt.plot(tcf_tarr,np.log(abs(tcf)))
        f =  open("/home/vgs23/Pickle_files/OTOC_{}_beta_{}_basis_{}_n_eigen_{}_tfinal_{}.dat".format('inv_harmonic',0.2,50,50,4.0),'rb+')
        t_arr = pickle.load(f,encoding='latin1')
        OTOC_arr = pickle.load(f,encoding='latin1')
        plt.plot(t_arr,np.log(OTOC_arr), linewidth=3,label='Quantum OTOC')
        f.close()

        plt.show() 
        
def Classical_OTOC_instance(N,beta,thermtime,deltat,dpotential,ddpotential,tcf_tarr,fprefix,inst,ctx):       
        func = partial(Classical_OTOC,N,beta,thermtime,deltat,dpotential,ddpotential,tcf_tarr, fprefix)
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


