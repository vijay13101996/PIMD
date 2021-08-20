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

def BAQC_MF_calculation(swarmobject,dpotential,n_sample,Q,deltat):
    print(swarmobject.n_beads,swarmobject.w_arr) 
    BAQC_MF = np.zeros((len(Q),swarmobject.dimension)) 
    rand_boltz = np.random.RandomState(1)
    swarmobject.p = rand_boltz.normal(0.0,swarmobject.m/swarmobject.beta,(swarmobject.N,swarmobject.dimension,swarmobject.n_beads))
    rng=np.random.RandomState(1)
    
    for j in range(len(Q)):
        swarmobject.q = np.tile([Q[j],0.0,0.0],(swarmobject.N,swarmobject.n_beads)) #np.zeros((swarmobject.N,swarmobject.dimension,swarmobject.n_beads))  
        swarmobject.q = np.reshape(swarmobject.q,(swarmobject.N,swarmobject.n_beads,swarmobject.dimension))
        swarmobject.q = np.transpose(swarmobject.q,[0,2,1])
        print('swarm',(swarmobject.q[0,:,0]))
        print('j and time',j, time.time()-start_time)   
        BAQC_q = np.zeros((swarmobject.N,swarmobject.dimension)) 
        BAQC_q[:,0] = Q[j] 
        BAQC_p = np.zeros_like(BAQC_q)
        lambda_curr = np.zeros((len(swarmobject.q),3))
        U = np.array([np.identity(3) for i in range(len(swarmobject.q))])
        print(np.shape(U))

        Langevin_thermostat.baqcmd_thermalize(1,swarmobject,BAQC_q,BAQC_p,U,lambda_curr,dpotential,deltat,10000,rng)
        print('Kinetic energy constrained', swarmobject.sys_kin())
        print('time', time.time() - start_time)
        #plt.scatter(swarmobject.q[0,0,:],swarmobject.q[0,1,:])
        #plt.show()
        for i in range(n_sample):
            BAQC_MF[j] += np.mean(Velocity_verlet.dpotential_baqcentroid(U,BAQC_q,swarmobject,dpotential),axis=0)
            print('i','QC_MF', BAQC_MF[j])
            #Langevin_thermostat.baqcmd_thermalize(1,swarmobject,BAQC_q,BAQC_p,U,lambda_curr,dpotential,deltat,5000,rng)   ### UNCOMMENT FOR MORE THAN ONE SAMPLE!
            print('time',time.time()-start_time, swarmobject.sys_kin())
    BAQC_MF/=n_sample
    ### BE CAREFUL WITH THE SIGN OF THE POTENTIAL HERE. You have used
    #   force and gradient of potential interchangeably
    f = open("/home/vgs23/Pickle_files/BAQCMD_Force_B_{}_NB_{}_N_{}_nsample_{}_deltat_{}_NQ_{}.dat".format(swarmobject.beta,swarmobject.n_beads,swarmobject.N,n_sample,deltat,len(Q)),'w+')
    
    BAQCMD_force = BAQC_MF[:,0]
    for x in zip(Q,BAQCMD_force):
        f.write("{}\t{}\n".format(x[0],x[1]))
    f.close()
   
    if(0):
        plt.plot(Q,BAQC_MF[:,0],color='r')
        plt.plot(Q,BAQC_MF[:,1],color='b')
        plt.show()
    

def BAQCMD_instance(beads,dpotential,beta,N,n_sample,Q,deltat):
    swarmobject = MD_System.swarm(N,beads,beta,3)
    importlib.reload(Velocity_verlet)
    
    #BAQC_MF_calculation(swarmobject,dpotential,n_sample,Q,deltat)
    
    f = open("/home/vgs23/Pickle_files/BAQCMD_Force_B_{}_NB_{}_N_{}_nsample_{}_deltat_{}_NQ_{}.dat".format(swarmobject.beta,swarmobject.n_beads,N,n_sample,deltat,len(Q)),'rb')       
    data = np.loadtxt(f)
    Q = data[:,0]
    BAQCMD_force = data[:,1]
    
    rforce = scipy.interpolate.interp1d(Q,BAQCMD_force,kind='cubic')
    if(1):
        plt.plot(Q,BAQCMD_force)
        plt.plot(Q,rforce(Q))
        plt.show()
    return rforce
    
    
def ABAQCMD_instance(rng,N,beads,dpotential,beta,T,deltat):
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

def compute_tcf_ABAQCMD(n_QCMD_instance,N,beads,dpotential,beta,T,deltat):
    print('beta',beta)
    for i in range(n_QCMD_instance):
        tcf =  AQCMD_instance((i+1)*100,N,beads,dpotential,beta,T,deltat)
        f = open('/home/vgs23/Pickle_files/ABAQCMD_tcf_N_{}_B_{}_inst_{}_T_{}_dt_{}_NB_{}.dat'.format(N*100,MD_System.beta*MD_System.n_beads,i,T,deltat,MD_System.n_beads),'wb')
        pickle.dump(tcf,f)
        #plt.plot(tcf_tarr,tcf,color='r')
        #plt.plot(tcf_tarr,np.cos(tcf_tarr),color='g')
        f.close()
        print(i,'completed')
        print('time',time.time() - start_time)
        
    return tcf
        
def compute_tcf_BAQCMD(n_instance,N,dpotential,beta,T,n_tp,deltat):
    
    for i in range(n_instance):
        tcf =  Ring_polymer_dynamics.RPMD_instance((i+1)*100,N,1,dpotential,beta,T,n_tp,deltat)
        print('beta',beta,'T',T)
        f = open('/home/vgs23/Pickle_files/BAQCMD_tcf_N_{}_B_{}_inst_{}_T_{}_dt_{}_SAMP1.dat'.format(N*100,beta,i,T,deltat),'wb')
        pickle.dump(tcf,f)
        #plt.plot(tcf_tarr,tcf,color='g')
        #plt.plot(tcf_tarr,np.cos(tcf_tarr),color='g')
        f.close()
        print(i,'completed')
        print('time',time.time() - start_time)
    
    return tcf
