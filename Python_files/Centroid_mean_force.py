#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 17:32:07 2019

@author: vgs23
"""

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy,copy
from numpy import linalg as LA
from numpy.fft import fft,ifft
from multiprocessing import Process
import multiprocessing as mp
from Langevin_thermostat import thermalize
from functools import partial
import scipy
from scipy import interpolate
#from MD_System import swarm,pos_op,beta,dimension,n_beads,w_n#,dpotential
import MD_System
import glob
import importlib
import Velocity_verlet
import time
import pickle
start_time = time.time()

data = np.loadtxt(glob.glob('*08B_064NB_257NS_avg_mf.dat')[0])



def mean_force_calculation(dpotential):
    print(MD_System.n_beads,MD_System.w_arr)
    N = 100
    deltat = 1e-2
    Q = np.linspace(-10,10,100)
    swarmobject = MD_System.swarm(N)
    rand_boltz = np.random.RandomState(1000)
    swarmobject.p = rand_boltz.normal(0.0,swarmobject.m/MD_System.beta,(swarmobject.N,MD_System.dimension,MD_System.n_beads))
    print(np.shape(swarmobject.p))
    n_sample = 10
    rng=np.random.RandomState(1000)
    CMD=1
    
    CMD_force = np.zeros((len(Q),MD_System.dimension))
    
    for j in range(len(Q)):
        
        swarmobject.q = np.zeros((swarmobject.N,MD_System.dimension,MD_System.n_beads)) + Q[j]
        #print(swarmobject.q)
        #print(np.shape(swarmobject.q))
        thermalize(CMD,0,0,swarmobject,dpotential,deltat,5,rng)
        
        for i in range(n_sample):
            CMD_force[j] += np.mean(dpotential(swarmobject.q))
            thermalize(CMD,0,0,swarmobject,dpotential,deltat,2,rng)
    
    
    CMD_force/=n_sample    
        
    f = open("CMD_Force_Morse_B_{}_NB_{}_100_10.dat".format(MD_System.beta,MD_System.n_beads),'w+')
    CMD_force = CMD_force[:,0]
    print(CMD_force)
    for x in zip(Q,CMD_force):
        f.write("{}\t{}\n".format(x[0],x[1]))
    f.close()

#MD_System.system_definition(8,64,MD_System.dpotential)
#
#importlib.reload(Velocity_verlet)
#mean_force_calculation(MD_System.dpotential)

def mean_force_calculation_2D(dpotential):
    print(MD_System.n_beads,MD_System.w_arr)
    N = 100
    deltat = 1e-2
    x = np.linspace(-5,5,101)
    y = np.linspace(-5,5,101)
    X,Y = np.meshgrid(x,y)
    
    swarmobject = MD_System.swarm(N)
    rand_boltz = np.random.RandomState(1000)
    swarmobject.p = rand_boltz.normal(0.0,swarmobject.m/MD_System.beta,(swarmobject.N,MD_System.dimension,MD_System.n_beads))
    print(np.amax(swarmobject.p))
    n_sample = 10
    rng=np.random.RandomState(1000)
    CMD=1
    
    
    CMD_force = np.zeros((len(x),len(y),MD_System.dimension))
    
    for i in range(len(x)):
        for j in range(len(y)):
            swarmobject.q = np.zeros((swarmobject.N,MD_System.dimension,MD_System.n_beads)) 
            
            swarmobject.q[:,0,:] = X[i][j]
            swarmobject.q[:,1,:] = Y[i][j]

        
            thermalize(CMD,0,0,swarmobject,dpotential,deltat,2,rng)
            for k in range(n_sample):
                #print(np.amax(swarmobject.p))
                force = np.mean(dpotential(swarmobject.q),axis=2)
                force = np.mean(force,axis=0)
                CMD_force[i][j] += force
                thermalize(CMD,0,0,swarmobject,dpotential,deltat,2,rng)
        print(time.time()-start_time)
    CMD_force/=n_sample
    
    f = open("CMD_Force_2D_B_{}_NB_{}_101_10.dat".format(MD_System.beta,MD_System.n_beads),'wb')
   
    plt.imshow(CMD_force[:,:,1],origin='lower')    
#    for i in range(len(x)):
#        for j in range(len(y)):
#            f.write("{}\t{}\t{}\n".format(X[i][j],Y[i][j],CMD_force[i][j]))
    

    pickle.dump(x,f)
    pickle.dump(y,f)
    pickle.dump(CMD_force,f)
    f.close()


#    l=64-15
#    u=64+15
#    plt.plot(Q[l:u],CMD_force[l:u])
#    ##plt.scatter(Q,CMD_force)
#    plt.plot(data[l:u,0],-data[l:u,1],color='r')
#    plt.show()
    #return CMF
#mean_force_calculation()
#print(mean_force(4.0))

#print(CMD_force)
    