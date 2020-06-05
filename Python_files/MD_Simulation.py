#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 14:43:00 2019

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
from Matsubara_potential import dpot
from Matsubara_potential_hh import dpot_hh
#import Centroid_mean_force
import Ring_polymer_dynamics
#import Centroid_dynamics
import Quasicentroid_dynamics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import Potentials_2D
import csv

potential = Potentials_2D.potential_cb
dpotential = Potentials_2D.dpotential_cb

if(0):
    x_ar = np.arange(0.1,6.0,0.1)
    y_ar = np.arange(0.1,6.0,0.1)
    X,Y = np.meshgrid(x_ar,y_ar)
    #plt.imshow(potential(X,Y))
    
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    plt.plot(x_ar,Potentials_2D.dpotential_cb_x(x_ar,0*y_ar))
    #Z = potential(X,Y)
    #Z[Z>1] = 1
    #ax.plot_surface(X,Y,Z)
    plt.show()

deltat = 5.0
N = 100
T = 20000.0#5.0*deltat#5000.0*deltat#3000*deltat
tcf_tarr = np.arange(0,T+0.0001,T/100.0)
tcf = np.zeros_like(tcf_tarr) 

beads=10
T_K = 200
beta= 315773/T_K#500.0    # Take care of redefinition of beta, when using QCMD!
n_instance = 10
print('N',N)#*100*n_instance)
print('Beta T_K',MD_System.beta, T_K)
print('deltat',deltat)
print('beads',beads)

if(0):
    rforce = Quasicentroid_dynamics.QCMD_instance(beads,dpotential,beta)
    print('time',time.time() - start_time)
    
    def force_xy(Q):
        x = Q[:,0,...]
        y = Q[:,1,...]
        r = (x**2+y**2)**0.5
        force_r = rforce(r)
        #force_xy = force_r*abs(np.array([x/r,y/r]))
        dpotx = force_r*(x/r)
        dpoty = force_r*(y/r)
        ret = np.transpose(np.array([dpotx,dpoty]),(1,0,2))
        return ret#np.array([dpotx,dpoty])
    #force_xy = np.vectorize(force_xy)
    
    x_ar = np.concatenate([np.arange(0.1,2.01,0.01),np.arange(-2.0,-0.1,0.01)])
    y_ar = np.concatenate([np.arange(0.1,2.01,0.01),np.arange(-2.0,-0.1,0.01)])
    X,Y = np.meshgrid(x_ar,y_ar)
    print(np.shape(x_ar),np.shape(y_ar))
    
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #Z = force_xy(X,Y)[1,:,:]
    #ax.plot_surface(X,Y,Z) 
    #plt.imshow(force_xy(X,Y)[0,:,:])
    #plt.show()
    
    Quasicentroid_dynamics.compute_tcf_QCMD(n_instance,N,force_xy,beta,T,deltat)
if(1):
        beta_arr = [315773/200,315773/400,315773/600,315773/800]
        T_arr = [200,400,600,800]
        color_arr = ['r','g','b','m']
        for (betaa,colour,t_k) in zip(beta_arr,color_arr,T_arr): 
            tcf = np.zeros_like(tcf_tarr)
            #n_instance = 1
            for i in range(n_instance):
                f = open('/home/vgs23/Pickle_files/QCMD_tcf_N_{}_B_{}_inst_{}_T_{}_dt_{}_SAMP1.dat'.format(N*100,betaa,i,T,deltat),'rb')
                #print('/home/vgs23/Pickle_files/QCMD_tcf_N_{}_B_{}_inst_{}_T_{}_dt_{}_SAMP1.dat'.format(N*100,betaa,i,T,deltat),'rb') 
                tcf += pickle.load(f)
                f.close()
                #print(i,'completed')
           
            tcf/=(n_instance)
            store_arr = zip(tcf_tarr,tcf)
            print(store_arr)
            txt_file =  open('/home/vgs23/Pickle_files/QCMD_tcf_N_{}_T_{}K_Time_{}_dt_{}.csv'.format(N*100,t_k,T,deltat),'w')
            writer = csv.writer(txt_file, delimiter='\t')
            writer.writerows(store_arr)
            print(tcf[0],tcf[3],tcf[7])
            plt.plot(tcf_tarr,tcf,color=colour)
            #plt.plot(tcf_tarr,np.cos(tcf_tarr)/(MD_System.beta*MD_System.n_beads),color='g')
        plt.show()
        
if(0):
    """
    This code is for the Adiabatic implementation of Quasicentroid 
    Molecular dynamics.
    """
    Quasicentroid_dynamics.compute_tcf_AQCMD(n_instance,N,beads,dpotential,beta,T,deltat)
    
    for i in range(n_instance):
        f = open('/home/vgs23/Pickle_files/AQCMD_tcf_N_{}_B_{}_inst_{}_T_{}_dt_{}_NB_{}.dat'.format(N*100,beta*beads,i,T,deltat,beads),'rb')
        tcf+= pickle.load(f)
        #plt.plot(tcf_tarr,tcf,color='r')
        #plt.plot(tcf_tarr,np.cos(tcf_tarr),color='g')
        f.close()
        print(i,'completed')
        print('time',time.time() - start_time)
    
    tcf/=(n_instance)
    plt.plot(tcf_tarr,tcf)

    tcf*=0.0
    for i in range(n_instance):
        f = open('/home/vgs23/Pickle_files/QCMD_tcf_N_{}_B_{}_inst_{}_T_{}_dt_{}_SAMP1.dat'.format(N*100,beta,i,T,0.5),'rb')
        tcf += pickle.load(f)
        f.close()
        print(i,'completed')
    plt.plot(tcf_tarr,tcf,color='g')
    plt.show()

if(0):
    
    """
    This code is for Adiabatic implementation of Centroid Molecular 
    dynamics.
    """
    Centroid_dynamics.compute_tcf(n_instance,N,beads,dpotential_morse,beta,T,deltat)

    for i in range(n_instance):
        f = open('/home/vgs23/Pickle_files/ACMD_tcf_N_{}_B_{}_inst_{}_dt_{}_NB_{}.dat'.format(N*100,MD_System.beta*MD_System.n_beads,i,deltat,MD_System.n_beads),'rb')
        tcf += pickle.load(f)
        f.close()
        print(i,'completed')
    
    tcf/=(n_instance)
    #tcf*=beads
    #print(tcf[0]*beads, tcf[5]*beads,tcf[45]*beads,tcf[89]*beads)
    print(tcf[0],tcf[3],tcf[7])
    plt.plot(tcf_tarr,tcf,color='r')
    #plt.plot(tcf_tarr,np.cos(tcf_tarr)/(MD_System.beta*MD_System.n_beads),color='g')
    plt.show()

#------------------------------------------------------- RPMD
if(0):
    #### The clumsy use of beta and beta_n interchangeably may 
    #### come back to haunt in the future. Whenever this code is used again,
    #### it has to be ensured that the temperature terms are all alright. 
   
    Ring_polymer_dynamics.compute_tcf(n_instance,N,beads,dpotential,beta,T,deltat)
    
    for i in range(n_instance):
        f = open('/home/vgs23/Pickle_files/RPMD_tcf_N_{}_B_{}_inst_{}_dt_{}_NB_{}.dat'.format(N*100,MD_System.beta*MD_System.n_beads,i,deltat,MD_System.n_beads),'rb')
        tcf += pickle.load(f)
        f.close()
        print(i,'completed')
    
    tcf/=(n_instance)
    #tcf*=beads
    #print(tcf[0]*beads, tcf[5]*beads,tcf[45]*beads,tcf[89]*beads)
    print(tcf[0],tcf[3],tcf[7])
    plt.plot(tcf_tarr,tcf,color='r')
    #plt.plot(tcf_tarr,np.cos(tcf_tarr)/(MD_System.beta*MD_System.n_beads),color='g')
    plt.show()
#---------------------------------------------------------CMD
if(0):
    xforce = Centroid_dynamics.CMD_instance(beads,dpotential_morse,beta)
#    Q=np.arange(0,10,0.01) 
#    plt.plot(Q,xforce(Q))
#    plt.plot(Q,dpotential_morse(Q))
#    plt.show()
    
    Ring_polymer_dynamics.compute_tcf(n_instance,N,1,xforce,beta, T,deltat)
    for i in range(n_instance):
        f = open('/home/vgs23/Pickle_files/CMD_tcf_N_{}_B_{}_inst_{}_dt_{}_NB_{}.dat'.format(N*100,MD_System.beta*MD_System.n_beads,i,deltat,MD_System.n_beads),'rb')
        tcf += pickle.load(f)
        f.close()
        print(i,'completed')
        
    tcf/=(n_instance)
    #tcf*=beads
    #print(tcf[0]*beads, tcf[5]*beads,tcf[45]*beads,tcf[89]*beads)
    print(tcf[0],tcf[3],tcf[7])
    plt.plot(tcf_tarr,tcf,color='r')
    #plt.plot(tcf_tarr,np.cos(tcf_tarr)/(MD_System.beta*MD_System.n_beads),color='g')
    plt.show()
#---------------------------------------------------------Matsubara

#n_mats_instance = 1
#print('N',N*100*n_mats_instance)
#print('Beta',MD_System.beta)
#print('deltat',deltat)
#
#for i in range(n_mats_instance):
#    tcf =  Matsubara_instance((i+1)*100,M)
#    f = open('Matsubara_tcf_N_{}_B_{}_inst_{}_dt_{}_M_{}.dat'.format(N*100,MD_System.beta,i,deltat,M),'wb')
#    pickle.dump(tcf,f)
#    f.close()
#    print(i,'completed')
#    print('time',time.time() - start_time)


  

#--------------------------------------

#n=n_mats_instance
#for i in range(n):
#    #tcf =  Matsubara_instance((i+1)*100)
#    f = open('Matsubara_tcf_N_{}_B_{}_inst_{}_dt_{}_M_{}.dat'.format(N*100,MD_System.beta,i,deltat,M),'rb')
#    tcf+=pickle.load(f)
#    #plt.plot(tcf_tarr,np.real(tcf)/(i+1))
#    #plt.plot(tcf_tarr,np.imag(tcf)/(i+1))
#    f.close()
#    #print(i,'completed')
#tcf/=n
#print(tcf[0])
  
#-------------------------------------

#f = open('Matsubara_tcf_N_{}_B_{}.dat'.format(N*100,MD_System.beta*MD_System.n_beads),'rb')
#tcf = pickle.load(f)
#f.close()

#data = np.loadtxt('mats_quart_qq_tcf_beta8nrp17mrp3numtraj640000.dat')
##data = np.loadtxt('mats_quart_qq_tcf_beta8nrp17mrp1numtraj640000.dat')
##data = np.loadtxt('quartic_M1_beta2_s1_CQQ.dat')
#print(data[0,0],data[0,1])
##print(tcf[0])
#
#plt.plot(tcf_tarr,np.real(tcf))
#plt.plot(tcf_tarr,np.imag(tcf),color='r')
##plt.plot(data[:,0],data[:,1])
#plt.show()
#plt.savefig('tcf.png')
##plt.plot(data[:,0],data[:,1],color='r')
#
#print('time',time.time() - start_time)


#----------------------------------------------------------CMD

#MD_System.system_definition(8,64,MD_System.dpotential)
##
##importlib.reload(Velocity_verlet)
#data = np.loadtxt("CMD_Force_B_{}_NB_{}.dat".format(MD_System.beta,MD_System.n_beads))
#
#
##print(data)
#
##print(data)
#Q = data[:,0]
#CMD_force = data[:,1]
##print(np.shape(data))
#CMF = scipy.interpolate.interp1d(Q,CMD_force,kind='cubic')
#
##fig, ax = plt.subplots()
##ax.plot(Q,-MD_System.dpotential(Q),color='g')
##ax.plot(Q,-CMF(Q),color='r')
##ax.set_xlim((-2,2))
##ax.set_ylim((-2,2))
#
#
#
#MD_System.system_definition(8,1,CMF)
#importlib.reload(Velocity_verlet)
#
#swarmobject = MD_System.swarm(N)
#swarmobject.q = np.zeros((swarmobject.N,MD_System.dimension,1))
#
#rand_boltz = np.random.RandomState(1000)
#swarmobject.p = rand_boltz.normal(0.0,swarmobject.m/MD_System.beta,np.shape(swarmobject.q))
#
#pool = mp.Pool(mp.cpu_count())
#CMD = 0
#func = partial(corr_function_upgrade,CMD,swarmobject,CMF,MD_System.pos_op,MD_System.pos_op,T,deltat)
#
#n_inst = 10
##print('here')
#results = pool.map(func, range(n_inst))
##print('there')
#pool.close()
#pool.join()
#
#tcf = np.sum(results,0)/n_inst
#
#
##tcf = corr_function(swarmobject, pos_op,pos_op,20*np.pi,1)
#tcf_tarr = np.arange(0,T+0.0001,T/500.0)
#
#data = np.loadtxt('170507_022844_08B_064NB_129NS_avg_cxx.dat')
#
#plt.plot(tcf_tarr,tcf)
#plt.plot(data[:,0],data[:,1],color='r')


#--------------------------------------------------------RPMD

#swarmobject = swarm(N)
#swarmobject.q = np.zeros((swarmobject.N,dimension,n_beads))
##print(swarmobject.q)
#rand_boltz = np.random.RandomState(1000)
#swarmobject.p = rand_boltz.normal(0.0,swarmobject.m/beta,np.shape(swarmobject.q))
##print(swarmobject.p)
#pool = mp.Pool(mp.cpu_count())
#
#func = partial(corr_function_upgrade,swarmobject,pos_op,pos_op,T,deltat)
#
#n_inst = 10
##print('here')
#results = pool.map(func, range(n_inst))
##print('there')
#pool.close()
#pool.join()
#
#tcf = np.sum(results,0)/n_inst
#
#
##tcf = corr_function(swarmobject, pos_op,pos_op,20*np.pi,1)
#tcf_tarr = np.arange(0,T+0.0001,T/500.0)
#
#plt.plot(tcf_tarr,tcf)


#---------------------------------------------------------STRAY CODE

#Q = np.linspace(-10,10,129)
##print(Q)
#
#data = np.loadtxt('170429_183220_01B_008NB_129NS_avg_mf.dat')
#
#swarmobject = swarm(N)
#
#rand_boltz = np.random.RandomState(1000)
#swarmobject.p = rand_boltz.normal(0.0,swarmobject.m/beta,(swarmobject.N,dimension,n_beads))
#
#n_sample = 10
#rng=np.random.RandomState(1000)
#CMD=1
#
#CMD_force = np.zeros((len(Q),dimension))
#
#
##-------
##swarmobject.q = np.zeros((swarmobject.N,dimension,n_beads))
##rand_boltz.normal(0.0,swarmobject.m/beta,(swarmobject.N,dimension,n_beads))
##
##thermalize(CMD,swarmobject,dpotential,deltat,2e0,rng)
##-------
#for j in range(len(Q)):
#    
#    swarmobject.q = np.zeros((swarmobject.N,dimension,n_beads)) + Q[j]
#    thermalize(CMD,swarmobject,dpotential,deltat,5,rng)
#    
#    for i in range(n_sample):
#        CMD_force[j] += np.mean(dpotential(swarmobject.q))
#        thermalize(CMD,swarmobject,dpotential,deltat,2,rng)
#
#
#CMD_force/=n_sample    
#
#mean_force = scipy.interpolate.interp1d(Q,CMD_force[:,0],kind='cubic')
##print(mean_force(4.0))
#
##print(CMD_force)
#l=64-10
#u=64+10
#plt.plot(Q[l:u],CMD_force[l:u])
###plt.scatter(Q,CMD_force)
#plt.plot(Q[l:u],-data[l:u,1],color='r')
##plt.show()
##plt.scatter(Q,-data[:,1])
