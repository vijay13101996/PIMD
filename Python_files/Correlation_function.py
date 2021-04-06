#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 14:41:26 2019

@author: vgs23
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from numpy.fft import fft,ifft
from multiprocessing import Process
import MD_System
import scipy
#from System_potential import dpotential,potential
from MD_Evolution import time_evolve,time_evolve_nc,time_evolve_qcmd
from Langevin_thermostat import thermalize, qcmd_thermalize, Andersen_thermalize, Theta_constrained_thermalize
import psutil
import time
import sys
import Velocity_verlet
import Monodromy

def corr_function_QCMD(swarmobj,QC_q,QC_p,lambda_curr,A,B,time_corr,deltat,dpotential,rng_ind):
    rng = np.random.RandomState(rng_ind)
    n_tp = 10
    tcf_tarr = np.arange(0,time_corr+0.0001,time_corr/n_tp)
    tcf = np.zeros_like(tcf_tarr)
    tcf_taux = np.arange(0,2*time_corr+0.0001,time_corr/n_tp)
    A_arr = np.zeros((len(tcf_taux),)+QC_q.shape)
    B_arr = np.zeros_like(A_arr)
    n_approx = 1
    qcmd_thermalize(0,swarmobj,QC_q,QC_p,lambda_curr,dpotential,deltat,1000,rng)
    kin_en = swarmobj.sys_kin()
    print('kin en', kin_en)
    print('kin_en qc',np.sum(QC_p*QC_p)/(2*swarmobj.m))
    qcx=[]#QC_q[:,0]
    qcy=[]#QC_q[:,1]
    
    for j in range(n_approx):
        
        A_arr[0] = A(QC_q,QC_p)
        B_arr[0] = B(QC_q,QC_p)
        print('j',j)
        for i in range(1,len(tcf_taux)):
            time_evolve_qcmd(swarmobj, QC_q,QC_p,lambda_curr,dpotential, deltat, tcf_taux[i]-tcf_taux[i-1],rng)
            A_arr[i] = A(QC_q,QC_p)
            B_arr[i] = B(QC_q,QC_p)
        
        A_arr[len(tcf):,:,:] = 0.0
        tcf_cr = convolution(A_arr,B_arr,0)
        tcf_cr = tcf_cr[:len(tcf),:,:] #Truncating the padded part
        tcf_cr = np.sum(tcf_cr,axis=2) #Summing over the dimension (Dot product)
        tcf+= np.sum(tcf_cr,axis=1)
        
#        for i in range(len(tcf)):
#            for k in range(len(tcf)):
#                #print(A_arr[i+j]*B_arr[j] + B_arr[i+j]*A_arr[j])
#                tcf[i] += np.sum(A_arr[i+k]*B_arr[k] + B_arr[i+k]*A_arr[k]) 
        
        print('j finished', time.time())
        qcmd_thermalize(0,swarmobj,QC_q,QC_p,lambda_curr,dpotential,deltat,500.0,rng)
    
    print('Dimension:', MD_System.dimension)
    tcf/=(n_approx*swarmobj.N*len(tcf))#*MD_System.dimension)    ## BE CAREFUL HERE!!!
    return tcf

def corr_function(swarmobj,A,B,time_corr,deltat,rng_ind): 
    """ 
    This function computes the time correlation function for one instance
    of the system prepared by seeding with rng_ind
    
    Arguments:
        swarmobj: Swarm class object
        A: First operator, to compute correlation (A function of p,q)
        B: Second operator, to compute correlation (A function of p,q)
        time_corr : Total duration of correlation (float)
        rng_ind : Integer, used to seed the Randomstate object
        
    """
    rng = np.random.RandomState(rng_ind)
    tcf_tarr = np.arange(0,time_corr+0.0001,time_corr/500.0)
    tcf = np.zeros_like(tcf_tarr)
    n_approx = 10
    thermalize(swarmobj,dpotential,deltat,10*np.pi,rng)
      
    for j in range(n_approx):
        A_0 = A(swarmobj.q,swarmobj.p)
        tcf[0]+= np.sum(A_0*A_0)
        # The above line have to be treated on a case to case basis
        for i in range(1,len(tcf)):
            time_evolve(swarmobj, dpotential, deltat, tcf_tarr[i]-tcf_tarr[i-1])
            B_t = B(swarmobj.q,swarmobj.p)
            tcf[i] += np.sum(A_0*B_t)
        thermalize(swarmobj,dpotential,deltat,2*np.pi,rng)
    tcf/=(n_approx*swarmobj.N)  #Change when you increase dimension
    
    return tcf

def corr_function_Andersen(ACMD,CMD,Matsubara,M,swarmobj,derpotential,A,B,time_corr,n_tp,deltat,rng_ind):
    rng = np.random.RandomState(rng_ind)
    tcf_tarr = np.arange(0,time_corr+0.0001,time_corr/n_tp)
    tcf = np.zeros_like(tcf_tarr)
   
    tcf_taux = np.arange(0,2*time_corr+0.0001,time_corr/n_tp)
    A_arr = np.zeros((len(tcf_taux),)+swarmobj.q.shape)
    B_arr = np.zeros_like(A_arr)

    A_centroid = np.zeros((len(tcf_taux),) + swarmobj.q[:,:,0].shape)
    B_centroid = np.zeros_like(A_centroid)
     
    start_time = time.time()
    print('time 1', time.time()-start_time)
    Andersen_thermalize(ACMD,CMD, Matsubara, M, swarmobj, 10, 20, derpotential, deltat, rng)
    
    n_approx = 5    ### Toggle thermalize on when using more than one sample
    print('kin en',swarmobj.sys_kin()/(len(swarmobj.q)*swarmobj.n_beads**2))
 
    for j in range(n_approx):
        print('j starting',j,time.time()-start_time)
        A_arr[0] = A()
        B_arr[0] = B()
        
        if(1):      
            for i in range(1,len(tcf_taux)):  
                if(psutil.virtual_memory()[2] > 60.0):
                    print('Memory limit exceeded, exiting...', psutil.virtual_memory()[2])
                    sys.exit()

                if(i%1000==0):
                     print(i, 'memory prev',psutil.virtual_memory()[3], psutil.virtual_memory()[2])       
                if(ACMD==0):
                    time_evolve(ACMD,CMD,Matsubara,M,swarmobj, derpotential, deltat, tcf_taux[i]-tcf_taux[i-1])
                else:
                    time_evolve_nc(ACMD,CMD,Matsubara,M,swarmobj, derpotential, deltat, tcf_taux[i]-tcf_taux[i-1],rng)  # ACMD!!!
                
                A_centroid[i] = np.sum(A(),axis=2)/swarmobj.n_beads
                B_centroid[i] = np.sum(B(),axis=2)/swarmobj.n_beads

                if(i%1000==0):
                    print( 'memory after',psutil.virtual_memory()[3], psutil.virtual_memory()[2])
        
        A_centroid[len(tcf):] = 0.0
        tcf_cr = convolution(A_centroid,B_centroid,0)
        tcf_cr = tcf_cr[:len(tcf),:,:] #Truncating the padded part
        tcf_cr = np.sum(tcf_cr,axis=2) #Summing over the dimension (Dot product)
        tcf+= np.sum(tcf_cr,axis=1)    #Summing over the particles
               # --- This part above is working alright!
        #print('j ended',j)
#        for i in range(len(tcf)):
#            for k in range(len(tcf)):
#                #print(A_arr[i+j]*B_arr[j] + B_arr[i+j]*A_arr[j])
#                tcf[i] += np.sum(A_centroid[i+k]*B_centroid[k] + B_centroid[i+k]*A_centroid[k]) 
    #tcf[i] += np.sum(A_0*B_t)
        
        Andersen_thermalize(ACMD,CMD, Matsubara, M, swarmobj, 10, 5, derpotential, deltat, rng)
        print('j ending',j,time.time()-start_time)
    
    tcf/=(n_approx*swarmobj.N*len(tcf))#*MD_System.dimension)
    print("RP Simulation Done, TCF length: ", len(tcf))
    #print('tcf', tcf)
    #!!! It is unclear if the TCF is to be scaled by dimension. Beware when you make any changes above.
    return tcf

def static_average(ACMD,CMD,Matsubara,M,swarmobj,derpotential,A,B,deltat,rng_ind):
    rng = np.random.RandomState(rng_ind)
    avg = 0.0
    ninst = 5
    for i in range(ninst):
        #thermalize(ACMD,CMD,Matsubara,M,swarmobj,derpotential,deltat,20,rng)
        Andersen_thermalize(ACMD,CMD, Matsubara, M, swarmobj, 10, 20, derpotential, deltat, rng)
        Aavg = np.sum(A(),axis=2)/swarmobj.n_beads
        Bavg = np.sum(B(),axis=2)/swarmobj.n_beads
        avg+= np.sum(Aavg*Bavg)
        print('avg', i, np.sum(Aavg*Bavg))
    print('avg', avg/(ninst*len(swarmobj.q)))

def corr_function_upgrade(ACMD,CMD,Matsubara,M,swarmobj,derpotential,A,B,time_corr,n_tp,deltat,rng_ind):
    rng = np.random.RandomState(rng_ind)
    tcf_tarr = np.arange(0,time_corr+0.0001,time_corr/n_tp)
    tcf = np.zeros_like(tcf_tarr)
   
    tcf_taux = np.arange(0,2*time_corr+0.0001,time_corr/n_tp)
    A_arr = np.zeros((len(tcf_taux),)+swarmobj.q.shape)
    B_arr = np.zeros_like(A_arr)

    A_centroid = np.zeros((len(tcf_taux),) + swarmobj.q[:,:,0].shape)
    B_centroid = np.zeros_like(A_centroid)
     
    start_time = time.time()
    print('time 1', time.time()-start_time)
    thermalize(ACMD,CMD,Matsubara,M,swarmobj,derpotential,deltat,50,rng)
    n_approx = 5    ### Toggle thermalize on when using more than one sample
    print('kin en',swarmobj.sys_kin()/(len(swarmobj.q)*swarmobj.n_beads**2))  # Should be 0.5 always!
    #The above line have to be treated on a case to case basis
    #Velocity_verlet.set_therm_param(deltat,64.0) 
    for j in range(n_approx):
        print('j starting',j,time.time()-start_time)
        A_arr[0] = A()
        B_arr[0] = B()
        
        if(1):      
            for i in range(1,len(tcf_taux)):  
                if(psutil.virtual_memory()[2] > 60.0):
                    print('Memory limit exceeded, exiting...', psutil.virtual_memory()[2])
                    sys.exit()

                if(i%1000==0):
                     print(i, 'memory prev',psutil.virtual_memory()[3], psutil.virtual_memory()[2])       
                if(ACMD==0):
                    time_evolve(ACMD,CMD,Matsubara,M,swarmobj, derpotential, deltat, tcf_taux[i]-tcf_taux[i-1])
                else:
                    time_evolve_nc(ACMD,CMD,Matsubara,M,swarmobj, derpotential, deltat, tcf_taux[i]-tcf_taux[i-1],rng)  # ACMD!!!
                                               
                #A_arr[i] = A(swarmobj.q,swarmobj.p)
                #B_arr[i] = B(swarmobj.q,swarmobj.p)
                
                A_centroid[i] = np.sum(A(),axis=2)/swarmobj.n_beads
                B_centroid[i] = np.sum(B(),axis=2)/swarmobj.n_beads

                #print('A_centroid', A_centroid)
                if(i%1000==0):
                    print( 'memory after',psutil.virtual_memory()[3], psutil.virtual_memory()[2])

        #A_centroid = np.sum(A_arr,axis=3)/MD_System.n_beads
        #B_centroid = np.sum(B_arr,axis=3)/MD_System.n_beads
        
        A_centroid[len(tcf):] = 0.0
        tcf_cr = convolution(A_centroid,B_centroid,0)
        tcf_cr = tcf_cr[:len(tcf),:,:] #Truncating the padded part
        tcf_cr = np.sum(tcf_cr,axis=2) #Summing over the dimension (Dot product)
        tcf+= np.sum(tcf_cr,axis=1)    #Summing over the particles
               # --- This part above is working alright!
        #print('j ended',j)
#        for i in range(len(tcf)):
#            for k in range(len(tcf)):
#                #print(A_arr[i+j]*B_arr[j] + B_arr[i+j]*A_arr[j])
#                tcf[i] += np.sum(A_centroid[i+k]*B_centroid[k] + B_centroid[i+k]*A_centroid[k]) 
    #tcf[i] += np.sum(A_0*B_t)
        thermalize(ACMD,CMD,Matsubara,M,swarmobj,derpotential,deltat,5,rng)
        print('j ending',j,time.time()-start_time)
    
    tcf/=(n_approx*swarmobj.N*len(tcf))#*MD_System.dimension)
    print("RP Simulation Done, TCF length: ", len(tcf))
    print('tcf', tcf)
    #!!! It is unclear if the TCF is to be scaled by dimension. Beware when you make any changes above.
    return tcf

def rearrangearr(M):
#    rearr = np.zeros(M,dtype= int)
#    for i in range(1,int(M/2)+1):
#            rearr[2*i-1]=i
#            rearr[2*i]=-i
#    rearr+=int(M/2) 
    rearr = np.array([0,2,1])  #Change later!!
    rearr = np.array([0])
    return rearr
 
def matsubara_phase_factor(M,swarmobj,rearr):
#    P_tilde = scipy.fftpack.rfft(swarmobj.p,axis=2)/MD_System.n_beads**0.5
#    Q_tilde = scipy.fftpack.rfft(swarmobj.q,axis=2)/MD_System.n_beads**0.5
#    #print(np.shape(P_tilde),np.shape(Q_tilde))
#    P_tilde = P_tilde[:,:,:M]
#    Q_tilde = Q_tilde[:,:,:M]
#    #print(Q_tilde)
#    w_marr = 2*np.arange(-int((M-1)/2),int((M-1)/2)+1)*np.pi\
#            /(MD_System.beta*MD_System.n_beads)
#    
#    w_marr = MD_System.w_arr[:M]
#       
#    Q_tilde = Q_tilde[:,:,rearr]
#    theta = (P_tilde*w_marr)
#    
#    theta*=Q_tilde
#    
#    theta = np.sum(theta,axis=1)  # Dot product.
#    
#    theta = np.sum(theta,axis=1) # Sum over matsubara modes
    #---------------------------------
    w_marr = 2*np.arange(-int((M-1)/2),int((M-1)/2)+1)*np.pi\
            /(swarmobj.beta)
    theta = swarmobj.p*w_marr*swarmobj.q[:,:,::-1]
    theta = np.sum(theta,axis=1) # Summing over the dimensions
    theta = np.sum(theta,axis=1) # Summing over the Matsubara modes
    ## Work out the theta expression once again for higher order Matsubara!
    return theta

#Phase_factor is being calculated correctly!!

def convolution(A,B,axisconv):
    A_tilde = np.fft.rfft(A,axis=axisconv)
    B_tilde = np.fft.rfft(B,axis=axisconv)
    
    AconvB_tilde = np.conj(A_tilde)*B_tilde
    return np.fft.irfft(AconvB_tilde,axis=axisconv)

def corr_function_phaseless_Matsubara(swarmobj,derpotential,A,B,time_corr,n_tp,theta_arr,deltat,rng_ind):
    tcf_thetat = np.zeros((n_tp+1,len(theta_arr))) +0j
    start_time =time.time() 
    n_sample = 10
    rng = np.random.RandomState(rng_ind)
    denom = 0.0j
    rearr = rearrangearr(swarmobj.n_beads)
    for i in range(n_sample):
        thermalize(0,0,1,swarmobj.n_beads,swarmobj,derpotential,deltat,20,rng)
        theta = matsubara_phase_factor(swarmobj.n_beads,swarmobj,rearr)
        exponent = 1j*(swarmobj.beta)*theta
        denom+= np.sum(np.exp(exponent))
        print('denom', np.sum(np.exp(exponent))/swarmobj.N)

    denom/=(swarmobj.N*n_sample)
    print('denom final', denom)
    abs_tcf = np.zeros(n_tp+1)
    for i in range(len(theta_arr)):
        tcf_thetat[:,i] = corr_function_theta(swarmobj,derpotential,theta_arr[i],A,B,time_corr,n_tp,deltat,rng_ind+i)#/denom
        print('i, theta, time',i,theta_arr[i], time.time()-start_time)
        if(i>0):
            dtheta = theta_arr[i] - theta_arr[i-1]
            print('conj',np.conj(tcf_thetat[5,i])*tcf_thetat[5,i])
            print('tcf value', tcf_thetat[:,i])
            abs_tcf+= abs(np.conj(tcf_thetat[:,i])*tcf_thetat[:,i]*dtheta)
            print('done')
    abs_tcf = abs_tcf**0.5

    print('abs_tcf',abs_tcf)
    return abs_tcf

def corr_function_theta(swarmobj,derpotential,thetag,A,B,time_corr,n_tp,deltat,rng_ind):
    rng = np.random.RandomState(rng_ind)
    tcf_tarr = np.arange(0,time_corr+0.0001,time_corr/n_tp)
    tcf = np.zeros_like(tcf_tarr) + 0j
   
    tcf_taux = np.arange(0,2*time_corr+0.0001,time_corr/n_tp)
    
    A_centroid = np.zeros((len(tcf_taux),) + swarmobj.q[:,:,0].shape)
    B_centroid = np.zeros_like(A_centroid)
     
    theta = np.zeros(swarmobj.N) + 0j
 
    Theta_constrained_thermalize(0,0,1,swarmobj.n_beads,swarmobj,thetag,1,100,derpotential,deltat,rng)
    print('kin en',swarmobj.sys_kin()/(len(swarmobj.q)*swarmobj.n_beads))
    n_approx = 1
    rearr = rearrangearr(swarmobj.n_beads)
    
    for j in range(n_approx):
        A_centroid[0] = A()[...,1]
        B_centroid[0] = B()[...,1]
        
        for i in range(1,len(tcf_taux)):
            
            time_evolve(0, 0, 1,swarmobj.n_beads,swarmobj, derpotential, deltat, tcf_taux[i]-tcf_taux[i-1])
            
            A_centroid[i] = A()[...,1]
            B_centroid[i] = B()[...,1]
        
        #print('A,B', np.sum(A_centroid*B_centroid)/(n_approx*swarmobj.N*len(A_centroid)))
        A_centroid[len(tcf):] = 0.0 #Padding for Fourier tranform
  
        tcf_cr = convolution(A_centroid,B_centroid,0)  
        tcf_cr = tcf_cr[:len(tcf),:,:]
        tcf_cr = np.sum(tcf_cr,axis=2) #Summing over the dimensions (Equivalently dot product)
         
        theta = matsubara_phase_factor(swarmobj.n_beads,swarmobj,rearr) 
        exponent = 1j*(swarmobj.beta)*theta
        tcf_cr = tcf_cr*np.exp(exponent)  #-- Line to be added after killing off dimension, be extra careful.
        tcf+= np.sum(tcf_cr,axis=1) #-- Summing up over the particles
    
        #Theta_constrained_thermalize(0,0,1,swarmobj.n_beads,swarmobj,theta,10,20,derpotential,deltat,rng)
    tcf/=(n_approx*swarmobj.N*len(tcf))
    #print('theta, tcf', rng_ind,tcf[20])
    return tcf

def corr_function_phase_dep_Matsubara(swarmobj,derpotential,A,B,time_corr,n_tp,theta_arr,deltat,rng_ind):
    tcf_thetat = np.zeros((n_tp+1,len(theta_arr))) +0j
    start_time =time.time() 
    
    n_sample = 10
    rng = np.random.RandomState(rng_ind)
    if(1): #For computing C and B(L)
        denom = 0.0j
        B_hist = np.zeros_like(theta_arr)
        rearr = rearrangearr(swarmobj.n_beads)
        for i in range(n_sample):
            thermalize(0,0,1,swarmobj.n_beads,swarmobj,derpotential,deltat,20,rng)
            theta = matsubara_phase_factor(swarmobj.n_beads,swarmobj,rearr)
            hist, bin_edges = np.histogram(theta, bins=len(theta_arr), range=(theta_arr[0],theta_arr[-1]),density=True)#len(theta_arr))
            B_hist+=hist/hist.sum()
            print('hist',hist.sum())
            #plt.hist(theta,bins = len(theta_arr), range=(-10.0,10.0),density=True)
            #plt.show()
            exponent = 1j*(swarmobj.beta)*theta
            denom+= np.sum(np.exp(exponent))
            print('denom', np.sum(np.exp(exponent))/swarmobj.N)

        denom/=(swarmobj.N*n_sample)
        B_hist/=n_sample
        print('denom final, B_hist ', denom, B_hist.sum())

    q2avg = 0.0j 
    for i in range(len(theta_arr)):
        print('i, theta, time',i,theta_arr[i], time.time()-start_time)
        print('B_hist', B_hist[i])
        tcf_thetat[:,i] = corr_function_theta(swarmobj,derpotential,theta_arr[i],A,B,time_corr,n_tp,deltat,rng_ind+i)
        q2avg+= (theta_arr[1]-theta_arr[0])*tcf_thetat[0,i]*B_hist[i]/denom

    print('q2avg', q2avg)
    return tcf_thetat

def corr_function_matsubara(ACMD,CMD,Matsubara,M,swarmobj,derpotential,A,B,time_corr,n_tp,deltat,rng_ind):
    rng = np.random.RandomState(rng_ind)
    tcf_tarr = np.arange(0,time_corr+0.0001,time_corr/n_tp)
    tcf = np.zeros_like(tcf_tarr) + 0j
   
    tcf_taux = np.arange(0,2*time_corr+0.0001,time_corr/n_tp)
    A_arr = np.zeros((len(tcf_taux),)+swarmobj.q.shape)
    B_arr = np.zeros_like(A_arr)
    
    A_centroid = np.zeros((len(tcf_taux),) + swarmobj.q[:,:,0].shape)
    B_centroid = np.zeros_like(A_centroid)
     
    theta = np.zeros(swarmobj.N) + 0j
 
    thermalize(ACMD,CMD,1,M,swarmobj,derpotential,deltat,20,rng)
    #Andersen_thermalize(ACMD,CMD,1, M, swarmobj, 10, 20, derpotential, deltat, rng)
    #Theta_constrained_thermalize(ACMD,CMD,1,M,swarmobj,10.0,10,20,derpotential,deltat,rng)
    print('kin en',swarmobj.sys_kin()/(len(swarmobj.q)*swarmobj.n_beads))
    denom=0.0j
    n_approx = 1
    rearr = rearrangearr(M)
    
    for j in range(n_approx):
        A_centroid[0] = A()[...,1]
        B_centroid[0] = B()[...,1]
        
        for i in range(1,len(tcf_taux)):
            
            time_evolve(ACMD, CMD, Matsubara,M,swarmobj, derpotential, deltat, tcf_taux[i]-tcf_taux[i-1])
            
            A_centroid[i] = A()[...,1]
            B_centroid[i] = B()[...,1]
        
        A_centroid[len(tcf):] = 0.0 #Padding for Fourier tranform
  
        tcf_cr = convolution(A_centroid,B_centroid,0)  
        tcf_cr = tcf_cr[:len(tcf),:,:]
        tcf_cr = np.sum(tcf_cr,axis=2) #Summing over the dimensions (Equivalently dot product)
         
        theta = matsubara_phase_factor(M,swarmobj,rearr)
        #print(np.max(theta),np.min(theta))
        exponent = 1j*(swarmobj.beta)*theta
        tcf_cr = tcf_cr*np.exp(exponent)  #-- Line to be added after killing off dimension, be extra careful.
        tcf+= np.sum(tcf_cr,axis=1) #-- Summing up over the particles
        denom+= np.sum(np.exp(exponent))
   
        #hist,binedges = np.histogram(theta,100,(-10,10))
        #print(list(hist),list(binedges))
        print(list(theta))
        #plt.hist(hist,binedges)
        #plt.show()
        
        #thermalize(ACMD,CMD,1,M,swarmobj,derpotential,deltat,5,rng)
        #Andersen_thermalize(ACMD,CMD, Matsubara, M, swarmobj, 10, 20, derpotential, deltat, rng)
        #Theta_constrained_thermalize(ACMD,CMD,1,M,swarmobj,10.0,10,20,derpotential,deltat,rng)
    tcf/=(n_approx*swarmobj.N*len(tcf))
    denom/=(swarmobj.N*n_approx)
    tcf/=denom
    #print('tcf',tcf)
    return tcf

def OTOC_Classical(swarmobj,derpotential,ddpotential,tim,n_tp,deltat,rng_ind):
    rng = np.random.RandomState(rng_ind)
    tcf_tarr = np.arange(0,tim+0.0001,tim/n_tp)
    tcf = np.zeros_like(tcf_tarr)
    start_time = time.time()
    thermalize(0,0,0,1,swarmobj,derpotential,deltat,20,rng)
    print('Time 1', time.time()-start_time)
    n_approx = 1
    print('thermalized', swarmobj.sys_kin()/(swarmobj.N), swarmobj.m, swarmobj.sm, np.sum(swarmobj.p**2))
    
    for j in range(n_approx): 
        traj_arr = Monodromy.ode_instance_classical(swarmobj,tcf_tarr,derpotential,ddpotential)    
        swarmobj.q = Monodromy.q_classical(traj_arr,swarmobj)[len(traj_arr)-1]
        swarmobj.p = Monodromy.p_classical(traj_arr,swarmobj)[len(traj_arr)-1]
        tcf_cr = Monodromy.detmqq_classical(traj_arr,swarmobj)
        tcf+= np.sum(tcf_cr,axis=1) #-- Summing up over the particles
         
        #thermalize(0,0,1,swarmobj,derpotential,deltat,20,rng)
        print(time.time()-start_time)
    
    tcf/=(n_approx*swarmobj.N) 
    return tcf

def OTOC_RP(CMD,Matsubara,M,swarmobj,derpotential,tim,n_tp,deltat,rng_ind):
    rng = np.random.RandomState(rng_ind)
    tarr = np.arange(0,tim+0.0001,tim/100.0)
        
    thermalize(CMD,Matsubara,M,swarmobj,derpotential,deltat,200,rng)
    print('Time 1', time.time()-start_time)
    n_approx = 10
    Quan_arr = np.zeros_like(tarr)
    print('thermalized', swarmobj.sys_kin()/(N*n_beads), swarmobj.m, swarmobj.sm, np.sum(swarmobj.p**2))
       
    for j in range(n_approx):
        temp_q = swarmobj.q.transpose(0,2,1)
        temp_p = swarmobj.p.transpose(0,2,1)
        print('temp_q',np.shape(temp_q))
        traj_arr = Monodromy.ode_instance_RP(temp_q,temp_p,tarr) ##
        if(0): ## This computes the usual 2 point TCF, without symplectic integration 
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
    return Quan_arr

def OTOC_Matsubara(CMD,Matsubara,M,swarmobj,derpotential,tim,n_tp,deltat,rng_ind): ##To be written!
    rng = np.random.RandomState(rng_ind)
    tarr = np.arange(0,tim+0.0001,tim/n_tp)
        
    thermalize(CMD,Matsubara,M,swarmobj,derpotential,deltat,200,rng)
    print('Time 1', time.time()-start_time)
    n_approx = 10
    Quan_arr = np.zeros_like(tarr)
    print('thermalized', swarmobj.sys_kin()/(N*n_beads), swarmobj.m, swarmobj.sm, np.sum(swarmobj.p**2))
       
    for j in range(n_approx):
        temp_q = swarmobj.q.transpose(0,2,1)
        temp_p = swarmobj.p.transpose(0,2,1)
        print('temp_q',np.shape(temp_q))
        traj_arr = Monodromy.ode_instance_RP(temp_q,temp_p,tarr) ##
        if(0): ## This computes the usual 2 point TCF, without symplectic integration 
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
    return Quan_arr

def OTOC_theta(swarmobj,derpotential,ddpotential,thetag,time_corr,n_tp,deltat,rng_ind):
    rng = np.random.RandomState(rng_ind)
    tcf_tarr = np.arange(0,time_corr+0.0001,time_corr/n_tp)
    tcf = np.zeros_like(tcf_tarr) + 0j
   
    Theta_constrained_thermalize(0,0,1,swarmobj.n_beads,swarmobj,thetag,1,200,derpotential,deltat,rng)
    #print('kin en',swarmobj.sys_kin()/(len(swarmobj.q)*swarmobj.n_beads))
    n_approx = 1
    rearr = rearrangearr(swarmobj.n_beads)
    
    for j in range(n_approx):     
        theta = matsubara_phase_factor(swarmobj.n_beads,swarmobj,rearr) 
        traj_arr = Monodromy.ode_instance_Matsubara(swarmobj,tcf_tarr,derpotential,ddpotential)    
        swarmobj.q = Monodromy.q_Matsubara(traj_arr,swarmobj)[len(traj_arr)-1]
        swarmobj.p = Monodromy.p_Matsubara(traj_arr,swarmobj)[len(traj_arr)-1]
        theta = matsubara_phase_factor(swarmobj.n_beads,swarmobj,rearr) 
        #print('thetaf', len(theta))
        exponent = 1j*(swarmobj.beta)*thetag
        tcf_cr = Monodromy.detmqq_Matsubara(traj_arr,swarmobj)*np.exp(exponent)
        #tcf_cr*= np.exp(exponent)  #-- Line to be added after killing off dimension, be extra careful. 
        tcf+= np.sum(tcf_cr,axis=1) #-- Summing up over the particles
         
        #Theta_constrained_thermalize(0,0,1,swarmobj.n_beads,swarmobj,theta,10,20,derpotential,deltat,rng)
    tcf/=(n_approx*swarmobj.N)
    #print('tcf',tcf.shape,tcf)
    return tcf

def compute_phase_histogram(n_sample,swarmobj,derpotential,beta,deltat,theta_arr,rng):
    denom = 0.0j
    B_hist = np.zeros_like(theta_arr)
    rearr = rearrangearr(swarmobj.n_beads)
    for i in range(n_sample):
        thermalize(0,0,1,swarmobj.n_beads,swarmobj,derpotential,deltat,50,rng)
        theta = matsubara_phase_factor(swarmobj.n_beads,swarmobj,rearr)
        hist, bin_edges = np.histogram(theta, bins=len(theta_arr), range=(theta_arr[0],theta_arr[-1]),density=True)#len(theta_arr))
        B_hist+=hist/hist.sum()
        print('hist',hist.sum(),np.sum(B_hist))
        plt.hist(theta,bins = len(theta_arr), range=(theta_arr[0],theta_arr[-1]),density=True)
        plt.show()
        exponent = 1j*(swarmobj.beta)*theta
        denom+= np.sum(np.exp(exponent))
        print('denom', np.sum(np.exp(exponent))/swarmobj.N)

    denom/=(swarmobj.N*n_sample)
    B_hist/=n_sample
    print('denom final, B_hist ', denom, B_hist.sum())

    return B_hist, denom

def OTOC_phase_dep_Matsubara(swarmobj,derpotential,ddpotential,time_corr,n_tp,theta_arr,deltat,rng_ind):
    tcf_thetat = np.zeros((n_tp+1,len(theta_arr))) +0j
    start_time =time.time() 
    
    n_sample = 10
    rng = np.random.RandomState(rng_ind)
    if(0): #For Normalization
        denom = 0.0j
        rearr = rearrangearr(swarmobj.n_beads)
        for i in range(n_sample):
            thermalize(0,0,1,swarmobj.n_beads,swarmobj,derpotential,deltat,20,rng)
            theta = matsubara_phase_factor(swarmobj.n_beads,swarmobj,rearr)
            exponent = 1j*(swarmobj.beta)*theta
            denom+= np.sum(np.exp(exponent))
            print('denom', np.sum(np.exp(exponent))/swarmobj.N)

        denom/=(swarmobj.N*n_sample)
        print('denom final', denom)
        
    for i in range(len(theta_arr)):
        print('i, theta, time',i,theta_arr[i], time.time()-start_time)
        tcf_thetat[:,i] = OTOC_theta(swarmobj,derpotential,ddpotential,theta_arr[i],time_corr,n_tp,deltat,rng_ind+i)
        
    #print('tcf_thetat', tcf_thetat)
    return tcf_thetat


