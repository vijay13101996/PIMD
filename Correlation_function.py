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
from Langevin_thermostat import thermalize, qcmd_thermalize
import psutil


def corr_function_QCMD(swarmobj,QC_q,QC_p,lambda_curr,A,B,time_corr,deltat,dpotential,rng_ind):
    rng = np.random.RandomState(rng_ind)
    tcf_tarr = np.arange(0,time_corr+0.0001,time_corr/100.0)
    tcf = np.zeros_like(tcf_tarr)
    tcf_taux = np.arange(0,2*time_corr+0.0001,time_corr/100.0)
    A_arr = np.zeros((len(tcf_taux),)+QC_q.shape)
    B_arr = np.zeros_like(A_arr)
    n_approx = 5
    #print('q',swarmobj.q)
    #print('p',swarmobj.p)
    qcmd_thermalize(swarmobj,QC_q,QC_p,lambda_curr,dpotential,deltat,1000,rng)
    kin_en = swarmobj.sys_kin()
    print('kin en', kin_en)
    print('kin_en qc',np.sum(QC_p*QC_p)/(2*swarmobj.m))
    qcx=[]#QC_q[:,0]
    qcy=[]#QC_q[:,1]
    #plt.plot(qcx,qcy)
    #plt.scatter(qcx,qcy)
    #plt.show()
    
    for j in range(n_approx):
        
        A_arr[0] = A(QC_q,QC_p)
        B_arr[0] = B(QC_q,QC_p)
        print('j',j)
        for i in range(1,len(tcf_taux)):
            time_evolve_qcmd(swarmobj, QC_q,QC_p,lambda_curr,dpotential, deltat, tcf_taux[i]-tcf_taux[i-1],rng)
            A_arr[i] = A(QC_q,QC_p)
            B_arr[i] = B(QC_q,QC_p)
            #print('QC_p one step',QC_p[0],QC_q[0])
            #qcx.append(QC_q[0][0])
            #qcy.append(QC_q[0][1])
            #plt.plot(swarmobj.q[0][0],swarmobj.q[0][1])
            #plt.scatter(swarmobj.q[0][0],swarmobj.q[0][1])
            #plt.plot(qcx,qcy)
            #plt.scatter(qcx,qcy)
            #plt.show()
        
        A_arr[len(tcf):,:,:] = 0.0
        tcf_cr = convolution(A_arr,B_arr,0)
        tcf_cr = tcf_cr[:len(tcf),:,:] #Truncating the padded part
        tcf_cr = np.sum(tcf_cr,axis=2) #Summing over the dimension (Dot product)
        tcf+= np.sum(tcf_cr,axis=1)
        #print(A_arr[len(tcf_taux)-1],B_arr[len(tcf)-1])
        #plt.plot(tcf_tarr,tcf)#B_arr[:len(tcf_tarr),0,0])
        #plt.plot(tcf_tarr,A_arr[:len(tcf_tarr),0,0])
        #plt.show()
        
#        for i in range(len(tcf)):
#            for k in range(len(tcf)):
#                #print(A_arr[i+j]*B_arr[j] + B_arr[i+j]*A_arr[j])
#                tcf[i] += np.sum(A_arr[i+k]*B_arr[k] + B_arr[i+k]*A_arr[k]) 
        
        print('j finished')
        qcmd_thermalize(swarmobj,QC_q,QC_p,lambda_curr,dpotential,deltat,200.0,rng)
    
    tcf/=(n_approx*swarmobj.N*len(tcf)*MD_System.dimension)
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

 
def corr_function_upgrade(CMD,Matsubara,M,swarmobj,derpotential,A,B,time_corr,deltat,rng_ind):
    rng = np.random.RandomState(rng_ind)
    tcf_tarr = np.arange(0,time_corr+0.0001,time_corr/100.0)
    
    tcf = np.zeros_like(tcf_tarr)
    
    tcf_taux = np.arange(0,2*time_corr+0.0001,time_corr/100.0)
    A_arr = np.zeros((len(tcf_taux),)+swarmobj.q.shape)
    B_arr = np.zeros_like(A_arr)

    thermalize(CMD,Matsubara,M,swarmobj,derpotential,deltat,1000,rng)
    n_approx = 10
    print('kin en',swarmobj.sys_kin())
     #The above line have to be treated on a case to case basis
    for j in range(n_approx):
        
        A_arr[0] = A(swarmobj.q,swarmobj.p)
        B_arr[0] = B(swarmobj.q,swarmobj.p)
        print('j',j)
        if(1):
            for i in range(1,len(tcf_taux)):
                
                time_evolve(CMD,Matsubara,M,swarmobj, derpotential, deltat, tcf_taux[i]-tcf_taux[i-1])
                #print('here')
                #time_evolve_nc(CMD,Matsubara,M,swarmobj, derpotential, deltat, tcf_taux[i]-tcf_taux[i-1],rng)
                A_arr[i] = A(swarmobj.q,swarmobj.p)
                B_arr[i] = B(swarmobj.q,swarmobj.p)
                
        A_centroid = np.sum(A_arr,axis=3)/MD_System.n_beads
        B_centroid = np.sum(B_arr,axis=3)/MD_System.n_beads
        
        A_centroid[len(tcf):] = 0.0
        tcf_cr = convolution(A_centroid,B_centroid,0)
        tcf_cr = tcf_cr[:len(tcf),:,:] #Truncating the padded part
        tcf_cr = np.sum(tcf_cr,axis=2) #Summing over the dimension (Dot product)
        tcf+= np.sum(tcf_cr,axis=1)    #Summing over the particles
               # --- This part above is working alright!
        
#        for i in range(len(tcf)):
#            for k in range(len(tcf)):
#                #print(A_arr[i+j]*B_arr[j] + B_arr[i+j]*A_arr[j])
#                tcf[i] += np.sum(A_centroid[i+k]*B_centroid[k] + B_centroid[i+k]*A_centroid[k]) 
    #tcf[i] += np.sum(A_0*B_t)
        thermalize(CMD,Matsubara,M,swarmobj,derpotential,deltat,200,rng)
    tcf/=(n_approx*swarmobj.N*len(tcf)*MD_System.dimension)

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
            /(MD_System.beta)
    theta = swarmobj.p*w_marr*swarmobj.q[:,:,::-1]
    theta = np.sum(theta,axis=1)
    theta = np.sum(theta,axis=1) 
    ## Work out the theta expression once again for higher order Matsubara!
    return theta

#Phase_factor is being calculated correctly!!

def convolution(A,B,axisconv):
    A_tilde = np.fft.rfft(A,axis=axisconv)
    B_tilde = np.fft.rfft(B,axis=axisconv)
    
    AconvB_tilde = np.conj(A_tilde)*B_tilde
    return np.fft.irfft(AconvB_tilde,axis=axisconv)

def corr_function_matsubara(CMD,Matsubara,M,swarmobj,derpotential,A,B,time_corr,deltat,rng_ind):
    rng = np.random.RandomState(rng_ind)
    tcf_tarr = np.arange(0,time_corr+0.0001,time_corr/100.0)
    tcf = np.zeros_like(tcf_tarr) + 0j
    print('hereh')
    tcf_taux = np.arange(0,2*time_corr+0.0001,time_corr/100.0)#time_corr/10.0)
    A_arr = np.zeros((len(tcf_taux),)+swarmobj.q.shape)
    B_arr = np.zeros_like(A_arr)
    
    A_centroid = np.zeros((len(tcf_taux),swarmobj.N,MD_System.dimension))
    B_centroid = np.zeros_like(A_centroid)
    A_tilde = np.zeros_like(A_centroid)
    B_tilde = np.zeros_like(A_tilde)
    tcf_tilde = np.zeros_like(B_tilde)
    
    tcf_cr1 = np.zeros_like(tcf_tilde)
    tcf_cr2 = np.zeros((len(tcf),swarmobj.N,MD_System.dimension))
    tcf_cr = np.zeros((len(tcf),swarmobj.N))
    theta = np.zeros(swarmobj.N) + 0j
    #~ Find a way to condense the above lines.
 
    thermalize(CMD,0,M,swarmobj,derpotential,deltat,8,rng)
    
    denom=0.0j
    n_approx = 10
    rearr = rearrangearr(M)
    
    for j in range(n_approx):
        A_arr[0] = A(swarmobj.q,swarmobj.p)
        B_arr[0] = B(swarmobj.q,swarmobj.p)
        
        for i in range(1,len(tcf_taux)):
            
            time_evolve(CMD, Matsubara,M,swarmobj, derpotential, deltat, tcf_taux[i]-tcf_taux[i-1])
        
            A_arr[i] = A(swarmobj.q,swarmobj.p)
            B_arr[i] = B(swarmobj.q,swarmobj.p)
         
        A_centroid[:] = A_arr[:,:,:,1]
        B_centroid[:] = B_arr[:,:,:,1]
        
        A_centroid[len(tcf):] = 0.0 #Padding for Fourier tranform
  

        tcf_cr1 = convolution(A_centroid,B_centroid,0)
        
        tcf_cr2 = tcf_cr1[:len(tcf),:,:]
        tcf_cr = np.sum(tcf_cr2,axis=2) #Summing over the dimensions (Equivalently dot product)
        
        
        theta = matsubara_phase_factor(M,swarmobj,rearr)
        exponent = 1j*(MD_System.beta)*theta
        tcf_cr = tcf_cr*np.exp(exponent)  #-- Line to be added after killing off dimension, be extra careful.
        tcf+= np.sum(tcf_cr,axis=1) #-- Summing up over the particles
        
        denom+= np.sum(np.exp(exponent))
        

        thermalize(CMD,0,M,swarmobj,derpotential,deltat,2*MD_System.beta,rng)
        
    tcf/=(n_approx*swarmobj.N*len(tcf)*MD_System.dimension)#*MD_System.n_beads)
    denom/=(swarmobj.N*n_approx)
    tcf/=denom
    #print(denom)
    return tcf
