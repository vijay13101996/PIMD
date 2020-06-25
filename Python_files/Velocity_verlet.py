#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 14:34:02 2019

@author: vgs23
"""

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy,copy
from numpy import linalg as LA
from numpy.fft import fft,ifft
from multiprocessing import Process
#from MD_System import gamma,beta,w_n,n_beads,w_arr
from scipy import fftpack
import scipy 
import MD_System
import psutil
#import MD_Simulation

global count 
count =0 


if(0):
    w_arr = MD_System.w_arr[1:]
    w_arr_scaled = MD_System.w_arr_scaled[1:]
    print('w_arr vv',len(w_arr),w_arr)
    print('w_arr_scaled',len(w_arr_scaled))
    cosw_arr = None 
    sinw_arr = None 

ft_rearrange= np.ones(MD_System.n_beads)*(-(2.0/MD_System.n_beads)**0.5)
ft_rearrange[0] = 1/MD_System.n_beads**0.5

print('ft_rearrange',ft_rearrange)

if(MD_System.n_beads%2 == 0):
    ft_rearrange[MD_System.n_beads-1]= 1/MD_System.n_beads**0.5
#~ Find a better way to define the cosines and sines.

quasi_centroid_x=[]
quasi_centroid_y=[]

qcx =[]
qcy =[]
def fourier_transform(swarmobj):
    swarmobj.q = scipy.fftpack.rfft(swarmobj.q,axis=2)
    swarmobj.p = scipy.fftpack.rfft(swarmobj.p,axis=2)
  
def inv_fourier_transform(swarmobj):
    swarmobj.q = scipy.fftpack.irfft(swarmobj.q,axis=2)
    swarmobj.p = scipy.fftpack.irfft(swarmobj.p,axis=2)
 
def normal_mode_force(swarmobj,dpotential):
    dpot = dpotential(swarmobj.q)
    #print(scipy.fftpack.rfft(swarmobj.q,axis=2))
    dpot_tilde = scipy.fftpack.rfft(dpot,axis=2)#/MD_System.n_beads
    return dpot_tilde
 

count=0      
def vv_step(ACMD,CMD,Matsubara,M,swarmobj,dpotential,deltat):
    
    """
    
    This function implements one step of the Velocity Verlet algorithm
    
    Arguments:
        Swarmobj : Swarm class object
        dpotential: Derivative of the system potential, to compute acceleration
        deltat : Time step size
    """
        
#-------------------------------------------------------------------
#    swarmobj.p -= (deltat/2.0)*dpotential(swarmobj.q)
#    swarmobj.q += (deltat/swarmobj.m)*swarmobj.p
#    swarmobj.p -= (deltat/2.0)*dpotential(swarmobj.q)
#------------------------------------------------------------------
    if(0): 
        cosw_arr = np.cos(w_arr*deltat)
        sinw_arr = np.sin(w_arr*deltat)
    
    if(0):  #ACMD
        cosw_arr = np.cos(w_arr_scaled*deltat)
        sinw_arr = np.sin(w_arr_scaled*deltat)   ## BEWARE WHEN YOU MAKE ANY CHANGES!
        #pot_factor = 1/MD_System.n_beads
    
    swarmobj.p -= (deltat/2.0)*dpotential(swarmobj.q)#swarmobj.sys_grad_pot()
    
    if(CMD==1):
        fourier_transform(swarmobj)
        swarmobj.p[:,:,0] = 0.0
        inv_fourier_transform(swarmobj)    

#    swarmobj.p -= (deltat/2.0)*swarmobj.sys_grad_ring_pot_coord()
#    swarmobj.q += (deltat/swarmobj.m)*swarmobj.p
#    swarmobj.p -= (deltat/2.0)*swarmobj.sys_grad_ring_pot_coord()
    
    fourier_transform(swarmobj)
    
    if(1):
        if(CMD!=1):
            swarmobj.q[:,:,0]+=(deltat/swarmobj.m)*swarmobj.p[:,:,0]
             
        
        temp = swarmobj.p[:,:,1:].copy()
        
        if(ACMD==0):
            print('ACMD')
            swarmobj.p[:,:,1:] = MD_System.cosw_arr*swarmobj.p[:,:,1:] \
                               - swarmobj.m*w_arr*MD_System.sinw_arr*swarmobj.q[:,:,1:]
            swarmobj.q[:,:,1:] = MD_System.sinw_arr*temp/(swarmobj.m*w_arr) \
                                + MD_System.cosw_arr*swarmobj.q[:,:,1:]
        if(ACMD==1):
            #print( 'mass factor', swarmobj.m*MD_System.mass_factor*MD_System.w_arr_scaled**2, swarmobj.m*MD_System.w_arr**2)
            #print('coswarr', (MD_System.cosw_arr),np.cos(MD_System.w_arr_scaled))
            swarmobj.p[:,:,1:] = MD_System.cosw_arr*swarmobj.p[:,:,1:] \
                               - swarmobj.m*MD_System.mass_factor*MD_System.w_arr_scaled*MD_System.sinw_arr*swarmobj.q[:,:,1:]
            swarmobj.q[:,:,1:] = MD_System.sinw_arr*temp/(swarmobj.m*MD_System.mass_factor*MD_System.w_arr_scaled) \
                                + MD_System.cosw_arr*swarmobj.q[:,:,1:]
                        
        if(Matsubara==1):
            
            swarmobj.q[:,:,M:]=0.0
            swarmobj.p[:,:,M:]=0.0
            
#    swarmobj.p -= (deltat/2.0)*swarmobj.sys_grad_ring_pot()
#    swarmobj.q += (deltat/swarmobj.m)*swarmobj.p  # Position update step
#    swarmobj.p -= (deltat/2.0)*swarmobj.sys_grad_ring_pot()                   
    
    inv_fourier_transform(swarmobj)
    
    swarmobj.p -= (deltat/2.0)*dpotential(swarmobj.q)#*swarmobj.sys_grad_pot()
   
    if(CMD==1):
        fourier_transform(swarmobj)
        swarmobj.p[:,:,0] = 0.0
        inv_fourier_transform(swarmobj)

    if(0):
        global count
        count+=1
        qcx.append(np.mean(swarmobj.q[0][0]))
        qcy.append(np.mean(swarmobj.q[0][1]))
        
        if(count%1000==0):
            print('centroid')
            plt.scatter(qcx,qcy)
            plt.plot(qcx,qcy)
            plt.show()
            
def RSPA(swarmobj,deltat):
    fourier_transform(swarmobj)
    CMD=0
    if(CMD!=1):
        swarmobj.q[:,:,0]+=(deltat/swarmobj.m)*swarmobj.p[:,:,0]
    
    
    temp = swarmobj.p[:,:,1:].copy()
    
    if(1):
        swarmobj.p[:,:,1:] = cosw_arr*swarmobj.p[:,:,1:] \
                           - swarmobj.m*w_arr*sinw_arr*swarmobj.q[:,:,1:]
        swarmobj.q[:,:,1:] = sinw_arr*temp/(swarmobj.m*w_arr) \
                            + cosw_arr*swarmobj.q[:,:,1:]
    if(0):
        swarmobj.p[:,:,1:] = cosw_arr*swarmobj.p[:,:,1:] \
                           - swarmobj.m*mass_factor*w_arr_scaled*sinw_arr*swarmobj.q[:,:,1:]
        swarmobj.q[:,:,1:] = sinw_arr*temp/(swarmobj.m*mass_factor*w_arr_scaled) \
                            + cosw_arr*swarmobj.q[:,:,1:]
                        
    
    inv_fourier_transform(swarmobj)
  
def adiabatize_p(swarmobj,deltat):
    fourier_transform(swarmobj)
    mass_factor = (w_arr*MD_System.beta_n/(MD_System.omega))**2 #*len(swarmobj.q[0,0,:])
    mass_factor = np.insert(mass_factor,0,1/MD_System.omega**2)
    #swarmobj.p[:,:,1:]+= -(deltat/2)*swarmobj.sys_grad_ring_pot()[:,:,1:]*mass_factor
    swarmobj.p+= -(deltat/2)*swarmobj.sys_grad_ring_pot()*mass_factor
    inv_fourier_transform(swarmobj)
 
def adiabatize_q(swarmobj,deltat):
    fourier_transform(swarmobj)
    mass_factor = (w_arr*MD_System.beta_n/(MD_System.omega))**2 #*len(swarmobj.q[0,0,:])
    mass_factor = np.insert(mass_factor,0,1/MD_System.omega**2)
    #swarmobj.q[:,:,0]+= (deltat/swarmobj.m)*swarmobj.p[:,:,0]
    #swarmobj.q[:,:,1:]+= (deltat/(swarmobj.m*mass_factor))*swarmobj.p[:,:,1:]
    swarmobj.q+= (deltat/(swarmobj.m*mass_factor))*swarmobj.p
    inv_fourier_transform(swarmobj)
    
def sigma(Q,R):
    x_arr = Q[...,0,:]
    y_arr = Q[...,1,:]
    temp = np.sum((1/len(x_arr[0]))*(abs(x_arr**2 + y_arr**2)**0.5), axis=1)
    #print(Q)
    #print('new',x_arr)
    #print('y',y_arr)
    #print('hereh',x_arr[0])#*(x_arr**2 + y_arr**2)**0.5)
    return temp-R##x**2+y**2-1

def dsigma(Q):
    r_arr = (1/len(Q[0,0,:]))*abs((Q[...,0,:]**2 + Q[...,1,:]**2)**0.5)
    r_arr = r_arr[:,np.newaxis]
    r_arr = np.repeat(r_arr,2,axis=1)
    ret = np.nan_to_num(Q/r_arr)
    return ret#Q/r_arr#np.array([2*x,2*y])


def SHAKE(swarmobj,q0,deltat,lambda_curr,R):
    #print('R',R)
    q_d=swarmobj.q.copy()
    count=0 
    while((abs(sigma(q_d,R)) > 5e-4).any()): 
        
        q_d = swarmobj.q + lambda_curr*(dsigma(q0))/swarmobj.m
        denom = np.sum(dsigma(q_d)*dsigma(q0),axis = 1)
        denom = np.sum(denom,axis=1)
        
        if((abs(denom)<1e-3).any()):
            print('start')
            #print(dsigma(q_d),dsigma(q0))
            #print(np.sum(dsigma(q_d)*dsigma(q0),axis = 1))
            print('denom',denom[abs(denom)<1e-3])
            #print('unstable',np.sum(q_d,axis=2))
            #print('unstable init', np.sum(q0,axis=2))
            #print('in-sigma',(abs(sigma(q_d,1.0))), q_d)
            print('end')
            #break  
        dlambda = -swarmobj.m*sigma(q_d,R)/denom 
        dlambda = np.tile(dlambda,(np.shape(swarmobj.q)[1],np.shape(swarmobj.q)[2],1))
        dlambda = np.transpose(dlambda,(2,0,1))
        lambda_curr+=dlambda
        count+=1
        
        if(count>1e6):
            print('count exceeded')#,(abs(sigma(q_d,R)))[abs(sigma(q_d,R))>1e-2]  )
            print(np.where(abs(sigma(q_d,R))>1e-2  ))
            break
        
    #print('sigma',abs(sigma(q_d,1.0))) 
    x_arr = q_d[...,0,:]
    y_arr = q_d[...,1,:]
    temp = np.sum((1/len(x_arr[0]))*(abs(x_arr**2 + y_arr**2))**0.5, axis=1)
    swarmobj.q+= (1/swarmobj.m)*lambda_curr*dsigma(q0)
    swarmobj.p+= (1/deltat)*lambda_curr*dsigma(q0)
    
def RATTLE(q,p):
    num = np.sum(dsigma(q)*p,axis=1)
    num = np.sum(num,axis=1)
    
    denom = np.sum(dsigma(q)*dsigma(q),axis=1)
    denom = np.sum(denom,axis=1)
    
    myu = - num/denom
    
    myu = np.tile(myu,(np.shape(q)[1],np.shape(q)[2],1))
    myu = np.transpose(myu,(2,0,1))
    myu = np.nan_to_num(myu)
    p += myu*dsigma(q)
    
def dpotential_qcentroid(QC_q,swarmobj,dpotential):
    
    r_arr = abs((swarmobj.q[...,0,:]**2 + swarmobj.q[...,1,:]**2)**0.5) #  (1/len(swarmobj.q[0,0,:]))*
    r_arr = r_arr[:,np.newaxis]
    r_arr = np.repeat(r_arr,2,axis=1)
    uv = swarmobj.q/r_arr
     
    dp = np.sum(uv*dpotential(swarmobj.q),axis=1)
    radial_force = (1/len(swarmobj.q[0,0,:]))*np.sum(dp,axis=1)  # add minus, if reqd.
    radial_force = radial_force[:,np.newaxis]
    radial_force = np.repeat(radial_force,2,axis=1)
     
    qc_norm = (QC_q[:,0]**2 + QC_q[:,1]**2)**0.5 # (1/len(QC_q[0,:]))*
    qc_norm = qc_norm[:,np.newaxis]
    qc_norm = np.repeat(qc_norm,2,axis=1)
    ret = (QC_q/qc_norm)*radial_force    # add minus if reqd.
    return ret

def vv_step_constrained(QCMD,swarmobj,QC_q,QC_p,lambda_curr,dpotential,deltat,adiabatic): 
    q0 = swarmobj.q.copy()
    
    if(QCMD!=1):
        QC_p+= -(deltat/2)*dpotential_qcentroid(QC_q,swarmobj,dpotential)
    swarmobj.p+= -(deltat/2)*dpotential(swarmobj.q)
    
    if(adiabatic==1):
        adiabatize_p(swarmobj,deltat)
    else:
        swarmobj.p+= -(deltat/2)*MD_System.dpotential_ring(swarmobj)
    RATTLE(swarmobj.q,swarmobj.p)
    
    if(QCMD!=1):
        QC_q+= (deltat/swarmobj.m)*QC_p
    
    if(adiabatic==1):
        adiabatize_q(swarmobj,deltat)
    else:
        swarmobj.q+= (deltat/swarmobj.m)*swarmobj.p
        
    qc_norm = (QC_q[:,0]**2 + QC_q[:,1]**2)**0.5
    
    SHAKE(swarmobj,q0,deltat,lambda_curr,qc_norm)
    
    if(QCMD!=1):
        QC_p+= -(deltat/2)*dpotential_qcentroid(QC_q,swarmobj,dpotential)
    swarmobj.p+= -(deltat/2)*dpotential(swarmobj.q) 
    
    if(adiabatic==1):
        adiabatize_p(swarmobj,deltat)
    else:
        swarmobj.p+= -(deltat/2)*MD_System.dpotential_ring(swarmobj)
    
    RATTLE(swarmobj.q,swarmobj.p)
    
    #plt.plot(swarmobj.q[0][0],swarmobj.q[0][1])
    #plt.scatter(swarmobj.q[0][0],swarmobj.q[0][1])
    #x_c = np.mean(swarmobj.q[0][0])
    #y_c = np.mean(swarmobj.q[0][1])
    #radius = (QC_q[0][0]**2 + QC_q[0][1]**2)**0.5
    #theta_arr = np.arange(0,2*np.pi+0.1,0.1)
    #quasi_centroid_x.append(QC_q[0][0])
    #quasi_centroid_y.append(QC_q[0][1])
    #plt.plot(quasi_centroid_x,quasi_centroid_y)
    #plt.scatter(quasi_centroid_x,quasi_centroid_y)
    #print(radius,x_c,y_c)
    #plt.plot(radius*np.cos(theta_arr),radius*np.sin(theta_arr))#, fill=False)
    #plt.show()
    
def vv_step_qcmd_adiabatize(swarmobj,QC_q,QC_p,lambda_curr,dpotential,deltat,rng):    
    
    global c1,c2
    mass_factor = (w_arr*MD_System.beta_n/MD_System.omega)**2
    mass_factor = np.insert(mass_factor,0,1/MD_System.omega**2)
    gamma_qc = 1.0
    c1_Q = np.exp(-(deltat/2.0)*gamma_qc)
    c2_Q = (1-c1_Q**2)**0.5 
    
    #rand_gaus = rng.normal(0.0,1.0,np.shape(QC_q))
    #QC_p[:] = c1_Q*QC_p + swarmobj.sm*(1/(MD_System.beta))**0.5*c2_Q*rand_gaus
    
    swarmobj.p = scipy.fftpack.rfft(swarmobj.p,axis=2)*ft_rearrange
    gauss_rand = rng.normal(0.0,1.0,np.shape(swarmobj.q))
    #swarmobj.p[:,:,1:] = c1*swarmobj.p[:,:,1:] + swarmobj.sm*mass_factor**0.5*(1/(1.0))**0.5*c2*gauss_rand[:,:,1:] 
    swarmobj.p =  c1*swarmobj.p + swarmobj.sm*mass_factor**0.5*(1/(MD_System.beta))**0.5*c2*gauss_rand
    swarmobj.p*=(1/ft_rearrange)
    swarmobj.p = scipy.fftpack.irfft(swarmobj.p,axis=2) 
    
    RATTLE(swarmobj.q,swarmobj.p)
    
    vv_step_constrained(0,swarmobj,QC_q,QC_p,lambda_curr,dpotential,deltat,1)
    
    #rand_gaus = rng.normal(0.0,1.0,np.shape(QC_q))
    #QC_p[:] = c1_Q*QC_p + swarmobj.sm*(1/(MD_System.beta))**0.5*c2_Q*rand_gaus
    
    swarmobj.p = scipy.fftpack.rfft(swarmobj.p,axis=2)*ft_rearrange
    gauss_rand = rng.normal(0.0,1.0,np.shape(swarmobj.q))         
    #swarmobj.p[:,:,1:] = c1_Q*swarmobj.p[:,:,1:] + swarmobj.sm*mass_factor**0.5*(1/(1.0))**0.5*c2_Q*gauss_rand[:,:,1:]  
    swarmobj.p =  c1*swarmobj.p + swarmobj.sm*mass_factor**0.5*(1/(MD_System.beta))**0.5*c2*gauss_rand
    swarmobj.p*=(1/ft_rearrange)
    swarmobj.p = scipy.fftpack.irfft(swarmobj.p,axis=2)
    
    RATTLE(swarmobj.q,swarmobj.p)

    #### It is supposed to be MD_System.beta_n not MD_System.beta..... BEWARE!
    
def vv_step_qcmd_thermostat(QCMD,swarmobj,QC_q,QC_p,lambda_curr,dpotential,deltat,rng):
    global c1,c2
    gamma_qc =1.0
    c1_Q = np.exp(-(deltat/2.0)*gamma_qc)
    c2_Q = (1-c1_Q**2)**0.5 
     
    rand_gaus = rng.normal(0.0,1.0,np.shape(QC_q))
    if(QCMD!=1):
        QC_p[:] = c1_Q*QC_p + swarmobj.sm*(1/(MD_System.beta))**0.5*c2_Q*rand_gaus
    
    swarmobj.p = scipy.fftpack.rfft(swarmobj.p,axis=2)*ft_rearrange
    gauss_rand = rng.normal(0.0,1.0,np.shape(swarmobj.q))
    swarmobj.p =  c1*swarmobj.p + swarmobj.sm*(1/(MD_System.beta_n))**0.5*c2*gauss_rand
    swarmobj.p*=(1/ft_rearrange)
    swarmobj.p = scipy.fftpack.irfft(swarmobj.p,axis=2) 
    
    RATTLE(swarmobj.q,swarmobj.p)
    
    vv_step_constrained(QCMD,swarmobj,QC_q,QC_p,lambda_curr,dpotential,deltat,0)
    
    rand_gaus = rng.normal(0.0,1.0,np.shape(QC_q))
    if(QCMD!=1):
        QC_p[:] = c1_Q*QC_p + swarmobj.sm*(1/(MD_System.beta))**0.5*c2_Q*rand_gaus
    
    swarmobj.p = scipy.fftpack.rfft(swarmobj.p,axis=2)*ft_rearrange
    gauss_rand = rng.normal(0.0,1.0,np.shape(swarmobj.q))         
    swarmobj.p =  c1*swarmobj.p + swarmobj.sm*(1/(MD_System.beta_n))**0.5*c2*gauss_rand
    swarmobj.p*=(1/ft_rearrange)
    swarmobj.p = scipy.fftpack.irfft(swarmobj.p,axis=2)
    
    RATTLE(swarmobj.q,swarmobj.p)
    
def vv_step_nc_thermostat(CMD,Matsubara,M,swarmobj,dpotential,deltat,rng):
    global c1,c2
    
    swarmobj.p = scipy.fftpack.rfft(swarmobj.p,axis=2)*ft_rearrange
    gauss_rand = rng.normal(0.0,1.0,np.shape(swarmobj.q))
    swarmobj.p[:,:,1:] = c1*swarmobj.p[:,:,1:] + swarmobj.sm*MD_System.mass_factor**0.5*(1/(MD_System.beta_n))**0.5*c2*gauss_rand[:,:,1:] 
    swarmobj.p*=(1/ft_rearrange)
    swarmobj.p = scipy.fftpack.irfft(swarmobj.p,axis=2) 
        
    #c1,c2 and mass might have to be vectorized at some point. Do take note!
     # When shifting to more than one dimension, the fft routine would be
      # be required, to couple each normal mode with a specific friction coefficient
      
    # Done! Taken care of!: 
       #  Be extra careful in the above step when you vectorize the code.
     
    vv_step(1,CMD,Matsubara,M,swarmobj,dpotential,deltat)
      
    swarmobj.p = scipy.fftpack.rfft(swarmobj.p,axis=2)*ft_rearrange
    gauss_rand = rng.normal(0.0,1.0,np.shape(swarmobj.q))         
    swarmobj.p[:,:,1:] = c1*swarmobj.p[:,:,1:] + swarmobj.sm*MD_System.mass_factor**0.5*(1/(MD_System.beta_n))**0.5*c2*gauss_rand[:,:,1:]  
    swarmobj.p*=(1/ft_rearrange)
    swarmobj.p = scipy.fftpack.irfft(swarmobj.p,axis=2)
            
 
def set_therm_param(deltat,gamma):
    global c1,c2
    print('gamma called',gamma)
    c1 = np.exp(-(deltat/2.0)*gamma)
    c2 = (1-c1**2)**0.5
    
def vv_step_thermostat(ACMD,CMD,Matsubara,M,swarmobj,dpotential,deltat,etherm,rng):
    
    """ 
    
    This function implements one step of the Velocity verlet, along with the 
    Langevin thermostat
    
    Arguments:
        Swarmobj : Swarm class object
        dpotential: Derivative of the system potential, to compute acceleration
        deltat : Time step size
        etherm : This variable is auxiliary, helps take care of correct
                 implementation of the thermostat.
        rng: A randomstate object, used to ensure that the code is seeded
             consistently

    """   
    global c1,c2
    
         # Can be passed as an argument if reqd.

    etherm[0] += swarmobj.sys_kin()
    
    swarmobj.p = scipy.fftpack.rfft(swarmobj.p,axis=2)*ft_rearrange
    gauss_rand = rng.normal(0.0,1.0,np.shape(swarmobj.q))
    swarmobj.p = c1*swarmobj.p + swarmobj.sm*(1/(MD_System.beta_n))**0.5*c2*gauss_rand 
    swarmobj.p*=(1/ft_rearrange)
    swarmobj.p = scipy.fftpack.irfft(swarmobj.p,axis=2) #MD_System.n_beads
    
    etherm[0] -= swarmobj.sys_kin()
    
    #c1,c2 and mass might have to be vectorized at some point. Do take note!
     # When shifting to more than one dimension, the fft routine would be
      # be required, to couple each normal mode with a specific friction coefficient
      
    # Done! Taken care of!: 
       #  Be extra careful in the above step when you vectorize the code.
    
   
    vv_step(ACMD,CMD,Matsubara,M,swarmobj,dpotential,deltat)
   
    etherm[0] += swarmobj.sys_kin()
    
    swarmobj.p = scipy.fftpack.rfft(swarmobj.p,axis=2)*ft_rearrange
    gauss_rand = rng.normal(0.0,1.0,np.shape(swarmobj.q))         
    swarmobj.p = c1*swarmobj.p + swarmobj.sm*(1/(MD_System.beta_n))**0.5*c2*gauss_rand 
    swarmobj.p*=(1/ft_rearrange)
    swarmobj.p = scipy.fftpack.irfft(swarmobj.p,axis=2)#MD_System.n_beads
    
    etherm[0] -= swarmobj.sys_kin()
    
#==================================================STRAY CODE
    
#    global count
#    
#    swarmobj.p -= (deltat/2.0)*swarmobj.sys_grad_pot()
#    
#    
#        
#    swarmobj.p = np.fft.rfft(swarmobj.p,n_beads,axis=2)
#    swarmobj.q = np.fft.rfft(swarmobj.q,n_beads,axis=2)
#    
#    
#    
#    swarmobj.q[:,:,0] += (deltat/swarmobj.m)*swarmobj.p[:,:,0]
#    
#    temp = swarmobj.p[:,:,1:].copy()
#    swarmobj.p[:,:,1:] = np.cos(w_arr*deltat)*swarmobj.p[:,:,1:] \
#                   - swarmobj.m*w_arr*np.sin(w_arr*deltat)*swarmobj.q[:,:,1:]
#    
#    swarmobj.q[:,:,1:] = np.sin(w_arr*deltat)*temp/(swarmobj.m*w_arr) \
#                    + np.cos(w_arr*deltat)*swarmobj.q[:,:,1:]
##    
#    
#    swarmobj.p = np.fft.irfft(swarmobj.p,n_beads,axis=2)
#    swarmobj.q = np.fft.irfft(swarmobj.q,n_beads,axis=2)                
#        
#    swarmobj.p -= (deltat/2.0)*swarmobj.sys_grad_pot()
#    
#    count+=1
# -------------------------------------------------------------------  
    #    print(swarmobj.sys_kin()+swarmobj.sys_pot())
#    cosw_arr = np.cos(w_arr*deltat)
#    sinw_arr = np.sin(w_arr*deltat)
#    
#    swarmobj.p -= (deltat/2.0)*swarmobj.sys_grad_pot()
#    
#    swarmobj.q = scipy.fftpack.rfft(swarmobj.q,axis=2)
#    swarmobj.p = scipy.fftpack.rfft(swarmobj.p,axis=2)
#    
#    swarmobj.q[:,:,0] += (deltat/swarmobj.m)*swarmobj.p[:,:,0]
#    temp = swarmobj.p[:,:,1:].copy()
#    swarmobj.p[:,:,1:] = cosw_arr*swarmobj.p[:,:,1:] \
#                   - swarmobj.m*w_arr*sinw_arr*swarmobj.q[:,:,1:]
#    swarmobj.q[:,:,1:] = sinw_arr*temp/(swarmobj.m*w_arr) \
#                    + cosw_arr*swarmobj.q[:,:,1:]
#                    
#    swarmobj.q = scipy.fftpack.irfft(swarmobj.q,axis=2)
#    swarmobj.p = scipy.fftpack.irfft(swarmobj.p,axis=2)
#    
#    swarmobj.p -= (deltat/2.0)*swarmobj.sys_grad_pot()
#    
#    
#------------------------------------------------------------------- 
    
#Transf = np.zeros((n_beads,n_beads))
#
#for l in range(n_beads):
#    for n in range(int(n_beads/2)+1):
#        if(n==0):
#            Transf[l,n] = 1/n_beads**0.5
#        else:
#            Transf[l,2*n-1] = (2/n_beads)**0.5\
#                    *np.cos(-2*np.pi*n*(l+1)/n_beads)
#            #print(-2*np.pi*n*(l-int(n_beads/2))/n_beads)
#            Transf[l,2*n] = (2/n_beads)**0.5\
#                    *np.sin(2*np.pi*n*(l+1)/n_beads)

#print(Transf)
    
#----------------------------------------------------------------------
    #rearr1 = np.zeros(n_beads,dtype=int)
#
#for i in range(int(n_beads/2)):
#    rearr1[i]=-(2*(i+1))
#    rearr1[i+int(n_beads/2)+1] =2*(i+1)
#    
#def rearrange(Q):
#    return Q[:,:,rearr]
#
#def reverse_rearrange(Q):
#    return Q[:,:,rearr1]

#----------------------------------------------------------------------
    
#arr = np.array(range(-int(n_beads/2),int(n_beads/2+1)))*np.pi/n_beads
#
#w_arr = 2*w_n*np.sin(arr)
#
#rearr = np.zeros(n_beads,dtype= int)
#
#if(n_beads%2!=0):
#    for i in range(1,int(n_beads/2)+1):
#        rearr[2*i-1]=-i
#        rearr[2*i]=i
#    rearr+=int(n_beads/2)
#else:
#    for i in range(1,int(n_beads/2)):
#        rearr[2*i-1]=-i
#        rearr[2*i]=i
#    rearr[len(rearr)-1] = -int(n_beads/2)
#    rearr+=int(n_beads/2)
#    
#
##print(rearr)
#w_arr = w_arr[rearr]
