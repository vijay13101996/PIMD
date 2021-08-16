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
from scipy import fftpack
import scipy 
import MD_System
import psutil
import sys

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
 
def B_step_pot(swarmobj,dpotential,deltat): 
    swarmobj.p -= (deltat/2.0)*dpotential(swarmobj.q)

def O_step(swarmobj,Matsubara,rng):
    global c1, c2
    if(Matsubara==1):
        gauss_rand = rng.normal(0.0,1.0,np.shape(swarmobj.q))
        swarmobj.p = c1*swarmobj.p + swarmobj.sm*(1/swarmobj.beta)**0.5*c2*gauss_rand     
    else:
        swarmobj.p = scipy.fftpack.rfft(swarmobj.p,axis=2)*swarmobj.ft_rearrange
        gauss_rand = rng.normal(0.0,1.0,np.shape(swarmobj.q))
        swarmobj.p = c1*swarmobj.p + swarmobj.sm*(1/(swarmobj.beta_n))**0.5*c2*gauss_rand 
        swarmobj.p/=swarmobj.ft_rearrange
        swarmobj.p = scipy.fftpack.irfft(swarmobj.p,axis=2) 
    
def vv_step(ACMD,CMD,Matsubara,M,swarmobj,dpotential,deltat):
    
    """
    
    This function implements one step of the Velocity Verlet algorithm
    
    Arguments:
        Swarmobj : Swarm class object
        dpotential: Derivative of the system potential, to compute acceleration
        deltat : Time step size
    """
        
    if(Matsubara==1):
        swarmobj.p -= (deltat/2.0)*dpotential(swarmobj.q)
        swarmobj.q += (deltat/swarmobj.m)*swarmobj.p
        swarmobj.p -= (deltat/2.0)*dpotential(swarmobj.q)   
     
    else:
        B_step_pot(swarmobj,dpotential,deltat)
        RSPA(ACMD,CMD,swarmobj,deltat) 
        B_step_pot(swarmobj,dpotential,deltat)

           
def RSPA(ACMD,CMD,swarmobj,deltat):
    fourier_transform(swarmobj)
    
    if(CMD!=1):
            swarmobj.q[:,:,0]+=(deltat/swarmobj.m)*swarmobj.p[:,:,0]
              
    temp = swarmobj.p[:,:,1:].copy()
    
    if(ACMD==0):
        swarmobj.p[:,:,1:] = swarmobj.cosw_arr*swarmobj.p[:,:,1:] \
                           - swarmobj.m*swarmobj.w_arr*swarmobj.sinw_arr*swarmobj.q[:,:,1:]
        swarmobj.q[:,:,1:] = swarmobj.sinw_arr*temp/(swarmobj.m*swarmobj.w_arr) \
                            + swarmobj.cosw_arr*swarmobj.q[:,:,1:]
    if(ACMD==1): 
        swarmobj.p[:,:,1:] = swarmobj.cosw_arr*swarmobj.p[:,:,1:] \
                           - swarmobj.m*swarmobj.mass_factor*swarmobj.w_arr_scaled*swarmobj.sinw_arr*swarmobj.q[:,:,1:]
        swarmobj.q[:,:,1:] = swarmobj.sinw_arr*temp/(swarmobj.m*swarmobj.mass_factor*swarmobj.w_arr_scaled) \
                            + swarmobj.cosw_arr*swarmobj.q[:,:,1:]
                    
    inv_fourier_transform(swarmobj)
  
def adiabatize_p(swarmobj,deltat):
    fourier_transform(swarmobj)
    mass_factor = (w_arr*swarmobj.beta_n/(swarmobj.omega))**2 #*len(swarmobj.q[0,0,:])
    mass_factor = np.insert(mass_factor,0,1/swarmobj.omega**2)
    #swarmobj.p[:,:,1:]+= -(deltat/2)*swarmobj.sys_grad_ring_pot()[:,:,1:]*mass_factor
    swarmobj.p+= -(deltat/2)*swarmobj.sys_grad_ring_pot()*mass_factor
    inv_fourier_transform(swarmobj)
 
def adiabatize_q(swarmobj,deltat):
    fourier_transform(swarmobj)
    mass_factor = (w_arr*swarmobj.beta_n/(swarmobj.omega))**2 #*len(swarmobj.q[0,0,:])
    mass_factor = np.insert(mass_factor,0,1/swarmobj.omega**2)
    #swarmobj.q[:,:,0]+= (deltat/swarmobj.m)*swarmobj.p[:,:,0]
    #swarmobj.q[:,:,1:]+= (deltat/(swarmobj.m*mass_factor))*swarmobj.p[:,:,1:]
    swarmobj.q+= (deltat/(swarmobj.m*mass_factor))*swarmobj.p
    inv_fourier_transform(swarmobj)

def U_update(U,Q):
    r_arr = eval_xi1(Q)
    x = Q/r_arr[:,None]
    
    z_prev = U[...,:,2]
    y = np.cross(z_prev,x,axisa=1,axisb=1)
    y/= np.linalg.norm(y,axis=1)[:,None]

    z = np.cross(x,y,axisa=1,axisb=1)
    z/= np.linalg.norm(z,axis=1)[:,None]

    U[...,:,0] = x
    U[...,:,1] = y
    U[...,:,2] = z    # CHECK x,y conventions!
    
def eval_xi1(Q):
    return np.sum(Q**2, axis=1)**0.5

def qcrot_mat_mul(U,Q):
    return np.matmul(U,Q) 

def dxi1(UQ_arr): 
    xi1 = eval_xi1(UQ_arr)
    xi1 = xi1[:,np.newaxis]
    xi1 = np.repeat(xi1,3,axis=1)
    return UQ_arr/xi1

def sigma_xi1(Q,R):
    temp = np.mean(eval_xi1(Q),axis=1)
    return temp - R

def dsigma_xi1(Q,U):
    UQ = qcrot_mat_mul(U,Q) 
    return np.nan_to_num(np.matmul(np.transpose(U,axes=[0,2,1]), dxi1(UQ))) ### Check for consistency!

def dxi2(UQ_arr):
    xi1 = eval_xi1(UQ_arr)
    x_arr = UQ_arr[...,0,:]
    y_arr = UQ_arr[...,1,:]
    z_arr = UQ_arr[...,2,:]
    
    temp = [-x_arr*y_arr/xi1**3,(x_arr**2+z_arr**2)/xi1**3,-y_arr*z_arr/xi1**3]
    ret = np.transpose(temp,axes=[1,0,2])  
    return ret 

def sigma_xi2(Q,U):
    xi1 = eval_xi1(Q)
    UQ = qcrot_mat_mul(U,Q)
    y_arr = UQ[...,1,:]     
    return np.mean(y_arr/xi1,axis=1)

def dsigma_xi2(Q,U):
    UQ = qcrot_mat_mul(U,Q) 
    return np.nan_to_num(np.matmul(np.transpose(U,axes=[0,2,1]), dxi2(UQ))) ### Check for consistency!

def dxi3(UQ_arr):
    xi1 = eval_xi1(UQ_arr)
    x_arr = UQ_arr[...,0,:]
    y_arr = UQ_arr[...,1,:]
    z_arr = UQ_arr[...,2,:]
 
    temp = [-x_arr*z_arr/xi1**3,-y_arr*z_arr/xi1**3,(x_arr**2+y_arr**2)/xi1**3]
    ret = np.transpose(temp,axes=[1,0,2])  
    return ret

def sigma_xi3(Q,U):
    xi1 = eval_xi1(Q)
    UQ = qcrot_mat_mul(U,Q)
    z_arr = UQ[...,1,:]
    return np.mean(z_arr/xi1,axis=1)

def dsigma_xi3(Q,U):
    UQ = qcrot_mat_mul(U,Q) 
    return np.nan_to_num(np.matmul(np.transpose(U,axes=[0,2,1]), dxi3(UQ))) ### Check for consistency!

def BAQC_SHAKE(swarmobj,q0,deltat,U,lambda_curr,R):
    q_d=swarmobj.q.copy()
    count=0
    while((abs(sigma_xi1(q_d,R)) > 5e-4).any() or (abs(sigma_xi2(q_d,U)) > 5e-4).any() or (abs(sigma_xi3(q_d,U)) > 5e-4).any()): 
        
        q_d = swarmobj.q + (lambda_curr[:,0, None, None]*(dsigma_xi1(q0,U)) +  lambda_curr[:,1, None, None]*(dsigma_xi2(q0,U)) + lambda_curr[:,2, None, None]*(dsigma_xi3(q0,U)))/swarmobj.m
        
        A = np.zeros((len(swarmobj.q),3,3))
        B = np.zeros((len(swarmobj.q),3))

        dsigma_arr = [dsigma_xi1,dsigma_xi2, dsigma_xi3]
        for i in range(3):
            for j in range(3):
                A[:,i,j] = np.sum(np.sum(dsigma_arr[i](q_d,U)*dsigma_arr[j](q0,U),axis = 1), axis=1)
        
        B[:,0] = -sigma_xi1(q_d,R)
        B[:,1] = -sigma_xi2(q_d,U)
        B[:,2] = -sigma_xi3(q_d,U)     ## CHECK!
       
        dlambda = np.linalg.solve(A,B)
        if ((np.linalg.cond(A) > 10.0).any()):#sys.float_info.epsilon
            print('shake A', np.linalg.cond(A))
            break
        lambda_curr+=dlambda
        count+=1

        if(count>1e5):
            print('count exceeded')
            print(np.where(abs(sigma(q_d,R))>1e-2  ))
            break
     
    swarmobj.q+= (lambda_curr[:,0,None,None]*(dsigma_xi1(q0,U)) +  lambda_curr[:,1,None,None]*(dsigma_xi2(q0,U)) + lambda_curr[:,2,None,None]*(dsigma_xi3(q0,U)))/swarmobj.m  
    swarmobj.p+= (1/deltat)*(lambda_curr[:,0,None,None]*(dsigma_xi1(q0,U)) +  lambda_curr[:,1,None,None]*(dsigma_xi2(q0,U)) + lambda_curr[:,2,None,None]*(dsigma_xi3(q0,U)))
    ## ALRIGHT!

 
def BAQC_RATTLE(q,p,U):
    #print('rattle')
    A = np.zeros((len(q),3,3))
    B = np.zeros((len(q),3))

    dsigma_arr = [dsigma_xi1,dsigma_xi2, dsigma_xi3]
    for i in range(3):
        for j in range(3):
            A[:,i,j] = np.sum(np.sum(dsigma_arr[i](q,U)*dsigma_arr[j](q,U),axis = 1), axis=1)

    B[:,0] = -np.sum(np.sum(dsigma_arr[0](q,U)*p,axis=1),axis=1)
    B[:,1] = -np.sum(np.sum(dsigma_arr[1](q,U)*p,axis=1),axis=1)
    B[:,2] = -np.sum(np.sum(dsigma_arr[2](q,U)*p,axis=1),axis=1)         ## CHECK!
          
    myu = np.linalg.solve(A,B)
    #print('cond', np.linalg.cond(A),B)
    if ((np.linalg.cond(A) >20.0).any()):#sys.float_info.epsilon
        print('rattle A', np.linalg.cond(A), dsigma_xi1(q,U))
        sys.exit(0)        
    myu = np.nan_to_num(myu)
    p += myu[:,0, None, None]*dsigma_xi1(q,U) + myu[:,1, None, None]*dsigma_xi2(q,U) + myu[:,2, None, None]*dsigma_xi3(q,U)  ## ALRIGHT!
    
def sigma(Q,R):
    x_arr = Q[...,0,:]
    y_arr = Q[...,1,:]
    temp = np.sum((1/len(x_arr[0]))*(abs(x_arr**2 + y_arr**2)**0.5), axis=1) 
    return temp-R

def dsigma(Q):
    r_arr = (1/len(Q[0,0,:]))*abs((Q[...,0,:]**2 + Q[...,1,:]**2)**0.5)
    r_arr = r_arr[:,np.newaxis]
    r_arr = np.repeat(r_arr,2,axis=1)
    ret = np.nan_to_num(Q/r_arr)
    return ret

def SHAKE(swarmobj,q0,deltat,lambda_curr,R):
    q_d=swarmobj.q.copy()
    count=0 
    while((abs(sigma(q_d,R)) > 5e-4).any()): 
        
        q_d = swarmobj.q + lambda_curr*(dsigma(q0))/swarmobj.m
        denom = np.sum(dsigma(q_d)*dsigma(q0),axis = 1)
        denom = np.sum(denom,axis=1)
        
        if((abs(denom)<1e-3).any()):
            print('start')    
            print('denom',denom[abs(denom)<1e-3]) 
            print('end')
            #break  
        dlambda = -swarmobj.m*sigma(q_d,R)/denom 
        dlambda = np.tile(dlambda,(np.shape(swarmobj.q)[1],np.shape(swarmobj.q)[2],1))
        dlambda = np.transpose(dlambda,(2,0,1))
        lambda_curr+=dlambda
        count+=1
        
        if(count>1e6):
            print('count exceeded')
            print(np.where(abs(sigma(q_d,R))>1e-2  ))
            break
     
    if(0):
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

def M_matrix_bead(UQ):
    xi1 = eval_xi1(UQ)
    x_arr = UQ[...,0,:]
    y_arr = UQ[...,1,:]
    z_arr = UQ[...,2,:]

    M_arr = np.zeros((len(UQ),3,3,len(UQ[0,0,:])))  # There will be a problem here!
    print('hereh',np.shape([x_arr/xi1,y_arr/xi1,z_arr/xi1]), np.shape(M_arr[...,0,:]))
    M_arr[...,0,:] = np.transpose([x_arr/xi1,y_arr/xi1,z_arr/xi1],axes = [1,0,2])
    M_arr[...,1,:] = np.transpose([-y_arr*xi1/x_arr, xi1, 0.0*x_arr], axes =[1,0,2])
    M_arr[...,2,:] = np.transpose([-z_arr*xi1/x_arr, 0.0*x_arr, xi1], axes = [1,0,2])
    
    print(np.shape(M_arr))
    return M_arr
    
def M_matrix_BAQC(UQ):
    xi1 = eval_xi1(UQ)
    x_arr = UQ[...,0]
    y_arr = UQ[...,1]
    z_arr = UQ[...,2]

    M_arr = np.zeros((len(UQ),3,3))
    print('hereh',np.shape([x_arr/xi1,y_arr/xi1,z_arr/xi1]), np.shape(M_arr[...,0,:]))
    M_arr[...,0] = np.transpose([x_arr/xi1,y_arr/xi1,z_arr/xi1])
    M_arr[...,1] = np.transpose([-y_arr*xi1/x_arr, xi1, 0.0*x_arr])
    M_arr[...,2] = np.transpose([-z_arr*xi1/x_arr, 0.0*x_arr, xi1])
    
    print(np.shape(M_arr))
    return M_arr

def dpotential_baqcentroid(U,BAQC_q,swarmobj,dpotential):
    bead_fcart = np.matmul(U,dpotential(swarmobj.q))
    UQ_bead = np.matmul(U,swarmobj.q)
    #print('shape0',np.shape(UQ_bead),np.shape(bead_fcart))
    bead_fcart = np.einsum('ijkl,ijl->ijl', M_matrix_bead(UQ_bead),bead_fcart)
    #print('shape1',np.shape(bead_fcart))
    bead_fcart_avg = np.sum(bead_fcart,axis=2)/swarmobj.n_beads
    MU = M_matrix_BAQC(np.einsum('ijk,ij->ij',U,BAQC_q))
    MU_inv = np.linalg.inv(MU)

    ret = np.einsum('ijk,ij->ij',MU_inv,bead_fcart_avg)
    return ret

def vv_step_constrained(QCMD,swarmobj,QC_q,QC_p,lambda_curr,dpotential,deltat,adiabatic): 
    q0 = swarmobj.q.copy()
    
    if(QCMD!=1):
        QC_p+= -(deltat/2)*dpotential_qcentroid(QC_q,swarmobj,dpotential)
    swarmobj.p+= -(deltat/2)*dpotential(swarmobj.q)
    
    if(adiabatic==1):
        adiabatize_p(swarmobj,deltat)
    else:
        swarmobj.p+= -(deltat/2)*swarmobj.dpotential_ring()
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
 
def vv_step_BAQC_constrained(BAQCMD,swarmobj,BAQC_q,BAQC_p,U,lambda_curr,dpotential,deltat,adiabatic): 
    q0 = swarmobj.q.copy()
    
    if(BAQCMD!=1):
        BAQC_p+= -(deltat/2)*dpotential_baqcentroid(BAQC_q,swarmobj,dpotential)
    swarmobj.p+= -(deltat/2)*dpotential(swarmobj.q)
    
    if(adiabatic==1):
        BA_adiabatize_p(swarmobj,deltat)
    else:
        swarmobj.p+= -(deltat/2)*swarmobj.dpotential_ring()
    BAQC_RATTLE(swarmobj.q,swarmobj.p,U)
    
    if(BAQCMD!=1):
        BAQC_q+= (deltat/swarmobj.m)*BAQC_p
    
    if(adiabatic==1):
        BA_adiabatize_q(swarmobj,deltat)
    else:
        swarmobj.q+= (deltat/swarmobj.m)*swarmobj.p
        
    qc_norm = (BAQC_q[:,0]**2 + BAQC_q[:,1]**2 + BAQC_q[:,2]**2)**0.5 ## Use linalg.norm or np.sum
    
    BAQC_SHAKE(swarmobj,q0,deltat,U,lambda_curr,qc_norm)
   
    U_update(U,BAQC_q)

    if(BAQCMD!=1):
        BAQC_p+= -(deltat/2)*dpotential_baqcentroid(BAQC_q,swarmobj,dpotential)
    swarmobj.p+= -(deltat/2)*dpotential(swarmobj.q) 
    
    if(adiabatic==1):
        BA_adiabatize_p(swarmobj,deltat)
    else:
        swarmobj.p+= -(deltat/2)*swarmobj.dpotential_ring()
    
    BAQC_RATTLE(swarmobj.q,swarmobj.p,U)
    
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

def vv_step_baqcmd_thermostat(BAQCMD,swarmobj,BAQC_q,BAQC_p,U,lambda_curr,dpotential,deltat,rng):
    global c1,c2
    gamma_qc =1.0
    c1_Q = np.exp(-(deltat/2.0)*gamma_qc)
    c2_Q = (1-c1_Q**2)**0.5 
     
    rand_gaus = rng.normal(0.0,1.0,np.shape(BAQC_q))
    if(BAQCMD!=1):
        BAQC_p[:] = c1_Q*BAQC_p + swarmobj.sm*(1/(swarmobj.beta))**0.5*c2_Q*rand_gaus
    
    O_step(swarmobj,Matsubara,rng)
    BAQC_RATTLE(swarmobj.q,swarmobj.p,U)
    
    vv_step_BAQC_constrained(BAQCMD,swarmobj,BAQC_q,BAQC_p,U,lambda_curr,dpotential,deltat,0)
    
    rand_gaus = rng.normal(0.0,1.0,np.shape(BAQC_q))
    if(BAQCMD!=1):
        BAQC_p[:] = c1_Q*BAQC_p + swarmobj.sm*(1/(swarmobj.beta))**0.5*c2_Q*rand_gaus
    
    O_step(swarmobj,Matsubara,rng) 
    BAQC_RATTLE(swarmobj.q,swarmobj.p,U)
  
    
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
    O_step(swarmobj,Matsubara,rng)
    etherm[0] -= swarmobj.sys_kin()
    
    vv_step(ACMD,CMD,Matsubara,M,swarmobj,dpotential,deltat)
    
    etherm[0] += swarmobj.sys_kin()
    O_step(swarmobj,Matsubara,rng) 
    etherm[0] -= swarmobj.sys_kin()
    
    #c1,c2 and mass might have to be vectorized at some point. Do take note!
     # When shifting to more than one dimension, the fft routine would be
      # be required, to couple each normal mode with a specific friction coefficient
      
    # Done! Taken care of!: 
       #  Be extra careful in the above step when you vectorize the code.
    
   
