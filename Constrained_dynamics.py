#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 17:16:32 2020

@author: vgs23
"""

import numpy as np
import scipy
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh,eigs
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
#matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import multiprocessing as mp
import time
import pickle
import MD_System
from scipy import fftpack




def potential(x,y):
    return 0.5*x**2 + 0.5*y**2

def dpotential(Q):
    #print('Q',Q)
    x = Q[0,...] #Change appropriately 
    y = Q[1,...]
    K=0.49
    r_c = 1.89
    
    D0 = 0.18748
    alpha = 1.1605
    r_c = 1.8324
    dpotx = 2.0*D0*alpha*x*(1 - np.exp(-alpha*(-r_c + (x**2 + y**2)**0.5)))*(x**2 + y**2)**(-0.5)*np.exp(-alpha*(-r_c + (x**2 + y**2)**0.5))
    dpoty = 2.0*D0*alpha*y*(1 - np.exp(-alpha*(-r_c + (x**2 + y**2)**0.5)))*(x**2 + y**2)**(-0.5)*np.exp(-alpha*(-r_c + (x**2 + y**2)**0.5))
    
    #dpotx = 1.0*K*x*(-r_c + (x**2 + y**2)**0.5)*(x**2 + y**2)**(-0.5)
    #dpoty = 1.0*K*y*(-r_c + (x**2 + y**2)**0.5)*(x**2 + y**2)**(-0.5)
    
    #dpotx = 1.0*x #+ 2*x*y + 4*x*(x**2 + y**2)
    #dpoty = 1.0*y #+ x**2 - 1.0*y**2 + 4*y*(x**2 + y**2) 
    ret = np.array([dpotx,dpoty])
    #print('ret',ret)
    #ret = np.transpose(np.array([dpotx,dpoty]),(1,0,2))
    #print(ret)
    return ret
    #return Q#np.array([x/(x**2+y**2)**0,y/(x**2+y**2)**0])

def dpotential_ring(Q):
    #print(w_n**2)
    w_n=1.0*len(Q[0,:])  ### CHANGE WHEN BETA CHANGES!!! 
    return m*w_n**2*(2*Q-np.roll(Q,1,axis=1)
                    - np.roll(Q,-1,axis=1))

def sigma(Q,R):
    x_arr = Q[...,0,:]
    y_arr = Q[...,1,:]
    temp = np.sum((1/len(x_arr))*(x_arr**2 + y_arr**2)**0.5)
    #print(len(x_arr))
    #temp = np.sum((1/len(Q[...,0,:]))*(Q[...,0,:]**2 + Q[...,1,:]**2)**0.5)
    return temp-R##x**2+y**2-1

def dsigma(Q):
    r_arr = (1/len(Q[...,0,:]))*(Q[...,0,:]**2 + Q[...,1,:]**2)**0.5
    ret = np.nan_to_num(Q/r_arr)
    return ret#Q/r_arr#np.array([2*x,2*y])

lambda_curr = [0.0]

deltat = 1e-1
m=1.0#1741.1


def fourier_transform(q,p):
    q = scipy.fftpack.rfft(q,axis=1)
    p = scipy.fftpack.rfft(p,axis=1)
  
def inv_fourier_transform(q,p):
    q = scipy.fftpack.irfft(q,axis=1)
    p = scipy.fftpack.irfft(p,axis=1)
    
def RSPA(q,p):
    cosw_arr = np.cos(w_arr*deltat)
    sinw_arr = np.sin(w_arr*deltat)
    
    fourier_transform(q,p)
    
    q[:,0]+=(deltat/m)*p[:,0]
    
    temp = p[:,1:].copy()

    p[:,1:] = cosw_arr*p[:,1:] - m*w_arr*sinw_arr*q[:,1:]
    q[:,1:] = sinw_arr*temp/(m*w_arr) + cosw_arr*q[:,1:]
                            
    inv_fourier_transform(q,p)

def constrained_vv(q,p,lambda_curr):
#    global lambda_curr
#    q0 = q.copy()
#    q+= deltat*p/m - deltat**2*dpotential(q[0],q[1])/(2*m)    
#
#    q_d=q
#    while(abs(sigma(q_d[0],q_d[1])) > 1e-4):
#        
#        q_d = q + lambda_curr*(dsigma(q0[0],q0[1]))/m
#        dlambda = -sigma(q_d[0],q_d[1])/np.dot(dsigma(q_d[0],q_d[1]),dsigma(q0[0],q0[1]))
#        lambda_curr+=dlambda
#    
#    q+= (1/m)*lambda_curr*dsigma(q0[0],q0[1])
#    
#    p+= -(deltat/2)*dpotential(q0[0],q0[1]) + (1/deltat)*lambda_curr*dsigma(q0[0],q0[1])
#    
#    p_d = p - (deltat/2)*dpotential(q[0],q[1])
#    
#    myu = - np.dot(dsigma(q[0],q[1]),p_d)/np.dot(dsigma(q[0],q[1]),dsigma(q[0],q[1]))
#    
#    p = p_d + myu*dsigma(q[0],q[1])
    
    #global lambda_curr
    q0 = q.copy()
    p+= -(deltat/2)*dpotential(q)
    p+= -(deltat/2)*dpotential_ring(q)
    #RATTLE(q,p)
    
    q+= (deltat/m)*p
    #SHAKE(q,p,q0,lambda_curr)
    #RATTLE(q,p)
    
    p+= -(deltat/2)*dpotential(q)
    p+= -(deltat/2)*dpotential_ring(q)
    #RATTLE(q,p) 
    
def SHAKE(q,p,q0,lambda_curr):
    q_d=q
    #print('new',lambda_curr)
    #lambda_curr[0] = 0.0
    while(abs(sigma(q_d,1.0)) > 5e-3):
        
        q_d = q + lambda_curr[0]*(dsigma(q0))/m
        #print('q_d',q_d)
        denom = np.sum(dsigma(q_d)*dsigma(q0))
        if(denom!=0):
            dlambda = -m*sigma(q_d,1.0)/denom 
        lambda_curr[0]+=dlambda
        #print('lamda',dsigma(q_d),dsigma(q0))
        #print('q_d',q_d)
        
        #print('sigma',abs(sigma(q_d,1.0)))
     
    #print('changed to',lambda_curr[0])
        
    q+= (1/m)*lambda_curr[0]*dsigma(q0)
    p+= (1/deltat)*lambda_curr[0]*dsigma(q0)
    #print('lambda_curr',lambda_curr)
    #print('SHAKE','q',q,'p',p,'\n')
    
def RATTLE(q,p):
    #print(np.sum(dsigma(q)*p))
    myu = - np.sum(dsigma(q)*p)/np.sum(dsigma(q)*dsigma(q))#np.dot(dsigma(q[0],q[1]),p)/np.dot(dsigma(q[0],q[1]),dsigma(q[0],q[1]))
    p += myu*dsigma(q)
    

T = 4000#5000*deltat#6.0
t = 0.0

rng = np.random.RandomState(2)
#p = rng.normal(0,1.0,(2,1))#np.array([[0.0],[1]])#np.random.randint(5,size=(2,1)) +1.0#np.array([0.0,1])
beads = 1
#((2,4))#np.array([[1.0],[0.0]])#np.random.randint(5,size=(2,1)) +0.0#np.array([1.0,0.0])
p = rng.normal(0,1.0,(2,beads))
q = np.zeros_like(p)+2.0

MD_System.system_definition(1.0,beads,2,dpotential)
w_arr = MD_System.w_arr
w_arr = w_arr[1:]
print('w_arr',w_arr)


print('q',q)
print('p',p)

count = 0
qcx =[]
qcy =[]
if(1):
    while(abs(t-T)>1e-4):
        constrained_vv(q,p,lambda_curr)
        t+=deltat
        
        if(0):#count%50 ==0):
            print('sigma',sigma(q,1))
            plt.plot(q[0],q[1])
            plt.scatter(q[0],q[1])
            plt.show()
        count+=1
        qcx.append(np.mean(q[0]).copy())
        qcy.append(np.mean(q[1]).copy())
        if(count%10==0):
            print('qcx',t)
            plt.plot(qcx,qcy)
            #plt.scatter(qcy,qcy)
            plt.show()
    print('qnew',q)
    print('pnew',p)
        
