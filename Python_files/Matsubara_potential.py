#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 13:31:47 2019

@author: vgs23
"""

import numpy as np

def pot_quartic_M3(Q):
    Q_m1 = Q[...,0]
    Q_0  =Q[...,1]
    Q_1  = Q[...,2]
    return 0.25*(1.5*Q_m1**4 + 6.0*Q_m1**2*Q_0**2 + 3.0*Q_m1**2*Q_1**2 + 1.0*Q_0**4 \
            + 6.0*Q_0**2*Q_1**2 + 1.5*Q_1**4)

def dpot_quartic_M3(Q):
    Q_m1 = Q[...,0].T
    Q_0  =Q[...,1].T
    Q_1  = Q[...,2].T
    
    return 0.25*np.array([12.0*Q_0**2*Q_m1 + 6.0*Q_1**2*Q_m1 + 6.0*Q_m1**3,4.0*Q_0**3 + 12.0*Q_0*Q_1**2 + 12.0*Q_0*Q_m1**2, \
            12.0*Q_0**2*Q_1 + 6.0*Q_1**3 + 6.0*Q_1*Q_m1**2]).T
   
def ddpot_quartic_M3(Q):
    Q_m1 = Q[...,0].T
    Q_0  =Q[...,1].T
    Q_1  = Q[...,2].T
    
    ret =  0.25*np.array([[12.0*Q_0**2 + 6.0*Q_1**2 + 18.0*Q_m1**2,   24.0*Q_0*Q_m1,   12.0*Q_1*Q_m1], 
                     [24.0*Q_0*Q_m1,12.0*Q_0**2 + 12.0*Q_1**2 + 12.0*Q_m1**2,24.0*Q_0*Q_1],
                     [12.0*Q_1*Q_m1,24.0*Q_0*Q_1,12.0*Q_0**2 + 18.0*Q_1**2 + 6.0*Q_m1**2]])

    return ret.T

def pot_harmonic_M3(Q):
    Q_m1 = Q[...,0]
    Q_0  =Q[...,1]
    Q_1  = Q[...,2]
    return 0.5*(Q_m1**2 + Q_0**2 + Q_1**2)

def dpot_harmonic_M3(Q):
    Q_m1 = Q[...,0].T
    Q_0  =Q[...,1].T
    Q_1  = Q[...,2].T
    ret = 0.5*np.array([2*Q_m1, 2.0*Q_0, 2.0*Q_1]).T
    return ret

def ddpot_harmonic_M3(Q):
    Q_m1 = Q[...,0].T
    Q_0  =Q[...,1].T
    Q_1  = Q[...,2].T
   
    ones = np.ones_like(Q_m1)
    ret =  0.5*np.array([[2.0*ones,0.0*ones,0.0*ones], [0.0*ones,2.0*ones,0.0*ones], [0.0*ones,0.0*ones,2.0*ones]])
    return ret.T

def pot_harmonic_M1(Q):
    return 0.5*Q**2

def dpot_harmonic_M1(Q):
    return Q 

def ddpot_harmonic_M1(Q):
    return np.ones_like(Q)

def pot_inv_harmonic_M1(Q):
    lamda = 2.0
    g = 1.0/50
    return  -(lamda**2/4.0)*Q**2 + g*Q**4 + lamda**4/(64*g)

def dpot_inv_harmonic_M1(Q):
    lamda = 2.0
    g = 1.0/50
    return -(lamda**2/2.0)*Q + 4*g*Q**3

def ddpot_inv_harmonic_M1(Q):
    lamda = 2.0
    g = 1.0/50
    return -(lamda**2/2.0)*np.ones_like(Q) + 12*g*Q**2

def pot_inv_harmonic_M3(Q):
    Q_m1 = Q[...,0]
    Q_0  =Q[...,1]
    Q_1  = Q[...,2]
    lamda = 2.0
    g = 1.0/50
    return -(lamda**2/4.0)*(Q_m1**2 + Q_0**2 + Q_1**2) + \
            g*(1.5*Q_m1**4 + 6.0*Q_m1**2*Q_0**2 + 3.0*Q_m1**2*Q_1**2 + 1.0*Q_0**4 \
            + 6.0*Q_0**2*Q_1**2 + 1.5*Q_1**4) + lamda**4/(64*g)

def dpot_inv_harmonic_M3(Q):
    Q_m1 = Q[...,0].T
    Q_0  =Q[...,1].T
    Q_1  = Q[...,2].T
    lamda = 2.0
    g = 1.0/50
    ret1 = (-(lamda**2/4.0)*np.array([2*Q_m1, 2.0*Q_0, 2.0*Q_1]))
    ret2 = g*np.array([12.0*Q_0**2*Q_m1 + 6.0*Q_1**2*Q_m1 + 6.0*Q_m1**3,4.0*Q_0**3 + 12.0*Q_0*Q_1**2 + 12.0*Q_0*Q_m1**2, \
                    12.0*Q_0**2*Q_1 + 6.0*Q_1**3 + 6.0*Q_1*Q_m1**2])
    
    return (ret1+ret2).T

def ddpot_inv_harmonic_M3(Q):
    Q_m1 = Q[...,0].T
    Q_0  =Q[...,1].T
    Q_1  = Q[...,2].T
    lamda = 2.0
    g = 1.0/50
    ones = np.ones_like(Q_m1)
    ret1 = -(lamda**2/4.0)*np.array([[2.0*ones,0.0*ones,0.0*ones], [0.0*ones,2.0*ones,0.0*ones], [0.0*ones,0.0*ones,2.0*ones]]) 
    ret2=  g*np.array([[12.0*Q_0**2 + 6.0*Q_1**2 + 18.0*Q_m1**2,   24.0*Q_0*Q_m1,   12.0*Q_1*Q_m1], 
                     [24.0*Q_0*Q_m1,12.0*Q_0**2 + 12.0*Q_1**2 + 12.0*Q_m1**2,24.0*Q_0*Q_1],
                     [12.0*Q_1*Q_m1,24.0*Q_0*Q_1,12.0*Q_0**2 + 18.0*Q_1**2 + 6.0*Q_m1**2]])

    ret = (ret1+ret2).T[:,:,None,:,:]
    #print('ret', ret)
    return ret

def pot_inv_harmonic_M5(Q):
    Q_m2 = Q[...,0]
    Q_m1 = Q[...,1]
    Q_0  = Q[...,2]
    Q_1  = Q[...,3]
    Q_2  = Q[...,4]
    lamda = 2.0
    g = 1.0/50
   
    ret1 = (-lamda**2/4.0)*(1.0*Q_0**2 + 1.0*Q_1**2 + 1.0*Q_2**2 + 1.0*Q_m1**2 + 1.0*Q_m2**2)
    ret2 =  g*(1.0*Q_0**4 + 6.0*Q_0**2*Q_1**2 + 6.0*Q_0**2*Q_2**2 + 6.0*Q_0**2*Q_m1**2 \
           + 6.0*Q_0**2*Q_m2**2 - 8.48528137423857*Q_0*Q_1**2*Q_m2 + 16.9705627484771*Q_0*Q_1*Q_2*Q_m1 \
           + 8.48528137423857*Q_0*Q_m1**2*Q_m2 + 1.5*Q_1**4 + 6.0*Q_1**2*Q_2**2 + 3.0*Q_1**2*Q_m1**2 \
           + 6.0*Q_1**2*Q_m2**2 + 1.5*Q_2**4 + 6.0*Q_2**2*Q_m1**2 + 3.0*Q_2**2*Q_m2**2 \
           + 1.5*Q_m1**4 + 6.0*Q_m1**2*Q_m2**2 + 1.5*Q_m2**4)

    return ret1+ret2

def dpot_inv_harmonic_M5(Q):
    Q_m2 = Q[...,0].T
    Q_m1 = Q[...,1].T
    Q_0  = Q[...,2].T
    Q_1  = Q[...,3].T
    Q_2  = Q[...,4].T
    lamda = 2.0
    g = 1.0/50
   
    ret1 = (-lamda**2/4.0)*np.array([2*Q_m2,2*Q_m1,2*Q_0,2*Q_1,2*Q_2])
    ret2 = g*np.array([12.0*Q_0**2*Q_m2 - 8.48528137423857*Q_0*Q_1**2 + 8.48528137423857*Q_0*Q_m1**2 \
                       + 12.0*Q_1**2*Q_m2 + 6.0*Q_2**2*Q_m2 + 12.0*Q_m1**2*Q_m2 + 6.0*Q_m2**3,
                       12.0*Q_0**2*Q_m1 + 16.9705627484771*Q_0*Q_1*Q_2 + 16.9705627484771*Q_0*Q_m1*Q_m2 \
                       + 6.0*Q_1**2*Q_m1 + 12.0*Q_2**2*Q_m1 + 6.0*Q_m1**3 + 12.0*Q_m1*Q_m2**2,
                       4.0*Q_0**3 + 12.0*Q_0*Q_1**2 + 12.0*Q_0*Q_2**2 + 12.0*Q_0*Q_m1**2 + 12.0*Q_0*Q_m2**2 \
                       - 8.48528137423857*Q_1**2*Q_m2 + 16.9705627484771*Q_1*Q_2*Q_m1 + 8.48528137423857*Q_m1**2*Q_m2,
                       12.0*Q_0**2*Q_1 - 16.9705627484771*Q_0*Q_1*Q_m2 + 16.9705627484771*Q_0*Q_2*Q_m1 \
                       + 6.0*Q_1**3 + 12.0*Q_1*Q_2**2 + 6.0*Q_1*Q_m1**2 + 12.0*Q_1*Q_m2**2,
                       12.0*Q_0**2*Q_2 + 16.9705627484771*Q_0*Q_1*Q_m1 + 12.0*Q_1**2*Q_2 \
                       + 6.0*Q_2**3 + 12.0*Q_2*Q_m1**2 + 6.0*Q_2*Q_m2**2])
        
    return (ret1+ret2).T
    
def ddpot_inv_harmonic_M5(Q):
    Q_m2 = Q[...,0].T
    Q_m1 = Q[...,1].T
    Q_0  = Q[...,2].T
    Q_1  = Q[...,3].T
    Q_2  = Q[...,4].T
    lamda = 2.0
    g = 1.0/50
    ones = np.ones_like(Q_m1)
    
    ret1 = (-lamda**2/4.0)*np.array([[2.0*ones,0.0*ones,0.0*ones,0.0*ones,0.0*ones],
                                    [0.0*ones,2.0*ones,0.0*ones,0.0*ones,0.0*ones],   
                                    [0.0*ones,0.0*ones,2.0*ones,0.0*ones,0.0*ones],
                                    [0.0*ones,0.0*ones,0.0*ones,2.0*ones,0.0*ones],
                                    [0.0*ones,0.0*ones,0.0*ones,0.0*ones,2.0*ones]])

    ret2 = g*np.array([[12.0*Q_0**2 + 12.0*Q_1**2 + 6.0*Q_2**2 + 12.0*Q_m1**2 + 18.0*Q_m2**2,
                       Q_m1*(16.9705627484771*Q_0 + 24.0*Q_m2),
                       (24.0*Q_0*Q_m2 - 8.48528137423857*Q_1**2 + 8.48528137423857*Q_m1**2),
                       Q_1*(-16.9705627484771*Q_0 + 24.0*Q_m2),
                       12.0*Q_2*Q_m2],
                       [Q_m1*(16.9705627484771*Q_0 + 24.0*Q_m2),
                        (12.0*Q_0**2 + 16.9705627484771*Q_0*Q_m2 + 6.0*Q_1**2 + 12.0*Q_2**2 + 18.0*Q_m1**2 + 12.0*Q_m2**2),
                        (24.0*Q_0*Q_m1 + 16.9705627484771*Q_1*Q_2 + 16.9705627484771*Q_m1*Q_m2),
                        (16.9705627484771*Q_0*Q_2 + 12.0*Q_1*Q_m1),
                        (16.9705627484771*Q_0*Q_1 + 24.0*Q_2*Q_m1)],
                       [(24.0*Q_0*Q_m2 - 8.48528137423857*Q_1**2 + 8.48528137423857*Q_m1**2),
                        (24.0*Q_0*Q_m1 + 16.9705627484771*Q_1*Q_2 + 16.9705627484771*Q_m1*Q_m2),
                        (12.0*Q_0**2 + 12.0*Q_1**2 + 12.0*Q_2**2 + 12.0*Q_m1**2 + 12.0*Q_m2**2),
                        (24.0*Q_0*Q_1 - 16.9705627484771*Q_1*Q_m2 + 16.9705627484771*Q_2*Q_m1),
                        (24.0*Q_0*Q_2 + 16.9705627484771*Q_1*Q_m1)],
                       [Q_1*(-16.9705627484771*Q_0 + 24.0*Q_m2),
                        (16.9705627484771*Q_0*Q_2 + 12.0*Q_1*Q_m1),
                        (24.0*Q_0*Q_1 - 16.9705627484771*Q_1*Q_m2 + 16.9705627484771*Q_2*Q_m1),
                        (12.0*Q_0**2 - 16.9705627484771*Q_0*Q_m2 + 18.0*Q_1**2 + 12.0*Q_2**2 + 6.0*Q_m1**2 + 12.0*Q_m2**2),
                        (16.9705627484771*Q_0*Q_m1 + 24.0*Q_1*Q_2)],
                       [12.0*Q_2*Q_m2,
                        (16.9705627484771*Q_0*Q_1 + 24.0*Q_2*Q_m1),
                        (24.0*Q_0*Q_2 + 16.9705627484771*Q_1*Q_m1),
                        (16.9705627484771*Q_0*Q_m1 + 24.0*Q_1*Q_2),
                        (12.0*Q_0**2 + 12.0*Q_1**2 + 18.0*Q_2**2 + 12.0*Q_m1**2 + 6.0*Q_m2**2)]])


    ret = (ret1+ret2).T[:,:,None,:,:]
    #print('ret', ret)
    return ret



