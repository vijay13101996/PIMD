#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 13:31:47 2019

@author: vgs23
"""

import numpy as np
import itertools
import sympy
from sympy import *

if(0):
    def A(i,x):
        if(i==0):
            return 1.0
        elif(i<0):
            return 2**0.5*cos(2*i*pi*x)
        else:
            return 2**0.5*sin(2*i*pi*x)
        
    iterate_quartic = list(itertools.product([0,1,2],repeat=4))

    x = sympy.Symbol('x')
    y = sin(x)

    Q_tilde = [sympy.Symbol('Q_m1'),sympy.Symbol('Q_0'),sympy.Symbol('Q_1')]

    ind = iterate_quartic[0]

    def coeff_quartic(i,j,k,l):
        return integrate(A(i,x)*A(j,x)*A(k,x)*A(l,x),(x,0,1))

    combi_quartic = iterate_quartic[0]
    potential_quartic = sympy.Symbol('')
    for combi in iterate_quartic:
        potential_quartic+= Q_tilde[combi[0]]*Q_tilde[combi[1]]*Q_tilde[combi[2]]*Q_tilde[combi[3]]*coeff_quartic(combi[0]-1,combi[1]-1,combi[2]-1,combi[3]-1)


    def coeff_harmonic(i,j):
        return integrate(A(i,x)*A(j,x),(x,0,1))

    iterate_harmonic = list(itertools.product([0,1,2],repeat=2))
    potential_harmonic = sympy.Symbol('')
    for combi in iterate_harmonic:
        #print('combi', combi, coeff_harmonic(combi[0]-1,combi[1]-1))
        potential_harmonic+= Q_tilde[combi[0]]*Q_tilde[combi[1]]*coeff_harmonic(combi[0]-1,combi[1]-1)

#print(0.5*potential_harmonic) 
#print(diff(potential,Q_tilde[0]))
#print(diff(potential,Q_tilde[1]))
#print(diff(potential,Q_tilde[2]))
#print(diff(potential,Q_tilde[0],Q_tilde[0]))
#print('\n')
#print(diff(potential,Q_tilde[0],Q_tilde[1]))
#print('\n')
#print(diff(potential,Q_tilde[0],Q_tilde[2]))
#print('\n')
#print(diff(potential,Q_tilde[1],Q_tilde[0]))
#print('\n')
#print(diff(potential,Q_tilde[1],Q_tilde[1]))
#print('\n')
#print(diff(potential,Q_tilde[1],Q_tilde[2]))
#print('\n')
#print(diff(potential,Q_tilde[2],Q_tilde[0]))
#print('\n')
#print(diff(potential,Q_tilde[2],Q_tilde[1]))
#print('\n')
#print(diff(potential,Q_tilde[2],Q_tilde[2]))

def pot_quartic(Q):
    Q_m1 = Q[...,0]
    Q_0  =Q[...,1]
    Q_1  = Q[...,2]
    return 0.25*(1.5*Q_m1**4 + 6.0*Q_m1**2*Q_0**2 + 3.0*Q_m1**2*Q_1**2 + 1.0*Q_0**4 \
            + 6.0*Q_0**2*Q_1**2 + 1.5*Q_1**4)

def dpot_quartic(Q):
    Q_m1 = Q[...,0].T
    Q_0  =Q[...,1].T
    Q_1  = Q[...,2].T
    
    return 0.25*np.array([12.0*Q_0**2*Q_m1 + 6.0*Q_1**2*Q_m1 + 6.0*Q_m1**3,4.0*Q_0**3 + 12.0*Q_0*Q_1**2 + 12.0*Q_0*Q_m1**2, \
            12.0*Q_0**2*Q_1 + 6.0*Q_1**3 + 6.0*Q_1*Q_m1**2]).T
   
def ddpot_quartic(Q):
    Q_m1 = Q[...,0].T
    Q_0  =Q[...,1].T
    Q_1  = Q[...,2].T
    
    ret =  0.25*np.array([[12.0*Q_0**2 + 6.0*Q_1**2 + 18.0*Q_m1**2,   24.0*Q_0*Q_m1,   12.0*Q_1*Q_m1], 
                     [24.0*Q_0*Q_m1,12.0*Q_0**2 + 12.0*Q_1**2 + 12.0*Q_m1**2,24.0*Q_0*Q_1],
                     [12.0*Q_1*Q_m1,24.0*Q_0*Q_1,12.0*Q_0**2 + 18.0*Q_1**2 + 6.0*Q_m1**2]])

    return ret.T

def pot_harmonic(Q):
    Q_m1 = Q[...,0]
    Q_0  =Q[...,1]
    Q_1  = Q[...,2]
    return 0.5*(Q_m1**2 + Q_0**2 + Q_1**2)

def dpot_harmonic(Q):
    Q_m1 = Q[...,0].T
    Q_0  =Q[...,1].T
    Q_1  = Q[...,2].T
    ret = 0.5*np.array([2*Q_m1, 2.0*Q_0, 2.0*Q_1]).T
    return ret

def ddpot_harmonic(Q):
    Q_m1 = Q[...,0].T
    Q_0  =Q[...,1].T
    Q_1  = Q[...,2].T
   
    ones = np.ones_like(Q_m1)
    ret =  0.5*np.array([[2.0*ones,0.0*ones,0.0*ones], [0.0*ones,2.0*ones,0.0*ones], [0.0*ones,0.0*ones,2.0*ones]])
    return ret.T

def pot_inv_harmonic(Q):
    Q_m1 = Q[...,0]
    Q_0  =Q[...,1]
    Q_1  = Q[...,2]
    lamda = 2.0
    g = 1.0/50
    return -(lamda**2/4.0)*(Q_m1**2 + Q_0**2 + Q_1**2) + \
            g*(1.5*Q_m1**4 + 6.0*Q_m1**2*Q_0**2 + 3.0*Q_m1**2*Q_1**2 + 1.0*Q_0**4 \
            + 6.0*Q_0**2*Q_1**2 + 1.5*Q_1**4) + lamda**4/(64*g)

def dpot_inv_harmonic(Q):
    Q_m1 = Q[...,0].T
    Q_0  =Q[...,1].T
    Q_1  = Q[...,2].T
    lamda = 2.0
    g = 1.0/50
    ret1 = (-(lamda**2/4.0)*np.array([2*Q_m1, 2.0*Q_0, 2.0*Q_1]))
    ret2 = g*np.array([12.0*Q_0**2*Q_m1 + 6.0*Q_1**2*Q_m1 + 6.0*Q_m1**3,4.0*Q_0**3 + 12.0*Q_0*Q_1**2 + 12.0*Q_0*Q_m1**2, \
                    12.0*Q_0**2*Q_1 + 6.0*Q_1**3 + 6.0*Q_1*Q_m1**2])
    
    return (ret1+ret2).T

def ddpot_inv_harmonic(Q):
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



