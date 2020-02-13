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

def A(i,x):
    if(i==0):
        return 1.0
    elif(i<0):
        return 2**0.5*cos(2*i*pi*x)
    else:
        return 2**0.5*sin(2*i*pi*x)
    
#print(list(itertools.permutations([1,2,3],3)))
iterate = list(itertools.product([0,1,2],repeat=4))
#print(iterate)
x = sympy.Symbol('x')
y = sin(x)

Q_tilde = [sympy.Symbol('Q_m1'),sympy.Symbol('Q_0'),sympy.Symbol('Q_1')]

ind = iterate[0]

def coeff(i,j,k,l):
    return integrate(A(i,x)*A(j,x)*A(k,x)*A(l,x),(x,0,1))


combi = iterate[0]
potential = sympy.Symbol('')
for combi in iterate:
    potential+= Q_tilde[combi[0]]*Q_tilde[combi[1]]*Q_tilde[combi[2]]*Q_tilde[combi[3]]*coeff(combi[0]-1,combi[1]-1,combi[2]-1,combi[3]-1)

#print(potential) 
#print(diff(potential,Q_tilde[0]))
#print(diff(potential,Q_tilde[1]))
#print(diff(potential,Q_tilde[2]))

#Q=np.array([[[0,2,0],[0,1,2],[2,0,1]]])

def pot(Q):
    Q_m1 = Q[...,0]
    Q_0  =Q[...,1]
    Q_1  = Q[...,2]
    return 1.5*Q_m1**4 + 6.0*Q_m1**2*Q_0**2 + 3.0*Q_m1**2*Q_1**2 + 1.0*Q_0**4 \
            + 6.0*Q_0**2*Q_1**2 + 1.5*Q_1**4

def dpot(Q):
    Q_m1 = Q[...,0].T
    Q_0  =Q[...,1].T
    Q_1  = Q[...,2].T
    #print('ehre',np.shape(Q_0))
    #ret = np.array([12.0*Q_0**2*Q_m1 + 6.0*Q_1**2*Q_m1 + 6.0*Q_m1**3,4.0*Q_0**3 + 12.0*Q_0*Q_1**2 + 12.0*Q_0*Q_m1**2, \
            #12.0*Q_0**2*Q_1 + 6.0*Q_1**3 + 6.0*Q_1*Q_m1**2]).T
    #print(np.shape(ret))
    return 0.25*np.array([12.0*Q_0**2*Q_m1 + 6.0*Q_1**2*Q_m1 + 6.0*Q_m1**3,4.0*Q_0**3 + 12.0*Q_0*Q_1**2 + 12.0*Q_0*Q_m1**2, \
            12.0*Q_0**2*Q_1 + 6.0*Q_1**3 + 6.0*Q_1*Q_m1**2]).T
    
#print(pot(Q))