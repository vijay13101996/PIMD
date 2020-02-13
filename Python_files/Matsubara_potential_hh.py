#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 17:36:45 2019

@author: vgs23
"""

import numpy as np
import itertools
import sympy
from sympy import *
import matplotlib.pyplot as plt


def A(i,x):
    if(i==0):
        return 1.0
    elif(i<0):
        return 2**0.5*cos(2*i*pi*x)
    else:
        return 2**0.5*sin(2*i*pi*x)
    
x = sympy.Symbol('x')

def coeff(indexlist):
    integrand = 1.0
    for ind in indexlist:
        integrand*=A(ind,x)  
    #print(integrand)
    return integrate(integrand,(x,0,1))

def coeff_1(i,j,k,l):
    #print(A(i,x)*A(j,x)*A(k,x)*A(l,x))
    return integrate(A(i,x)*A(j,x)*A(k,x)*A(l,x),(x,0,1))

Q1_tilde = [sympy.Symbol('Q_I_m1'),sympy.Symbol('Q_I_0'),sympy.Symbol('Q_I_1')]
Q2_tilde = [sympy.Symbol('Q_II_m1'),sympy.Symbol('Q_II_0'),sympy.Symbol('Q_II_1')]

def harmpot(coord):
    iterate = list(itertools.product([0,1,2],repeat=2))
    if(coord == 'x'):
        potential = sympy.Symbol('0')
        for combi in iterate:
            potential+= 0.5*Q1_tilde[combi[0]]*Q1_tilde[combi[1]]*coeff([combi[0]-1,combi[1]-1])
        return potential     
    if(coord == 'y'):
        potential = sympy.Symbol('0')
        for combi in iterate:
            potential+= 0.5*Q2_tilde[combi[0]]*Q2_tilde[combi[1]]*coeff([combi[0]-1,combi[1]-1])
        return potential 
    
def cubic():
    iterate = list(itertools.product([0,1,2],repeat=3))
    potential = sympy.Symbol('')
    for combi in iterate:
        potential+= (1/3)*Q2_tilde[combi[0]]*Q2_tilde[combi[1]]*Q2_tilde[combi[2]]*coeff([combi[0]-1,combi[1]-1,combi[2]-1])
    return potential 
    
def cross_term():
    iterate = list(itertools.product([0,1,2],repeat=3))
    potential = sympy.Symbol('')
    for combi in iterate:
        potential+= Q1_tilde[combi[0]]*Q1_tilde[combi[1]]*Q2_tilde[combi[2]]*coeff([combi[0]-1,combi[1]-1,combi[2]-1])
    return potential 

lamda = sympy.Symbol('lamda')
hh = harmpot('x')+ harmpot('y')  + lamda*(cross_term()-cubic())

#print(hh)
#print('x')
#print(diff(hh,Q1_tilde[0]))
#print(diff(hh,Q1_tilde[1]))
#print(diff(hh,Q1_tilde[2]))
#
#print('y')
#print(diff(hh,Q2_tilde[0]))
#print(diff(hh,Q2_tilde[1]))
#print(diff(hh,Q2_tilde[2]))

def pot_hh(Q):
    lamda = 1.0
    Q_I_m1 = Q[...,0,0]
    Q_I_0 = Q[...,0,1]
    Q_I_1 = Q[...,0,2]
    
    Q_II_m1 = Q[...,1,0]
    Q_II_0 = Q[...,1,1]
    Q_II_1 = Q[...,1,2]
    
    return 0.5*Q_II_0**2 + 0.5*Q_II_1**2 + 0.5*Q_II_m1**2 +\
         0.5*Q_I_0**2 + 0.5*Q_I_1**2 + 0.5*Q_I_m1**2 \
         + lamda*(-0.333333333333333*Q_II_0**3 - 1.0*Q_II_0*Q_II_1**2 \
         - 1.0*Q_II_0*Q_II_m1**2 + 1.0*Q_II_0*Q_I_0**2 + \
         1.0*Q_II_0*Q_I_1**2 + 1.0*Q_II_0*Q_I_m1**2 + \
         2.0*Q_II_1*Q_I_0*Q_I_1 + 2.0*Q_II_m1*Q_I_0*Q_I_m1)
    
Q = np.array([[[1,2,3],[2,1,3]],[[1,0,3],[2,2,3]]])

def dpot_hh(Q):
    lamda = 1.0
    Q_I_m1 = Q[...,0,0]
    Q_I_0 = Q[...,0,1]
    Q_I_1 = Q[...,0,2]
    
    Q_II_m1 = Q[...,1,0]
    Q_II_0 = Q[...,1,1]
    Q_II_1 = Q[...,1,2]
    
    dxpot = np.array([1.0*Q_I_m1 + lamda*(2.0*Q_II_0*Q_I_m1 + 2.0*Q_II_m1*Q_I_0),
             1.0*Q_I_0 + lamda*(2.0*Q_II_0*Q_I_0 + 2.0*Q_II_1*Q_I_1 + 2.0*Q_II_m1*Q_I_m1),
             1.0*Q_I_1 + lamda*(2.0*Q_II_0*Q_I_1 + 2.0*Q_II_1*Q_I_0)])
    
    dypot = np.array([1.0*Q_II_m1 + lamda*(-2.0*Q_II_0*Q_II_m1 + 2.0*Q_I_0*Q_I_m1),
             1.0*Q_II_0 + lamda*(-1.0*Q_II_0**2 - 1.0*Q_II_1**2 \
        - 1.0*Q_II_m1**2 + 1.0*Q_I_0**2 + 1.0*Q_I_1**2 + 1.0*Q_I_m1**2),
        1.0*Q_II_1 + lamda*(-2.0*Q_II_0*Q_II_1 + 2.0*Q_I_0*Q_I_1)])
    
    dpot = np.array([dxpot,dypot])
    return np.transpose(dpot,(2,0,1))

#print(np.shape(Q))
#print(np.shape(dpot_hh(Q)))
#def potential(x,y):
#    return x**2/2.0 + y**2/2.0 + x**2*y - y**3/3.0
#
#x = np.linspace(-5,5,200)
#y = np.linspace(-5,5,200)
#X,Y = np.meshgrid(x,y)
#
#plt.contour(X,Y,potential(X,Y))