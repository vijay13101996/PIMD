#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 11:17:03 2019

@author: vgs23
"""

import numpy as np
from scipy.integrate import odeint, ode
import sympy
from sympy import *
import scipy.integrate
import matplotlib.pyplot as plt
#from numpy import exp
from mpl_toolkits.mplot3d import Axes3D
import multiprocessing as mp
from functools import partial
import time
import cProfile
start_time = time.time()
#-----------------------------------------------------------------
x = sympy.Symbol('x')
y = sympy.Symbol('y')

V_hh = x**2/2.0 + y**2/2.0 + x**2*y - y**3/3.0 +(x**2+y**2)**2

r = (x**2+y**2)**0.5

r_c = sympy.Symbol('r_c') 
y_c = sympy.Symbol('y_c')
x_c = sympy.Symbol('x_c')

s1 = sympy.Symbol('s1')
s2 = sympy.Symbol('s2')
K = sympy.Symbol('K')
x_co = sympy.Symbol('x_co')
y_co = sympy.Symbol('y_co')
D0 = sympy.Symbol('D0')
alpha = sympy.Symbol('alpha')
V_rh = (K/2)*(r-r_c)**2

V_cb = D0*(1 - exp(-alpha*(r-r_c)))**2
#print(diff(V_cb,x))
#print('new')
#print(diff(V_cb,y))

eps = sympy.Symbol('eps')
A_s = sympy.Symbol('A_s')
alp_s =  sympy.Symbol('alp_s')
R =  sympy.Symbol('R')
R_m =  sympy.Symbol('R_m')
X = R/R_m
beta_s =  sympy.Symbol('beta_s')
D =  sympy.Symbol('D')
C_6 =  sympy.Symbol('C_6')
C_8 =  sympy.Symbol('C_8')
C_10 = sympy.Symbol('C_10')

V_HFD_T1 = eps*(A_s*exp(-alp_s*X+beta_s*X**2) - exp(-(D/X-1)**2)*(C_6/X**6 + C_8/X**8 + C_10/X**10))
V_HFD_T2 = eps*(A_s*exp(-alp_s*X+beta_s*X**2) - (C_6/X**6 + C_8/X**8 + C_10/X**10))

#print('Term 1')
#print(diff(V_HFD_T1,R))
#print('double_derivative')
#print(diff(V_HFD_T1,R,R))

#print('Term 2')
#print(diff(V_HFD_T2,R))
#print('double_derivative')
#print(diff(V_HFD_T2,R,R))

#V_lg = 1/(1+exp(s1*(r-r_c))) + 1/(1+exp(-s2*(y-y_c))) + 1/(1+exp(-s2*(x-x_c)))\
    #+ 1/(1+exp(s2*(y+y_c))) + 1/(1+exp(s2*(x+x_c)))

#s =sympy.Symbol('s')
#L = sympy.Symbol('L')
#r = ((x+L/2)**2 +y**2)**0.5

#r = ((x-L/2)**2 +y**2)**0.5
 
#V_lg_m = K*(1/(exp(s2*(y + y_c)) + 1) + 1/(exp(s2*(x + x_c)) + 1)  + 1/(1 + exp(-s2*(y - y_c))) + 1/(1 + exp(-s2*(x - x_c))))
##for x_co in np.arange(-6,7,2):
##    for y_co in np.arange(-6,7,2):
#V_lg_m = K/(exp(s1*(-r_c + ((x-x_co)**2 + (y-y_co)**2)**0.5)) + 1)
#
#
#               
#             
##V_sb = 1/(1+exp(-s*(r-r_c)))
#V= V_lg_m
b = sympy.Symbol('b')
V_cq = (b/4)*(x**4+y**4) + 0.5*x**2*y**2
V=V_cq

V = (1/4.0)*(x**2+y**2) + b*x**2*y**2

#V = 0.5*(x**2 + y**2) + x*y
#print(V)
#print('\n')
#print(diff(V,x))
#print('\n')
#print(diff(V,y))
#print('\n')
#print(diff(V,x,x))
#print('\n')
#print(diff(V,x,y))
#print('\n')
#print(diff(V,y,x))
#print('\n')
#print(diff(V,y,y))

#---------------------------------------------------------------

pot_code = 'V_cq'

global potential, dxpotential, dypotential, ddpotential1, ddpotential2, ddpotential3, ddpotential4

b = 0.1
def potential(x,y):
    global b
    return (b/4.0)*(x**4+y**4) + 0.5*x**2*y**2

def dpotential_coupled_quartic(q):
    global b
    x = q[...,0]
    y = q[...,1]
    #dpotx = b*x**3 + 1.0*x*y**2#0.5*x + 2*b*x*y**2
    #dpoty = b*y**3 + 1.0*x**2*y#0.5*y + 2*b*x**2*y #
    
    dpotx = 0.5*x + 2*b*x*y**2
    dpoty = 0.5*y + 2*b*x**2*y 
    
    ret = np.array([dpotx,dpoty])
    ret = np.transpose(ret, (1,2,0))
    return ret

def dpotential_cq_vv(Q):
    global b
    x = Q[:,0,...]
    y = Q[:,1,...]
    #dpotx = b*x**3 + 1.0*x*y**2# 0.5*x + 2*b*x*y**2 #
    #dpoty = b*y**3 + 1.0*x**2*y#0.5*y + 2*b*x**2*y #    
   
    dpotx = 0.5*x + 2*b*x*y**2
    dpoty = 0.5*y + 2*b*x**2*y 
    
    ret = np.array([dpotx,dpoty])
    ret = np.transpose(ret, (1,0,2))
    return ret

def ddpotential_coupled_quartic(q):
    global b
    x = q[...,0]
    y = q[...,1]
    oa = np.ones_like(x)
    #ddpot1 = 3*b*x**2 + 1.0*y**2#0.5+ 2*b*y**2#
    #ddpot2 = 2.0*x*y # 4*b*x*y  #
    #ddpot3 = 2.0*x*y # 4*b*x*y  #
    #ddpot4 = 3*b*y**2 + 1.0*x**2# 0.5+ 2*b*x**2#
    
    ddpot1 = 0.5+ 2*b*y**2#
    ddpot2 = 4*b*x*y  #
    ddpot3 = 4*b*x*y  #
    ddpot4 = 0.5+ 2*b*x**2#
    ret = np.array([[ddpot1,ddpot2],[ddpot3,ddpot4]]) 
    ret = np.transpose(ret,(2,3,1,0))
    #print(ret)
    return ret

if(pot_code == 'V_hh'):
    print('here')
    def potential(x,y):
        return x**2/2.0 + y**2/2.0 + x**2*y - y**3/3.0
    
    def dxpotential(x,y):
        return  1.0*x + 2*x*y #+ 4*x*(x**2 + y**2)
     
    def dypotential(x,y):
        return  1.0*y + x**2 - 1.0*y**2  #+ 4*y*(x**2 + y**2)

    def ddpotential1(x,y):
        return  1.0 + 2*y  #+12*x**2 + 4*y**2
    
    def ddpotential2(x,y):
        return 0.0 + 2*x #+2*x*(4*y)

    def ddpotential3(x,y):
        return 0.0 + 2*x #+ 2*x*(4*y)
    
    def ddpotential4(x,y):
        return 1.0 -2.0*y #+ 4*x**2 + 12*y**2

elif(pot_code == 'V_lg'):
    s1 = 4 
    s2 = 2
    r_c = 1.0
    y_c = 5.0
    x_c = 5.0
    def potential(x,y):
        #r = (x**2+y**2)**0.5
        V_lg = 1/(exp(s2*(y + y_c)) + 1) + 1/(exp(s2*(x + x_c)) + 1) + 1/(exp(s1*(-r_c + (x**2 + y**2)**0.5)) + 1) + 1/(1 + exp(-s2*(y - y_c))) + 1/(1 + exp(-s2*(x - x_c)))
        return V_lg
    
    def dxpotential(x,y):
        return  -1.0*s1*x*(x**2 + y**2)**(-0.5)*exp(s1*(-r_c + (x**2 + y**2)**0.5))/(exp(s1*(-r_c + (x**2 + y**2)**0.5)) + 1)**2 - s2*exp(s2*(x + x_c))/(exp(s2*(x + x_c)) + 1)**2 + s2*exp(-s2*(x - x_c))/(1 + exp(-s2*(x - x_c)))**2
     
    def dypotential(x,y):
        return  -1.0*s1*y*(x**2 + y**2)**(-0.5)*exp(s1*(-r_c + (x**2 + y**2)**0.5))/(exp(s1*(-r_c + (x**2 + y**2)**0.5)) + 1)**2 - s2*exp(s2*(y + y_c))/(exp(s2*(y + y_c)) + 1)**2 + s2*exp(-s2*(y - y_c))/(1 + exp(-s2*(y - y_c)))**2

    def ddpotential1(x,y):
        return  -1.0*s1**2*x**2*(x**2 + y**2)**(-1.0)*exp(-s1*(r_c - (x**2 + y**2)**0.5))/(1 + exp(-s1*(r_c - (x**2 + y**2)**0.5)))**2 + 2.0*s1**2*x**2*(x**2 + y**2)**(-1.0)*exp(-2*s1*(r_c - (x**2 + y**2)**0.5))/(1 + exp(-s1*(r_c - (x**2 + y**2)**0.5)))**3 + 1.0*s1*x**2*(x**2 + y**2)**(-1.5)*exp(-s1*(r_c - (x**2 + y**2)**0.5))/(1 + exp(-s1*(r_c - (x**2 + y**2)**0.5)))**2 - 1.0*s1*(x**2 + y**2)**(-0.5)*exp(-s1*(r_c - (x**2 + y**2)**0.5))/(1 + exp(-s1*(r_c - (x**2 + y**2)**0.5)))**2 - s2**2*exp(s2*(x + x_c))/(exp(s2*(x + x_c)) + 1)**2 + 2*s2**2*exp(2*s2*(x + x_c))/(exp(s2*(x + x_c)) + 1)**3 - s2**2*exp(-s2*(x - x_c))/(1 + exp(-s2*(x - x_c)))**2 + 2*s2**2*exp(-2*s2*(x - x_c))/(1 + exp(-s2*(x - x_c)))**3
    
    def ddpotential2(x,y):
        return s1*x*y*(-1.0*s1*(x**2 + y**2)**(-1.0) + 2.0*s1*(x**2 + y**2)**(-1.0)*exp(-s1*(r_c - (x**2 + y**2)**0.5))/(1 + exp(-s1*(r_c - (x**2 + y**2)**0.5))) + 1.0*(x**2 + y**2)**(-1.5))*exp(-s1*(r_c - (x**2 + y**2)**0.5))/(1 + exp(-s1*(r_c - (x**2 + y**2)**0.5)))**2

    def ddpotential3(x,y):
        return s1*x*y*(-1.0*s1*(x**2 + y**2)**(-1.0) + 2.0*s1*(x**2 + y**2)**(-1.0)*exp(-s1*(r_c - (x**2 + y**2)**0.5))/(1 + exp(-s1*(r_c - (x**2 + y**2)**0.5))) + 1.0*(x**2 + y**2)**(-1.5))*exp(-s1*(r_c - (x**2 + y**2)**0.5))/(1 + exp(-s1*(r_c - (x**2 + y**2)**0.5)))**2
    
    def ddpotential4(x,y):
        return -1.0*s1**2*y**2*(x**2 + y**2)**(-1.0)*exp(-s1*(r_c - (x**2 + y**2)**0.5))/(1 + exp(-s1*(r_c - (x**2 + y**2)**0.5)))**2 + 2.0*s1**2*y**2*(x**2 + y**2)**(-1.0)*exp(-2*s1*(r_c - (x**2 + y**2)**0.5))/(1 + exp(-s1*(r_c - (x**2 + y**2)**0.5)))**3 + 1.0*s1*y**2*(x**2 + y**2)**(-1.5)*exp(-s1*(r_c - (x**2 + y**2)**0.5))/(1 + exp(-s1*(r_c - (x**2 + y**2)**0.5)))**2 - 1.0*s1*(x**2 + y**2)**(-0.5)*exp(-s1*(r_c - (x**2 + y**2)**0.5))/(1 + exp(-s1*(r_c - (x**2 + y**2)**0.5)))**2 - s2**2*exp(s2*(y + y_c))/(exp(s2*(y + y_c)) + 1)**2 + 2*s2**2*exp(2*s2*(y + y_c))/(exp(s2*(y + y_c)) + 1)**3 - s2**2*exp(-s2*(y - y_c))/(1 + exp(-s2*(y - y_c)))**2 + 2*s2**2*exp(-2*s2*(y - y_c))/(1 + exp(-s2*(y - y_c)))**3

elif(pot_code == 'V_lg_m'):
    s1 = 4 
    s2 = 2
    r_c = 0.4
    y_c = 10.0
    x_c = 10.0
    K=10.0
    l = -5
    u = 7
    space = 2
    
    #print(np.arange(l,u,space))
    def potential(x,y):
        V_lg_m = K*(1/(exp(s2*(y + y_c)) + 1) + 1/(exp(s2*(x + x_c)) + 1)  + 1/(1 + exp(-s2*(y - y_c))) + 1/(1 + exp(-s2*(x - x_c))))
        for x_co in np.arange(l,u,space):
            for y_co in np.arange(l,u,space):
                #print(l,u)
                V_lg_m+= K/(exp(s1*(-r_c + ((x-x_co)**2 + (y-y_co)**2)**0.5)) + 1)
        
        return V_lg_m
    
    def dxpotential(x,y):
        V_lg_m = K*(-s2*exp(s2*(x + x_c))/(exp(s2*(x + x_c)) + 1)**2 + s2*exp(-s2*(x - x_c))/(1 + exp(-s2*(x - x_c)))**2)
        for x_co in np.arange(l,u,space):
            for y_co in np.arange(l,u,space):
                V_lg_m+= -K*s1*(1.0*x - 1.0*x_co)*((x - x_co)**2 + (y - y_co)**2)**(-0.5)*exp(s1*(-r_c + ((x - x_co)**2 + (y - y_co)**2)**0.5))/(exp(s1*(-r_c + ((x - x_co)**2 + (y - y_co)**2)**0.5)) + 1)**2
         
        return V_lg_m
    
    def dypotential(x,y):
        V_lg_m = K*(-s2*exp(s2*(y + y_c))/(exp(s2*(y + y_c)) + 1)**2 + s2*exp(-s2*(y - y_c))/(1 + exp(-s2*(y - y_c)))**2)
        for x_co in np.arange(l,u,space):
            for y_co in np.arange(l,u,space):
                V_lg_m+= -K*s1*(1.0*y - 1.0*y_co)*((x - x_co)**2 + (y - y_co)**2)**(-0.5)*exp(s1*(-r_c + ((x - x_co)**2 + (y - y_co)**2)**0.5))/(exp(s1*(-r_c + ((x - x_co)**2 + (y - y_co)**2)**0.5)) + 1)**2
        return V_lg_m
        
    def ddpotential1(x,y):
        V_lg_m = -K*s2**2*(exp(s2*(x + x_c))/(exp(s2*(x + x_c)) + 1)**2 - 2*exp(2*s2*(x + x_c))/(exp(s2*(x + x_c)) + 1)**3 + exp(-s2*(x - x_c))/(1 + exp(-s2*(x - x_c)))**2 - 2*exp(-2*s2*(x - x_c))/(1 + exp(-s2*(x - x_c)))**3)
        for x_co in np.arange(l,u,space):
            for y_co in np.arange(l,u,space):
                V_lg_m+= -K*s1*(1.0*s1*(x - x_co)**2*((x - x_co)**2 + (y - y_co)**2)**(-1.0) - 2.0*s1*(x - x_co)**2*((x - x_co)**2 + (y - y_co)**2)**(-1.0)*exp(-s1*(r_c - ((x - x_co)**2 + (y - y_co)**2)**0.5))/(1 + exp(-s1*(r_c - ((x - x_co)**2 + (y - y_co)**2)**0.5))) - 1.0*(x - x_co)**2*((x - x_co)**2 + (y - y_co)**2)**(-1.5) + 1.0*((x - x_co)**2 + (y - y_co)**2)**(-0.5))*exp(-s1*(r_c - ((x - x_co)**2 + (y - y_co)**2)**0.5))/(1 + exp(-s1*(r_c - ((x - x_co)**2 + (y - y_co)**2)**0.5)))**2
        return V_lg_m
        
    def ddpotential2(x,y):
        V_lg_m = 0.0
        for x_co in np.arange(l,u,space):
            for y_co in np.arange(l,u,space):
                V_lg_m+= K*s1*(x - x_co)*(y - y_co)*(-1.0*s1*((x - x_co)**2 + (y - y_co)**2)**(-1.0) + 2.0*s1*((x - x_co)**2 + (y - y_co)**2)**(-1.0)*exp(-s1*(r_c - ((x - x_co)**2 + (y - y_co)**2)**0.5))/(1 + exp(-s1*(r_c - ((x - x_co)**2 + (y - y_co)**2)**0.5))) + 1.0*((x - x_co)**2 + (y - y_co)**2)**(-1.5))*exp(-s1*(r_c - ((x - x_co)**2 + (y - y_co)**2)**0.5))/(1 + exp(-s1*(r_c - ((x - x_co)**2 + (y - y_co)**2)**0.5)))**2
        return V_lg_m
        
    def ddpotential3(x,y):
        V_lg_m = 0.0
        for x_co in np.arange(l,u,space):
            for y_co in np.arange(l,u,space):
                V_lg_m+= K*s1*(x - x_co)*(y - y_co)*(-1.0*s1*((x - x_co)**2 + (y - y_co)**2)**(-1.0) + 2.0*s1*((x - x_co)**2 + (y - y_co)**2)**(-1.0)*exp(-s1*(r_c - ((x - x_co)**2 + (y - y_co)**2)**0.5))/(1 + exp(-s1*(r_c - ((x - x_co)**2 + (y - y_co)**2)**0.5))) + 1.0*((x - x_co)**2 + (y - y_co)**2)**(-1.5))*exp(-s1*(r_c - ((x - x_co)**2 + (y - y_co)**2)**0.5))/(1 + exp(-s1*(r_c - ((x - x_co)**2 + (y - y_co)**2)**0.5)))**2
        return V_lg_m
    
    def ddpotential4(x,y):
        V_lg_m = -K*s2**2*(exp(s2*(y + y_c))/(exp(s2*(y + y_c)) + 1)**2 - 2*exp(2*s2*(y + y_c))/(exp(s2*(y + y_c)) + 1)**3 + exp(-s2*(y - y_c))/(1 + exp(-s2*(y - y_c)))**2 - 2*exp(-2*s2*(y - y_c))/(1 + exp(-s2*(y - y_c)))**3)
        for x_co in np.arange(l,u,space):
            for y_co in np.arange(l,u,space):
                V_lg_m+= -K*s1*(1.0*s1*(y - y_co)**2*((x - x_co)**2 + (y - y_co)**2)**(-1.0) - 2.0*s1*(y - y_co)**2*((x - x_co)**2 + (y - y_co)**2)**(-1.0)*exp(-s1*(r_c - ((x - x_co)**2 + (y - y_co)**2)**0.5))/(1 + exp(-s1*(r_c - ((x - x_co)**2 + (y - y_co)**2)**0.5))) - 1.0*(y - y_co)**2*((x - x_co)**2 + (y - y_co)**2)**(-1.5) + 1.0*((x - x_co)**2 + (y - y_co)**2)**(-0.5))*exp(-s1*(r_c - ((x - x_co)**2 + (y - y_co)**2)**0.5))/(1 + exp(-s1*(r_c - ((x - x_co)**2 + (y - y_co)**2)**0.5)))**2
        return V_lg_m
    
elif(pot_code == 'V_sb'):
    
    L = 10.0  
    s=2.0
    r_c =5.0
    K = 10.0
    
    def potential(x,y):
        if(abs(x)<L/2):
            return K*(1/(exp(s*(r_c + y)) + 1) + 1/(1 + exp(-s*(-r_c + y))))
        else:
            if(x<0):
                r = ((x+L/2)**2 +y**2)**0.5
                return K*(1/(1 + exp(-s*(-r_c + (y**2 + (L/2 + x)**2)**0.5))))
            else:
                r = ((x-L/2)**2 +y**2)**0.5
                return K*(1/(1 + exp(-s*(-r_c + (y**2 + (-L/2 + x)**2)**0.5))))
            
    def dxpotential(x,y):
        #print('herhe')
        if(abs(x)<L/2):
            return 0.0
        
        else:
            if(x<0):
                return K*(s*(0.5*L + 1.0*x)*(y**2 + (L/2 + x)**2)**(-0.5)*exp(-s*(-r_c + (y**2 + (L/2 + x)**2)**0.5))/(1 + exp(-s*(-r_c + (y**2 + (L/2 + x)**2)**0.5)))**2)
    
            else:
                r = ((x-L/2)**2 +y**2)**0.5
                return K*(s*(-0.5*L + 1.0*x)*(y**2 + (-L/2 + x)**2)**(-0.5)*exp(-s*(-r_c + (y**2 + (-L/2 + x)**2)**0.5))/(1 + exp(-s*(-r_c + (y**2 + (-L/2 + x)**2)**0.5)))**2)
            
    def dypotential(x,y):
        if(abs(x)<L/2):
            return K*((-s*exp(s*(r_c + y))/(exp(s*(r_c + y)) + 1)**2 + s*exp(-s*(-r_c + y))/(1 + exp(-s*(-r_c + y)))**2))
        
        else:
            if(x<0):
                return K*(1.0*s*y*(y**2 + (L/2 + x)**2)**(-0.5)*exp(-s*(-r_c + (y**2 + (L/2 + x)**2)**0.5))/(1 + exp(-s*(-r_c + (y**2 + (L/2 + x)**2)**0.5)))**2)
            else:
                r = ((x-L/2)**2 +y**2)**0.5
                return K*(1.0*s*y*(y**2 + (-L/2 + x)**2)**(-0.5)*exp(-s*(-r_c + (y**2 + (-L/2 + x)**2)**0.5))/(1 + exp(-s*(-r_c + (y**2 + (-L/2 + x)**2)**0.5)))**2)
            
    def ddpotential1(x,y):
        if(abs(x)<L/2):
            return 0.0
        
        else:
            if(x<0):
                return K*(s*(-s*(0.5*L + 1.0*x)**2*(y**2 + (L + 2*x)**2/4)**(-1.0) + 2*s*(0.5*L + 1.0*x)**2*(y**2 + (L + 2*x)**2/4)**(-1.0)*exp(s*(r_c - (y**2 + (L + 2*x)**2/4)**0.5))/(exp(s*(r_c - (y**2 + (L + 2*x)**2/4)**0.5)) + 1) - (0.5*L + 1.0*x)**2*(y**2 + (L + 2*x)**2/4)**(-1.5) + 1.0*(y**2 + (L + 2*x)**2/4)**(-0.5))*exp(s*(r_c - (y**2 + (L + 2*x)**2/4)**0.5))/(exp(s*(r_c - (y**2 + (L + 2*x)**2/4)**0.5)) + 1)**2) 
            else:
                r = ((x-L/2)**2 +y**2)**0.5
                return K*(s*(-s*(0.5*L - 1.0*x)**2*(y**2 + (L - 2*x)**2/4)**(-1.0) + 2*s*(0.5*L - 1.0*x)**2*(y**2 + (L - 2*x)**2/4)**(-1.0)*exp(s*(r_c - (y**2 + (L - 2*x)**2/4)**0.5))/(exp(s*(r_c - (y**2 + (L - 2*x)**2/4)**0.5)) + 1) - (0.5*L - 1.0*x)**2*(y**2 + (L - 2*x)**2/4)**(-1.5) + 1.0*(y**2 + (L - 2*x)**2/4)**(-0.5))*exp(s*(r_c - (y**2 + (L - 2*x)**2/4)**0.5))/(exp(s*(r_c - (y**2 + (L - 2*x)**2/4)**0.5)) + 1)**2)
            
    def ddpotential2(x,y):
        if(abs(x)<L/2):
            return 0.0
        
        else:
            if(x<0):
                return K*(s*y*(0.5*L + 1.0*x)*(-1.0*s*(y**2 + (L + 2*x)**2/4)**(-1.0) + 2.0*s*(y**2 + (L + 2*x)**2/4)**(-1.0)*exp(s*(r_c - (y**2 + (L + 2*x)**2/4)**0.5))/(exp(s*(r_c - (y**2 + (L + 2*x)**2/4)**0.5)) + 1) - 1.0*(y**2 + (L + 2*x)**2/4)**(-1.5))*exp(s*(r_c - (y**2 + (L + 2*x)**2/4)**0.5))/(exp(s*(r_c - (y**2 + (L + 2*x)**2/4)**0.5)) + 1)**2)
            else:
                r = ((x-L/2)**2 +y**2)**0.5
                return K*(s*y*(0.5*L - 1.0*x)*(1.0*s*(y**2 + (L - 2*x)**2/4)**(-1.0) - 2.0*s*(y**2 + (L - 2*x)**2/4)**(-1.0)*exp(s*(r_c - (y**2 + (L - 2*x)**2/4)**0.5))/(exp(s*(r_c - (y**2 + (L - 2*x)**2/4)**0.5)) + 1) + 1.0*(y**2 + (L - 2*x)**2/4)**(-1.5))*exp(s*(r_c - (y**2 + (L - 2*x)**2/4)**0.5))/(exp(s*(r_c - (y**2 + (L - 2*x)**2/4)**0.5)) + 1)**2)
            
    def ddpotential3(x,y):
        if(abs(x)<L/2):
            return 0.0
        
        else:
            if(x<0):
                return K*(s*y*(0.5*L + 1.0*x)*(-1.0*s*(y**2 + (L + 2*x)**2/4)**(-1.0) + 2.0*s*(y**2 + (L + 2*x)**2/4)**(-1.0)*exp(s*(r_c - (y**2 + (L + 2*x)**2/4)**0.5))/(exp(s*(r_c - (y**2 + (L + 2*x)**2/4)**0.5)) + 1) - 1.0*(y**2 + (L + 2*x)**2/4)**(-1.5))*exp(s*(r_c - (y**2 + (L + 2*x)**2/4)**0.5))/(exp(s*(r_c - (y**2 + (L + 2*x)**2/4)**0.5)) + 1)**2)
            else:
                r = ((x-L/2)**2 +y**2)**0.5
                return K*(s*y*(0.5*L - 1.0*x)*(1.0*s*(y**2 + (L - 2*x)**2/4)**(-1.0) - 2.0*s*(y**2 + (L - 2*x)**2/4)**(-1.0)*exp(s*(r_c - (y**2 + (L - 2*x)**2/4)**0.5))/(exp(s*(r_c - (y**2 + (L - 2*x)**2/4)**0.5)) + 1) + 1.0*(y**2 + (L - 2*x)**2/4)**(-1.5))*exp(s*(r_c - (y**2 + (L - 2*x)**2/4)**0.5))/(exp(s*(r_c - (y**2 + (L - 2*x)**2/4)**0.5)) + 1)**2)
            
    def ddpotential4(x,y):
        if(abs(x)<L/2):
            return K*(s**2*(-exp(s*(r_c + y))/(exp(s*(r_c + y)) + 1)**2 + 2*exp(2*s*(r_c + y))/(exp(s*(r_c + y)) + 1)**3 - exp(s*(r_c - y))/(exp(s*(r_c - y)) + 1)**2 + 2*exp(2*s*(r_c - y))/(exp(s*(r_c - y)) + 1)**3))

        else:
            if(x<0):
                return K*(s*(-1.0*s*y**2*(y**2 + (L + 2*x)**2/4)**(-1.0) + 2.0*s*y**2*(y**2 + (L + 2*x)**2/4)**(-1.0)*exp(s*(r_c - (y**2 + (L + 2*x)**2/4)**0.5))/(exp(s*(r_c - (y**2 + (L + 2*x)**2/4)**0.5)) + 1) - 1.0*y**2*(y**2 + (L + 2*x)**2/4)**(-1.5) + 1.0*(y**2 + (L + 2*x)**2/4)**(-0.5))*exp(s*(r_c - (y**2 + (L + 2*x)**2/4)**0.5))/(exp(s*(r_c - (y**2 + (L + 2*x)**2/4)**0.5)) + 1)**2)
            else:
                r = ((x-L/2)**2 +y**2)**0.5
                return K*(s*(-1.0*s*y**2*(y**2 + (L - 2*x)**2/4)**(-1.0) + 2.0*s*y**2*(y**2 + (L - 2*x)**2/4)**(-1.0)*exp(s*(r_c - (y**2 + (L - 2*x)**2/4)**0.5))/(exp(s*(r_c - (y**2 + (L - 2*x)**2/4)**0.5)) + 1) - 1.0*y**2*(y**2 + (L - 2*x)**2/4)**(-1.5) + 1.0*(y**2 + (L - 2*x)**2/4)**(-0.5))*exp(s*(r_c - (y**2 + (L - 2*x)**2/4)**0.5))/(exp(s*(r_c - (y**2 + (L - 2*x)**2/4)**0.5)) + 1)**2)
            
def f(t,r,dxpotential,dypotential,ddpotential1,ddpotential2,ddpotential3,ddpotential4):
    
    
    x,px,y,py,mpp1,mpp2,mpp3,mpp4,mpq1,mpq2,mpq3,mpq4,mqp1,mqp2,mqp3,mqp4,mqq1,mqq2,mqq3,mqq4=  r

    ddV1 = ddpotential1(x,y)
    ddV2 = ddpotential2(x,y)
    ddV3 = ddpotential3(x,y)
    ddV4 = ddpotential4(x,y)
    
    drdt = np.array([px,-dxpotential(x,y), py, -dypotential(x,y),-(ddV1*mqp1 + ddV2*mqp3),-(ddV1*mqp2 + ddV2*mqp4),
                 -(ddV3*mqp1 + ddV4*mqp3),-(ddV3*mqp2 + ddV4*mqp4),
                 -(ddV1*mqq1 + ddV2*mqq3),-(ddV1*mqq2 + ddV2*mqq4),
                 -(ddV3*mqq1 + ddV4*mqq3),-(ddV3*mqq2 + ddV4*mqq4),
                   mpp1,mpp2,mpp3,mpp4, mpq1,mpq2,mpq3,mpq4])
    
    return drdt
tarr =[]
marr= []
def solout(t,r):
    x,px,y,py, mpp1,mpp2,mpp3,mpp4,mpq1,mpq2,mpq3,mpq4,mqp1,mqp2,mqp3,mqp4,mqq1,mqq2,mqq3,mqq4=  r
    tarr.append(t)
    marr.append(np.log(abs(mqq1*mqq4 - mqq2*mqq3)))
    #plt.scatter(t,mqq1*mqq4 - mqq2*mqq3,color='r')
    #ax.scatter(x,y,s=1)

def dpotential(Q):
    x = Q[:,0,...] #Change appropriately 
    y = Q[:,1,...]
    dxpoten = np.vectorize(dxpotential)
    dypoten = np.vectorize(dypotential)
    dpotx = dxpoten(x,y)#1.0*x + 2*x*y + 4*x*(x**2 + y**2)
    dpoty = dypoten(x,y)#1.0*y + x**2 - 1.0*y**2 + 4*y*(x**2 + y**2) 
    ret = np.transpose(np.array([dpotx,dpoty]),(1,0,2))
    return ret

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')

def Integration_instance(Q_arr,P_arr):
    xarr = Q_arr[...,0,0]
    yarr = Q_arr[...,1,0]
    pxarr = P_arr[...,0,0]
    pyarr = P_arr[...,1,0]
    oa =np.ones_like(xarr)
    za =np.zeros_like(xarr)
    
    init_arr = list(zip(xarr,pxarr,yarr,pyarr,oa,za,za,oa,za,za,za,za,za,za,za,za,oa,za,za,oa))
    init_arr = np.array(init_arr)
    
    return init_arr
    
def integrate(time,init_array):
    #print(init_value)
    sol = scipy.integrate.ode(f)
    sol.set_integrator('dop853',nsteps=100000)
    #sol.set_solout(solout)
    sol.set_initial_value(init_array,t=0.0)
    sol.set_f_params(dxpotential,dypotential,ddpotential1,ddpotential2,ddpotential3,ddpotential4)
    sol.integrate(time)
    return sol.y

def compute_monodromy_mqq(arr):
    x,px,y,py, mpp1,mpp2,mpp3,mpp4,mpq1,mpq2,mpq3,mpq4,mqp1,mqp2,mqp3,mqp4,mqq1,mqq2,mqq3,mqq4=  arr
    return np.log(abs(mqq1*mqq4 - mqq2*mqq3))


def vector_integrate(pool,tim,init_array):
    func = partial(integrate,tim)
    result = pool.map(func, init_array)
    return result

def vector_mqq(pool,arr):
    result = pool.map(compute_monodromy_mqq, arr)
    return result
#----------------------------------------------------------------------------------------------------- The following code is to be used for propagating Monodromy matrices for interacting systems.
n_particles = 20
n_dim = 3
N = n_particles*n_dim
m=4.0  #Has to be changed; It is very risky to define it as a global variable.
D=1.4135
A_s = 1.9221529e5 
beta_s = -1.89296514
R_m = 0.2970*18.897259885789
C_6= 1.34920045
C_8= 0.41365922
C_10= 0.17078164
alp_s = 10.73520708
eps = 10.94*0.31668115634e-5   #k_B = 0.31668115634e-5 a.u./K   # 1/k_B = 315775.02481 in atomic units (a.u./K)

#import Liquid_Helium_potential
#dpotential_He = Liquid_Helium_potential.liquid_helium_potential.dpotential
#vddpotential_He = Liquid_Helium_potential.liquid_helium_potential.vector_ddpotential
#ddpotential_He =  Liquid_Helium_potential.liquid_helium_potential.ddpotential 
#Liquid_Helium_potential.liquid_helium_potential.assign_pot_params()

def system_definition(n_p,n_d):
    global n_particles,n_dim,N
    n_particles = n_p
    n_dim = n_d
    N = n_particles*n_dim
    print('Now, n_particles, n_dim = ', n_particles,n_dim)

def func(y,t): 
    global m
    q = y[:N].reshape(n_particles,n_dim)
    p = y[N:2*N].reshape(n_particles,n_dim)
    n_mmat = n_particles*n_dim*n_dim
    Mpp = y[2*N:2*N+n_mmat].reshape(n_particles,n_dim,n_dim)
    Mpq = y[2*N+n_mmat:2*N+2*n_mmat].reshape(n_particles,n_dim,n_dim)
    Mqp = y[2*N+2*n_mmat:2*N+3*n_mmat].reshape(n_particles,n_dim,n_dim)
    Mqq = y[2*N+3*n_mmat:2*N+4*n_mmat].reshape(n_particles,n_dim,n_dim)
    dd_mpp = -np.matmul(ddpotential_He(q),Mqp)
    dd_mpq = -np.matmul(ddpotential_He(q),Mqq)
    #print(q,p)
    dydt = np.concatenate((p.flatten(),-dpotential_He(q).flatten(),dd_mpp.flatten(),dd_mpq.flatten(),Mpp.flatten()/m, Mpq.flatten()/m  ))
    
    return dydt

def scalar_dpotential(R):
    return 2*R*np.exp(-R**2)# 1/R**2#2*R*np.exp(-R**2)
    #print('D',D)
    if(0):
        if(R<=D):
            ret = eps*(A_s*(2*R*beta_s/R_m**2 - alp_s/R_m)*exp(R**2*beta_s/R_m**2 - R*alp_s/R_m) + 2*D*R_m*(D*R_m/R - 1)*(-C_10*R_m**10/R**10 - C_6*R_m**6/R**6 - C_8*R_m**8/R**8)*exp(-(D*R_m/R - 1)**2)/R**2                 + (10*C_10*R_m**10/R**11 + 6*C_6*R_m**6/R**7 + 8*C_8*R_m**8/R**9)*exp(-(D*R_m/R - 1)**2))
            #print('ret',ret)
            return float(ret)
        else:
            ret = eps*(A_s*(2*R*beta_s/R_m**2 - alp_s/R_m)*exp(R**2*beta_s/R_m**2 - R*alp_s/R_m) + 10*C_10*R_m**10/R**11 + 6*C_6*R_m**6/R**7 + 8*C_8*R_m**8/R**9)
            #print('ret',ret)
            return float(ret) 

def scalar_ddpotential(R):
    return (2-4*R**2)*np.exp(-R**2)
    if(0):
        if(R<=D):
            ret= eps*(2*A_s*beta_s*exp(R*(R*beta_s/R_m - alp_s)/R_m)/R_m**2 + A_s*(2*R*beta_s/R_m - alp_s)**2*exp(R*(R*beta_s/R_m - alp_s)/R_m)/R_m**2 - 4*D**2*R_m**8*(D*R_m/R - 1)**2*(C_10*R_m**4/R**4 + C_6 + C_8*R_m**2/R**2)*exp(-(D*R_m/R - 1)**2)/R**10 + 2*D**2*R_m**8*(C_10*R_m**4/R**4 + C_6 + C_8*R_m**2/R**2)*exp(-(D*R_m/R - 1)**2)/R**10 + 4*D*R_m**7*(D*R_m/R - 1)*(C_10*R_m**4/R**4 + C_6 + C_8*R_m**2/R**2)*exp(-(D*R_m/R - 1)**2)/R**9 + 8*D*R_m**7*(D*R_m/R - 1)*(5*C_10*R_m**4/R**4 + 3*C_6 + 4*C_8*R_m**2/R**2)*exp(-(D*R_m/R - 1)**2)/R**9 - 2*R_m**6*(55*C_10*R_m**4/R**4 + 21*C_6 + 36*C_8*R_m**2/R**2)*exp(-(D*R_m/R - 1)**2)/R**8)
            return float(ret)
        else:
            ret= -eps*(-2*A_s*beta_s*exp(R*(R*beta_s/R_m - alp_s)/R_m)/R_m**2 - A_s*(2*R*beta_s/R_m - alp_s)**2*exp(R*(R*beta_s/R_m - alp_s)/R_m)/R_m**2 + 110*C_10*R_m**10/R**12 + 42*C_6*R_m**6/R**8 + 72*C_8*R_m**8/R**10)
            return float(ret)

def force_dpotential(q):
    #print('Here')
    dpotential = np.zeros_like(q)
    for i in range(len(q)):
        for j in range(len(q)):
            if(i!=j):
                R = np.sum((q[i]-q[j])**2)**0.5
                #print('R py', R)
                unit = (q[i]-q[j])/R
                #print('hereh',scalar_dpotential(R))
                dpotential[i]+= scalar_dpotential(R)*unit

    return dpotential

def dpotential_He_aux(q):
    return dpotential_He(q)[:,:,np.newaxis]

def vector_ddpotential(R_vector,R):
    ret = R**2*np.identity(n_dim) - (np.einsum('i,j->ij', R_vector.ravel(), R_vector.ravel())) # np.outer(R_vector,R_vector)
    ret*=(1/R**3)
    return ret 

def ddpotential(q):
    ddpotential = np.zeros((n_particles,n_dim,n_dim))
    for i in range(n_particles):
        for j in range(n_particles):
            if(i!=j): 
                R_vector = q[i]-q[j]
                R = np.sum((R_vector)**2)**0.5
                unit = R_vector/R
                ddpotential[i]+= scalar_ddpotential(R)*(np.einsum('i,j->ij', unit.ravel(), unit.ravel())) + scalar_dpotential(R)*vector_ddpotential(R_vector,R)##np.outer(unit,unit) 
    return ddpotential

def ode_instance(q,p,tarr):
    print('Defining ODE instance with n_particles,n_dim',n_particles,n_dim)
    Mpp= np.zeros((n_particles,n_dim,n_dim))
    Mpq= np.zeros_like(Mpp)
    Mqp= np.zeros_like(Mpp)
    Mqq= np.zeros_like(Mpp)

    Mpp = np.array([np.identity(n_dim) for i in range(n_particles)])
    Mqq = Mpp.copy()
    y0=np.concatenate((q.flatten(), p.flatten(),Mpp.flatten(),Mpq.flatten(),Mqp.flatten(),Mqq.flatten() ))
    #print(y0)
    sol = scipy.integrate.odeint(func,y0,tarr,mxstep=100000)
    print(sol[len(tarr)-1,:10])
    return sol

def detmqq(sol):
    n_mmat = n_particles*n_dim**2
    sol_mqq = sol[:,2*N+3*n_mmat:2*N+4*n_mmat].reshape(len(sol),n_particles,n_dim,n_dim)
    #print('determinant',np.linalg.det(sol_mqq))
    print('mqq',np.shape(sol_mqq))
    return np.linalg.det(sol_mqq)

#rng = np.random.RandomState(1)
#q = rng.rand(n_particles,n_dim)
#p = rng.normal(0,1,(n_particles,n_dim)) #np.zeros_like(q)
#tarr = np.linspace(0,1,20)
print('changed')
#print('compare',scalar_dpotential(4.1),Liquid_Helium_potential.liquid_helium_potential.scalar_dpotential(4.1))
#print('compare',force_dpotential(q),Liquid_Helium_potential.liquid_helium_potential.dpotential(q))
#print('compare',vector_ddpotential(np.array([1.1,2.5,3.9]),1.9), vddpotential_He(np.array([1.1,2.5,3.9]),1.9))
#print('compare',ddpotential(q), ddpotential_He(q))

#print('FORTRAN')
#ode_instance(q,p,tarr)
#cProfile.run('ode_instance(q,p,tarr)')
#-------------------------------------------------------------
m=  0.5
n_particles = 100
n_dim =2 
n_beads= 4
w_n = n_beads
id_mat =  np.array([[np.identity(n_dim) for i in range(n_beads)] for j in range(n_particles)])
count = 0

x_arr = []
y_arr = []

def system_definition_RP(n_p,n_d,n_b,omega_n):
    global n_particles,n_dim,n_beads,w_n,id_mat
    n_particles = n_p
    n_dim = n_d
    n_beads = n_b
    w_n = omega_n

    id_mat =  np.array([[np.identity(n_dim) for i in range(n_beads)] for j in range(n_particles)])

def dpotential_c(q):
    return np.sum(dpotential_coupled_quartic(q),axis=1)/n_beads#**0.5    ### Why is this scaled as n_b^0.5?

def ddpotential_c(q):
    return np.sum(ddpotential_coupled_quartic(q),axis=1)/n_beads#**2

def ddpotential_ring(q):
    global m,w_n
    if(n_beads==1):
        return np.zeros((n_particles,n_beads,n_dim,n_dim))
    else:
        return 2*m*w_n**2*id_mat

def dpotential_ring(q):
    global m,w_n
    return m*w_n**2*(2*q - np.roll(q,1,axis=1) - np.roll(q,-1,axis=1))

def func_RP(y,t): 
    global m,n_particles,n_beads,n_dim, count
    count+=1
        
    N = n_particles*n_beads*n_dim
    q = y[:N].reshape(n_particles,n_beads,n_dim)
    p = y[N:2*N].reshape(n_particles,n_beads,n_dim)
    n_mmat = n_particles*n_dim*n_dim*n_beads
    Mpp = y[2*N:2*N+n_mmat].reshape(n_particles,n_beads,n_dim,n_dim)
    Mpq = y[2*N+n_mmat:2*N+2*n_mmat].reshape(n_particles,n_beads,n_dim,n_dim)
    Mqp = y[2*N+2*n_mmat:2*N+3*n_mmat].reshape(n_particles,n_beads,n_dim,n_dim)
    Mqq = y[2*N+3*n_mmat:2*N+4*n_mmat].reshape(n_particles,n_beads,n_dim,n_dim)
    dd_mpp = -np.matmul(ddpotential_coupled_quartic(q)+ddpotential_ring(q),Mqp)
    dd_mpq = -np.matmul(ddpotential_coupled_quartic(q)+ddpotential_ring(q),Mqq)
    

    if(count%100==0):
        x_arr.append(q[0,0,0])
        y_arr.append(q[0,0,1])
    #if(count>50000):
    #    plt.plot(q[0,:,0],q[0,:,1])   #np.mean(q,axis=2)[:,0],np.mean(q,axis=2)[:,1])
    #    plt.show()
    
    if(1):
        n_RP_coord = 2*N+4*n_mmat
        N_c = n_particles*n_dim
        Q_c = y[n_RP_coord: n_RP_coord+N_c].reshape(n_particles,n_dim)
        P_c = y[n_RP_coord+N_c: n_RP_coord+2*N_c].reshape(n_particles,n_dim)
        n_mmat_c = n_particles*n_dim*n_dim
        Mpp_c = y[n_RP_coord+2*N_c:n_RP_coord+2*N_c+n_mmat_c].reshape(n_particles,n_dim,n_dim)
        Mpq_c = y[n_RP_coord+2*N_c+n_mmat_c:n_RP_coord+2*N_c+2*n_mmat_c].reshape(n_particles,n_dim,n_dim)
        Mqp_c = y[n_RP_coord+2*N_c+2*n_mmat_c:n_RP_coord+2*N_c+3*n_mmat_c].reshape(n_particles,n_dim,n_dim)
        Mqq_c = y[n_RP_coord+2*N_c+3*n_mmat_c:n_RP_coord+2*N_c+4*n_mmat_c].reshape(n_particles,n_dim,n_dim)
        dd_mpp_c = -np.matmul(ddpotential_c(q),Mqp_c)
        dd_mpq_c = -np.matmul(ddpotential_c(q),Mqq_c)
      
        dycdt = np.concatenate((P_c.flatten()/m,-dpotential_c(q).flatten(),dd_mpp_c.flatten(),dd_mpq_c.flatten(),Mpp_c.flatten()/m,Mpq_c.flatten()/m))
    dydt = np.concatenate((p.flatten()/m,-(dpotential_coupled_quartic(q)+dpotential_ring(q)).flatten(),dd_mpp.flatten(),dd_mpq.flatten(),Mpp.flatten()/m, Mpq.flatten()/m  ))
    dYdt = np.concatenate((dydt,dycdt))
    print('count', count,t, (Mqq>1e7).any(), np.log(np.linalg.det(Mqq[0,0,:])))
    return dYdt

def ode_instance_RP(q,p,tarr):
    print('Defining ODE instance with n_particles,n_dim,n_beads ',n_particles,n_dim,n_beads)
    Mpp= np.zeros((n_particles,n_beads,n_dim,n_dim))
    Mpq= np.zeros_like(Mpp)
    Mqp= np.zeros_like(Mpp)
    Mqq= np.zeros_like(Mpp)

    Mpp = np.array([[np.identity(n_dim) for i in range(n_beads)] for j in range(n_particles)])
    Mqq = Mpp.copy()
    
    Q_c = np.mean(q,axis=1)
    P_c = np.mean(p,axis=1)

    Mpp_c= np.zeros((n_particles,n_dim,n_dim))
    Mpq_c= np.zeros_like(Mpp_c)
    Mqp_c= np.zeros_like(Mpp_c)
    Mqq_c= np.zeros_like(Mpp_c)
    
    Mpp_c = np.array([np.identity(n_dim) for i in range(n_particles)])
    Mqq_c = Mpp_c.copy()
    
    y0=np.concatenate((q.flatten(), p.flatten(),Mpp.flatten(),Mpq.flatten(),Mqp.flatten(),Mqq.flatten(),Q_c.flatten(),P_c.flatten(),Mpp_c.flatten(),Mpq_c.flatten(),Mqp_c.flatten(),Mqq_c.flatten()  ))
    #print('y0',len(y0))
    sol = scipy.integrate.odeint(func_RP,y0,tarr,mxstep=100000)
    print(sol[len(tarr)-1,:10])
    
    #plt.figure(1)
    #plt.plot(x_arr,y_arr)
    #plt.show()
    return sol

def detmqq_RP(sol):
    global n_particles,n_beads,n_dim
    n_mmat = n_particles*n_beads*n_dim**2
    N = n_particles*n_beads*n_dim
    n_RP_coord = 2*N+4*n_mmat
    N_c = n_particles*n_dim
    n_mmat_c = n_particles*n_dim*n_dim
    sol_mqq = sol[:,n_RP_coord+2*N_c+3*n_mmat_c:n_RP_coord+2*N_c+4*n_mmat_c].reshape(len(sol),n_particles,n_dim,n_dim)
    ret1 = (np.linalg.det(sol_mqq))**2
    
    sol_mqq_ring = sol[:,2*N+3*n_mmat:2*N+4*n_mmat].reshape(len(sol),n_particles,n_beads,n_dim,n_dim) 
    sol_mqq_det = (np.linalg.det(sol_mqq_ring))#**2
    print('mqq',np.shape(sol_mqq_det))
    ret2 = np.mean(sol_mqq_det,axis=2)**2 
    
    return ret2
    
def q_RP(sol):
    global n_particles, n_beads, n_dim
    n_mmat = n_particles*n_beads*n_dim**2
    N = n_particles*n_beads*n_dim
    n_RP_coord = 2*N+4*n_mmat
    N_c = n_particles*n_dim
    n_mmat_c = n_particles*n_dim*n_dim
    sol_q = sol[:,:N].reshape(len(sol),n_particles,n_beads,n_dim)
    print('q',np.shape(sol_q))
    return sol_q

if(0):
    rng = np.random.RandomState(1)
    q = np.zeros((n_particles,n_beads,n_dim))
    p = rng.rand(n_particles,n_beads,n_dim)
    tarr = np.linspace(0,20,100)

    #print(np.shape(ddpotential_coupled_quartic(q)))

    sol = ode_instance_RP(q,p,tarr)
    plt.plot(tarr,np.mean(abs(detmqq_RP(sol)),axis=1))
    plt.show()
    #print(np.shape(dpotential_cq_vv(q)))
#-------------------------------------------------------------
#temp =integrate(40.0,[0.5661284083880359, 0.2435087008868868, 1.8554349125204095, 0.9124931713411446,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1] )

#import psutil
#import os
#process = psutil.Process(os.getpid())
#
#
#Q_arr = np.random.rand(100,2,3)
#P_arr = np.random.rand(100,2,3)
##print(Q_arr,P_arr)
#temp = Integration_instance(Q_arr,P_arr)#vector_integrate(20.0,temp_rand)
#
#pool = mp.Pool(mp.cpu_count()-5)
#for i in range(1):
#    #print('1',process.memory_info().rss)
#    temp = vector_integrate(pool,200.0,temp)
#    #print('2',process.memory_info().rss)
#    
#
#
##print(vector_mqq(temp))
#pool.close()
#pool.join()





#print(temp)

#print(compute_monodromy(temp))
#Q_arr = np.random.rand(5,2,3)
#P_arr = np.random.rand(5,2,3)

#print(integration_instance(Q_arr,P_arr,2.0))

#--------------------------------------------------------------

#lo=-12
#up=12
#xarr = np.linspace(lo,up,100)
#yarr = np.linspace(lo,up,100)
#
#X,Y = np.meshgrid(xarr,yarr)
#Z = np.zeros_like(X)
#for i in range(len(xarr)):
#    for j in range(len(yarr)):
#        Z[i][j] = potential(X[i][j],Y[i][j])
##Z[Z>0.2] = 0.2
##Z[Z<-0.2] = -0.2
#    
#integrate(1000.0,[1.0,0.5,0.0,0.5  ,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1] )
##plt.imshow(Z,origin='lower')
##cset = ax.contour(X, Y, Z)
#
#
#plt.plot(tarr,marr)
#plt.show()
#-------------------------------------------------------------

#for i in range(100):
#    sol.integrate(0.1)
#    plt.scatter(sol.y[0],sol.y[2])
#    
#plt.show()


