#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 15:15:02 2020

@author: vgs23
"""
import numpy as np

"""
Eliminate all the global variable declaration, Clean up unwanted code, Maintain a standard format for potential definition

"""

"""
The potential below is the stadium billiards potential that
is generically used for understanding quantum chaos. Here, we 
use two forms of the potential: One is 'smooth', and 
the other is hard walled. 

Le: Length of the billiards, excluding the two curved region
s: Smoothness parameter
r_c: Radius of the curved semicircular region
K: Energy of the potential 'wall'
"""
Le =  2*(1/(np.pi+4)**0.5)  
s= 20.0
r_c = 1/(np.pi+4)**0.5
K = 10.0

print('Le', Le, 'r_c', r_c)
#print('Area',np.pi*r_c**2 + Le*2*r_c)

def radial_to_cartesian(rforce):
    def force_2D(x,y):
        r = (x**2+y**2)**0.5
        #if(r>=5.5):
            #print('x,y',x,y,r)
        force_r = rforce(r)
        force_xy = force_r*abs(np.array([x/r,y/r]))
        force_xy = np.array(force_xy)
        return force_xy
    return force_2D
        
def potential_harmonic(x,y):
    return 0.5*(x**2+y**2)

def potential_sb(x,y):
#    if(abs(x)<Le/2):
#        return K*(1/(np.exp(s*(r_c + y)) + 1) + 1/(1 + np.exp(-s*(-r_c + y))))
#    else:
#        if(x<0):
#            #r = ((x+Le/2)**2 +y**2)**0.5
#            return K*(1/(1 + np.exp(-s*(-r_c + (y**2 + (Le/2 + x)**2)**0.5))))
#        else:
#            #r = ((x-Le/2)**2 +y**2)**0.5
#            return K*(1/(1 + np.exp(-s*(-r_c + (y**2 + (-Le/2 + x)**2)**0.5))))
    #print('hi', Le, r_c)    
    if(abs(x)<Le/2 and abs(y)<r_c):
        return 0.0
    elif(abs(x)<(Le/2+r_c) and abs(y)<r_c):
        if(x>0):
            if(((x-Le/2)**2+y**2)<r_c**2):
                return 0.0
            else:
                return 1e5
        else:
            if(((x+Le/2)**2+y**2)<r_c**2):
                return 0.0
            else:
                return 1e5
    else:
        return 1e5



def pot_barrier(x,y,x_c,y_c,r_c):
    r = ((x-x_c)**2 + (y-y_c)**2)**0.5
    if(abs(r)<=r_c):
        return 1e5
    else:
        return 0.0
        
pot_barrier = np.vectorize(pot_barrier)

#Ls = 10.0
#r_c = 0.3
#n_barrier = 25
def potential_lg(x,y):
    if(abs(x)>Ls/2 or abs(y)>Ls/2):
        return 1e5
    else:
        x_ar = np.arange(-2,2.1,1.0)
        y_ar = np.arange(-2,2.1,1.0)
        X_a, Y_a = np.meshgrid(x_ar,y_ar)
        return np.sum(pot_barrier(x,y,X_a,Y_a,r_c))
        #return pot_barrier(x,y,-0.0,-0.0,r_c) #+ pot_barrier(x,y,0.2,0.2,r_c) \
               #+ pot_barrier(x,y,-0.2,0.2,r_c) + pot_barrier(x,y,0.2,-0.2,r_c)
    
def potential_coupled_quartic(x,y):
    b=0.1
    return  (b/4.0)*(x**4 + y**4) + 0.5*x**2*y**2

def dpotential_coupled_quartic(Q):
    b=0.1
    print('unwanted call')
    x = Q[:,0,...] #Change appropriately 
    y = Q[:,1,...]
    dpotx = x#(b/4.0)*(4*x**3 + y**4) + x*y**2
    dpoty = y#(b/4.0)*(x**4 + 4*y**3) + x**2*y
    
    ret = np.transpose(np.array([dpotx,dpoty]),(1,0,2))
    #print(ret)
    return ret

def potential_cb(x,y):
    
    D0 = 0.18748
    alpha = 1.1605
    r_c = 1.8324
    r= (x**2 + y**2)**0.5
    V_cb = D0*(1 - np.exp(-alpha*(r-r_c)))**2
    return V_cb

def dpotential_cb_x(x,y):
    K=0.49
    r_c = 1.89
    
    D0 = 0.18748
    alpha = 1.1605
    r_c = 1.8324
    dpotx = 2.0*D0*alpha*x*(1 - np.exp(-alpha*(-r_c + (x**2 + y**2)**0.5)))*(x**2 + y**2)**(-0.5)*np.exp(-alpha*(-r_c + (x**2 + y**2)**0.5))

    return dpotx

def dpotential_cb(Q):
    #print('hereh',np.shape(Q))
    x = Q[:,0,...] #Change appropriately 
    y = Q[:,1,...]
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
    ret = np.transpose(np.array([dpotx,dpoty]),(1,0,2))
    #print(ret)
    return ret

def potential_rh(x,y):
    K=0.49
    r_c = 1.89
    r= (x**2 + y**2)**0.5
    V_rh = (K/2)*(r-r_c)**2
    
    return V_rh

def dpotential_morse(Q):
    D0 =  0.18748
    Q_c = 1.8324
    alpha = 1.1605
    return 2*D0*alpha*(1 - np.exp(-alpha*(Q - Q_c)))*np.exp(-alpha*(Q - Q_c))

def potential_quartic(x,y):
    return 0.25*(x**4+y**4)

def dpotential_quartic(Q):
    return Q**3

pot_code = 'V_hh'

global potential, dxpotential, dypotential, ddpotential1, ddpotential2, ddpotential3, ddpotential4

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
            
