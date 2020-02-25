#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 15:15:02 2020

@author: vgs23
"""
import numpy as np

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
Le = 2*(1/(np.pi+4)**0.5)  
s= 20.0
r_c = 1/(np.pi+4)**0.5
K = 10.0
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

Ls = 10.0
r_c = 0.3
n_barrier = 25
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
    return (b/4.0)*(x**4 + y**4) + 0.5*x**2*y**2

def dpotential_coupled_quartic(Q):
    b=0.1
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