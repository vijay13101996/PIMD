#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 15:15:02 2020

@author: vgs23
"""
import numpy as np
Le = 2*(1/(np.pi+4)**0.5)  
s= 20.0
r_c = 1/(np.pi+4)**0.5
K = 10.0
print('Area',np.pi*r_c**2 + Le*2*r_c)


Ls = 10.0
r_c = 1.0
n_barrier = 0
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
def potential_lg(x,y):
    if(abs(x)>Ls/2 or abs(y)>Ls/2):
        return 1e5
    else:
        x_ar = np.arange(-1,1.1,1.0)
        y_ar = np.arange(-1,1.1,1.0)
        X_a, Y_a = np.meshgrid(x_ar,y_ar)
        #return np.sum(pot_barrier(x,y,X_a,Y_a,r_c))
        return pot_barrier(x,y,-0.0,-0.0,r_c) #+ pot_barrier(x,y,0.2,0.2,r_c) \
               #+ pot_barrier(x,y,-0.2,0.2,r_c) + pot_barrier(x,y,0.2,-0.2,r_c)
    
def potential_quartic(x,y):
    b=0.1
    return (b/4.0)*(x**4 + y**4) + 0.5*x**2*y**2