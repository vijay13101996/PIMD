#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 14:03:02 2019

@author: vgs23
"""

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy,copy
from numpy import linalg as LA
from numpy.fft import fft,ifft
from multiprocessing import Process
import MD_System
import multiprocessing as mp
from functools import partial
import scipy
from sympy import *
#from Langevin_thermostat import thermalize
#from Centroid_mean_force import CMF
#from External_potential import dpotential
import Centroid_mean_force
import Velocity_verlet
import importlib
import time
import pickle
import sympy
start_time = time.time()
from Matsubara_potential_hh import dpot_hh,pot_hh

#1. Input initial conditions (To be chosen according to an initial energy constraint)
#2. Chose time step to propagate, and to check for recurrence. Set recurrence axes 
#3. Propagate the trajectories in the potential and record recurrence
#4. Plot the phase space along the recurrence axes. 

x = sympy.Symbol('x')
y = sympy.Symbol('y')

V_hh = x**2/2.0 + y**2/2.0 + x**2*y - y**3/3.0 +(x**2+y**2)**2

print(diff(V_hh,x))
print(diff(V_hh,y))

def potential(x,y):
    return x**2/2.0 + y**2/2.0 + x**2*y - y**3/3.0 +(x**2+y**2)**2

def potential_morse(Q):
    D0 =  0.18748
    Q_c = 1.8324
    alpha = 1.1605
    return D0*(1-np.exp(-alpha*(Q-Q_c)))**2

Q = sympy.Symbol('Q')
Q_c = sympy.Symbol('Q_c')
alpha = sympy.Symbol('alpha')
D0 = sympy.Symbol('D0')

#V_morse = D0*(1-exp(-alpha*(Q-Q_c)))**2

#print(diff(V_morse,Q))


def dpotential_morse(Q):
    D0 =  0.18748
    Q_c = 1.8324
    alpha = 1.1605
    return 2*D0*alpha*(1 - np.exp(-alpha*(Q - Q_c)))*np.exp(-alpha*(Q - Q_c))
    
def Energy(Q,P):
    x = Q[:,0,0]
    y = Q[:,1,0]
    px = P[:,0,0]
    py = P[:,1,0]
    return px**2/2.0 + py**2/2.0 + potential(x,y)

def dpotential(Q):
    x = Q[:,0,...] #Change appropriately 
    y = Q[:,1,...]
    dpotx = 2*x*y + 1.0*x + 4*x*(x**2 + y**2)
    dpoty = x**2 - 1.0*y**2 + 1.0*y + 4*y*(x**2 + y**2) 
    ret = np.transpose(np.array([dpotx,dpoty]),(1,0,2))
    return ret
  
Y =[]
PY = []

def Poincare_section(dpotential,swarmobject):
    
    t=0.0
    deltat = 1e-2
    T=500.0
    count = 0
    
    #print(Energy(swarmobject.q,swarmobject.p))
    prev = swarmobject.q[0,0,0]
    curr = swarmobject.q[0,0,0]
    while(t<=T):
        Velocity_verlet.vv_step(0,0,1,swarmobject,dpotential,deltat)
        
        x = swarmobject.q[0,0,0]
        px = swarmobject.p[0,0,0]
        y = swarmobject.q[0,1,0]
        py = swarmobject.p[0,1,0]
        curr =x
        if(count%1==0):
            #print(prev,curr)
            if( prev*curr<0.0 and px<0.0 ):
                #print('herhe',count)
                Y.append(y)
                PY.append(py)
        prev = curr
        t+=deltat
        count+=1

def Poincare_section_mats(dpot,swarmobject):
    
    t=0.0
    deltat = 1e-2
    T=500.0
    count = 0
    
    #print(Energy(swarmobject.q,swarmobject.p))
#    prev = swarmobject.p[0,0,0]
#    curr = swarmobject.p[0,0,0]
    while(t<=T):
        Velocity_verlet.vv_step(0,0,1,swarmobject,dpot,deltat)
        
        x = swarmobject.q[0,0,1]
        px = swarmobject.p[0,0,1]
        y = swarmobject.q[0,1,1]
        py = swarmobject.p[0,1,1]
        #curr =x
        if(count%1==0):
            #print(prev,curr)
            if( abs(x)<1e-3 and px<0.0 ):
                #print('herhe',count)
                Y.append(y)
                PY.append(py)
        #prev = curr
        t+=deltat
        count+=1


rand = np.random.RandomState(1)    
MD_System.system_definition(8,3,2,dpotential)
importlib.reload(Velocity_verlet)
swarmobject = MD_System.swarm(1)
#E = 0.08

def compute_P_section(E,Y,PY):
    for i in range(10):
        x0 = rand.uniform(-(0.5*E)**0.5,(0.5*E)**0.5)
        y0 = rand.uniform(-(0.5*E)**0.5,(0.5*E)**0.5)
        
        differ = E-potential(x0,y0)
        split = rand.uniform(0,2)
        px0 = (split*differ)**0.5
        py0 = ((2-split)*differ)**0.5
        swarmobject.q = np.array([[[x0],[y0]]]) 
        swarmobject.p = np.array([[[px0],[py0]]])
        Poincare_section(dpotential,swarmobject) 
        #print(swarmobject.q)
        px0 = -(split*differ)**0.5
        py0 = -((2-split)*differ)**0.5
        swarmobject.q = np.array([[[x0],[y0]]])
        swarmobject.p = np.array([[[px0],[py0]]])
        Poincare_section(dpotential,swarmobject)
        
        px0 = (split*differ)**0.5
        py0 = -((2-split)*differ)**0.5
        swarmobject.q = np.array([[[x0],[y0]]])
        swarmobject.p = np.array([[[px0],[py0]]])
        Poincare_section(dpotential,swarmobject)
        
        px0 = -(split*differ)**0.5
        py0 = ((2-split)*differ)**0.5
        swarmobject.q = np.array([[[x0],[y0]]])
        swarmobject.p = np.array([[[px0],[py0]]])
        Poincare_section(dpotential,swarmobject)
        print(i, 'completed')
    
    f = open('Poincare_section_Classical_E_{}.dat'.format(E),'wb')
    pickle.dump(Y,f)
    pickle.dump(PY,f)
    f.close()
    
    plt.suptitle('Poincare section from Classical dynamics')
    plt.title('$E={}$'.format(E),fontsize=8)
    plt.xlabel('$y$')
    plt.ylabel('$p_y$')
    plt.scatter(Y,PY,s=2)
    plt.savefig('Poincare_section_Classical_E_{}.png'.format(E))

def compute_P_section_Matsubara(E,Y,PY):   
    for i in range(5):
        x0 = rand.uniform(-(0.5*E)**0.5,(0.5*E)**0.5)
        y0 = rand.uniform(-(0.5*E)**0.5,(0.5*E)**0.5)
        swarmobject.q = np.array([[[0.0,x0,0.0],[0.0,y0,0.0]]])
        #E = 0.1
        #print(np.shape(swarmobject.q))
        
        differ = E-np.sum(pot_hh(swarmobject.q))
        split = rand.uniform(0,2)
        px0 = (split*differ)**0.5
        py0 = ((2-split)*differ)**0.5
        
        swarmobject.p = np.array([[[0,px0,0],[0,py0,0]]])
        Poincare_section_mats(dpot_hh,swarmobject)
        #print(swarmobject.q)
        
        differ = E-np.sum(pot_hh(swarmobject.q))
        split = rand.uniform(0,2)
        px0 = -(split*differ)**0.5
        py0 = -((2-split)*differ)**0.5
        
        swarmobject.p = np.array([[[0,px0,0],[0,py0,0]]])
        Poincare_section_mats(dpot_hh,swarmobject)
        
        differ = E-np.sum(pot_hh(swarmobject.q))
        split = rand.uniform(0,2)
        px0 = -(split*differ)**0.5
        py0 = ((2-split)*differ)**0.5
        
        swarmobject.p = np.array([[[0,px0,0],[0,py0,0]]])
        Poincare_section_mats(dpot_hh,swarmobject)
        
        differ = E-np.sum(pot_hh(swarmobject.q))
        split = rand.uniform(0,2)
        px0 = (split*differ)**0.5
        py0 = -((2-split)*differ)**0.5
        
        swarmobject.p = np.array([[[0,px0,0],[0,py0,0]]])
        Poincare_section_mats(dpot_hh,swarmobject)
        print(i, 'completed')
    
#    f = open('Poincare_section_Matsubara_E_{}.dat'.format(E),'wb')
#    pickle.dump(Y,f)
#    pickle.dump(PY,f)
#    f.close()
#    
#    plt.suptitle('Poincare section from Matsubara dynamics')
#    plt.title('$E={}$'.format(E),fontsize=8)
#    plt.xlabel('$y$')
#    plt.ylabel('$p_y$')
#    plt.scatter(Y,PY,s=2)
#    plt.savefig('Poincare_section_Matsubara_E_{}.png'.format(E))
#---------------------------------------------------------
E=1.0
print('Energy (Non centroid)',E)
compute_P_section(E,Y,PY)

f = open('Poincare_section_Matsubara_E_{}_ncentroid.dat'.format(E),'wb')
pickle.dump(Y,f)
pickle.dump(PY,f)
f.close()

plt.suptitle('Poincare section from Matsubara dynamics (Non-Centroid)')
plt.title('$E={}$'.format(E),fontsize=8)
plt.xlabel('$y$')
plt.ylabel('$p_y$')
plt.scatter(Y,PY,s=2)
plt.savefig('Poincare_section_Matsubara_E_{}_ncentroid.png'.format(E))

#--------------------------------------------------------

#def dpot_harmonic(Q):
#    x = Q[:,0,...]
#    y = Q[:,1,...]
#    dpotx =  x
#    dpoty =  y
#    #print(Q,np.shape(Q))
#    ret = np.transpose(np.array([dpotx,dpoty]),(1,0,2))
#    #print('herhe',np.shape(ret))
#    #print(ret)
#    #print('new')
#    return ret
#
#beads = 8
#MD_System.system_definition(8,beads,2,dpotential)
#importlib.reload(Velocity_verlet)
#Centroid_mean_force.mean_force_calculation_2D(dpotential)
# 
#f = open("CMD_Force_2D_B_{}_NB_{}_101_10.dat".format(MD_System.beta,MD_System.n_beads),'rb')       
#xarr = pickle.load(f)
#yarr = pickle.load(f)
#CMD_force = pickle.load(f)
#
#xforce = scipy.interpolate.RectBivariateSpline(xarr,yarr,CMD_force[:,:,0])
#print(xforce(0.2,0.2))     
 
#------------------------------------------------------------------
#l=-1
#u=1       
#xarr = np.linspace(l,u,100)
#yarr = np.linspace(l,u,100)
#
#X,Y = np.meshgrid(xarr,yarr)
#Z = potential(X,Y)
#
##plt.imshow(Z,origin='lower')
#
#from mpl_toolkits.mplot3d import Axes3D
#
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#
#cset = ax.contour(X, Y, Z)
#plt.show()
#Axes3D.contour(X,Y,Z)