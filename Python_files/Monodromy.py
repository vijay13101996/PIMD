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
import Matsubara_potential

start_time = time.time()

# TO DO:
# 1. Get rid of the sympy code below. Create a new exclusive file for sympy code.
# 2. Comment out this code. Remove unnecessary potential definitions inside. 
# 3. Try to obtain a file structure similar to that of MD Simulation files
# 4. Rewrite code to ensure array ordering: (N, ndim, nbeads)
#---------------------------------------------------------------           
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

def solout(t,r):
    x,px,y,py, mpp1,mpp2,mpp3,mpp4,mpq1,mpq2,mpq3,mpq4,mqp1,mqp2,mqp3,mqp4,mqq1,mqq2,mqq3,mqq4=  r
    tarr.append(t)
    marr.append(np.log(abs(mqq1*mqq4 - mqq2*mqq3)))
    
def dpotential(Q):
    x = Q[:,0,...] #Change appropriately 
    y = Q[:,1,...]
    dxpoten = np.vectorize(dxpotential)
    dypoten = np.vectorize(dypotential)
    dpotx = dxpoten(x,y)#1.0*x + 2*x*y + 4*x*(x**2 + y**2)
    dpoty = dypoten(x,y)#1.0*y + x**2 - 1.0*y**2 + 4*y*(x**2 + y**2) 
    ret = np.transpose(np.array([dpotx,dpoty]),(1,0,2))
    return ret

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
def func_classical(y,t,swarmobj,dpotential,ddpotential): 
    m = swarmobj.m
    n_particles = swarmobj.N
    n_dim = swarmobj.dimension
    
    N = n_particles*n_dim
    q = y[:N].reshape(n_particles,n_dim)
    p = y[N:2*N].reshape(n_particles,n_dim)
    n_mmat = n_particles*n_dim*n_dim
    Mpp = y[2*N:2*N+n_mmat].reshape(n_particles,n_dim,n_dim)
    Mpq = y[2*N+n_mmat:2*N+2*n_mmat].reshape(n_particles,n_dim,n_dim)
    Mqp = y[2*N+2*n_mmat:2*N+3*n_mmat].reshape(n_particles,n_dim,n_dim)
    Mqq = y[2*N+3*n_mmat:2*N+4*n_mmat].reshape(n_particles,n_dim,n_dim)
    dd_mpp = -ddpotential(q)[:,:,None]*Mqp  ## CHANGE HERE FOR MORE THAN 1D
    dd_mpq = -ddpotential(q)[:,:,None]*Mqq
    #print('Mpp', Mpp, np.cos(t)) 
    #print('Energy', p**2/(2*m) + Matsubara_potential.pot_inv_harmonic_M1(q)) 
    dydt = np.concatenate((p.flatten()/swarmobj.m,-dpotential(q).flatten(),dd_mpp.flatten(),dd_mpq.flatten(),Mpp.flatten()/m, Mpq.flatten()/m)) 
    return dydt

def ode_instance_classical(swarmobj,tarr,dpotential,ddpotential):
    n_particles = swarmobj.N
    n_dim = swarmobj.dimension
    print('Defining ODE instance with n_particles,n_dim',n_particles,n_dim)
    q = swarmobj.q.copy()
    p = swarmobj.p.copy()
    
    Mpp= np.zeros((n_particles,n_dim,n_dim))
    Mpq= np.zeros_like(Mpp)
    Mqp= np.zeros_like(Mpp)
    Mqq= np.zeros_like(Mpp)

    Mpp = np.array([np.identity(n_dim) for i in range(n_particles)])
    Mqq = Mpp.copy()
    y0=np.concatenate((q.flatten(), p.flatten(),Mpp.flatten(),Mpq.flatten(),Mqp.flatten(),Mqq.flatten()))
    sol = scipy.integrate.odeint(func_classical,y0,tarr,args = (swarmobj,dpotential,ddpotential),mxstep=100)
    print(sol[len(tarr)-1,:10])
    return sol

def detmqq_classical(sol,swarmobj):
    n_particles = swarmobj.N
    n_dim = swarmobj.dimension
    N = n_particles*n_dim
    n_mmat = n_particles*n_dim**2
    sol_mqq = sol[:,2*N+3*n_mmat:2*N+4*n_mmat].reshape(len(sol),n_particles,n_dim,n_dim)
    
    ret = sol_mqq[:,:,0,0]**2#(np.linalg.det(sol_mqq))**2 
    return ret

def q_classical(sol,swarmobj):  
    n_particles = swarmobj.N
    n_dim = swarmobj.dimension
    n_mmat = n_particles*n_dim**2
    N = n_particles*n_dim
    sol_q = sol[:,:N].reshape(len(sol),n_particles,n_dim)
    #print('q',np.shape(sol_q))
    return sol_q

def p_classical(sol,swarmobj):  
    n_particles = swarmobj.N
    n_dim = swarmobj.dimension
    n_mmat = n_particles*n_dim**2
    N = n_particles*n_dim
    sol_p = sol[:,N:2*N].reshape(len(sol),n_particles,n_dim)
    #print('q',np.shape(sol_q))
    return sol_p

#----------------------------------------------------------------------------------------------------
def dpotential_c(q,dpotential):
    return np.sum(dpotential(q),axis=2)/n_beads#**0.5    ### Why is this scaled as n_b^0.5?

def ddpotential_c(q,ddpotential):
    return np.sum(ddpotential(q),axis=2)/n_beads#**2

def func_RP(y,t,swarmobj,dpotential,ddpotential): 
    m = swarmobj.m
    n_particles = len(swarmobj)
    n_dim = len(swarmobj.dimension)
    n_beads = len(swarmobj.n_beads)
    
    global count
    count+=1
        
    N = n_particles*n_beads*n_dim
    q = y[:N].reshape(n_particles,n_beads,n_dim)
    p = y[N:2*N].reshape(n_particles,n_beads,n_dim)
    n_mmat = n_particles*n_dim*n_dim*n_beads
    Mpp = y[2*N:2*N+n_mmat].reshape(n_particles,n_beads,n_dim,n_dim)
    Mpq = y[2*N+n_mmat:2*N+2*n_mmat].reshape(n_particles,n_beads,n_dim,n_dim)
    Mqp = y[2*N+2*n_mmat:2*N+3*n_mmat].reshape(n_particles,n_beads,n_dim,n_dim)
    Mqq = y[2*N+3*n_mmat:2*N+4*n_mmat].reshape(n_particles,n_beads,n_dim,n_dim)
    dd_mpp = -np.matmul(ddpotential(q)+swarmobj.ddpotential_ring(),Mqp)
    dd_mpq = -np.matmul(ddpotential(q)+ddpotential_ring(q),Mqq)
    
    if(count%100==0):
        x_arr.append(q[0,0,0])
        y_arr.append(q[0,0,1])
    
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
        dd_mpp_c = -np.matmul(ddpotential_c(q,ddpotential),Mqp_c)
        dd_mpq_c = -np.matmul(ddpotential_c(q,ddpotential),Mqq_c)
      
        dycdt = np.concatenate((P_c.flatten()/m,-dpotential_c(q,dpotential).flatten(),dd_mpp_c.flatten(),dd_mpq_c.flatten(),Mpp_c.flatten()/m,Mpq_c.flatten()/m))
    dydt = np.concatenate((p.flatten()/m,-(dpotential(q)+swarmobj.dpotential_ring()).flatten(),dd_mpp.flatten(),dd_mpq.flatten(),Mpp.flatten()/m, Mpq.flatten()/m  ))
    dYdt = np.concatenate((dydt,dycdt))
    print('count', count,t, (Mqq>1e7).any(), np.log(np.linalg.det(Mqq[0,0,:])))
    return dYdt

def ode_instance_RP(swarmobj,tarr):
    n_particles = len(swarmobj)
    n_dim = len(swarmobj.dimension)
    n_beads = len(swarmobj.n_beads)
    
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
    
    y0=np.concatenate((swarmbj.q.flatten(), swarmobj.p.flatten(),Mpp.flatten(),Mpq.flatten(),\
            Mqp.flatten(),Mqq.flatten(),Q_c.flatten(),P_c.flatten(),Mpp_c.flatten(),Mpq_c.flatten(),Mqp_c.flatten(),Mqq_c.flatten()  ))
    sol = scipy.integrate.odeint(func_RP,y0,tarr,mxstep=100000)
    print(sol[len(tarr)-1,:10])
    
    return sol

def detmqq_RP(sol,swarmobj):
    n_particles = len(swarmobj)
    n_beads = swarmobj.n_beads
    n_dim = swarmobj.dimension
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

#-------------------------------------------------------------------------------------------------------------
count = 0
def func_Matsubara(y,t,swarmobj,dpotential,ddpotential): 
    #print('t',t)
    m = swarmobj.m
    n_particles = swarmobj.N
    n_dim = swarmobj.dimension
    n_beads = swarmobj.n_beads
    
    global count
    count+=1
        
    N = n_particles*n_beads*n_dim
    q = y[:N].reshape(n_particles,n_dim,n_beads)
    p = y[N:2*N].reshape(n_particles,n_dim,n_beads)
    n_mmat = n_particles*n_dim*n_dim*n_beads*n_beads
    Mpp = y[2*N:2*N+n_mmat].reshape(n_particles,n_dim,n_dim,n_beads,n_beads)
    Mpq = y[2*N+n_mmat:2*N+2*n_mmat].reshape(n_particles,n_dim,n_dim,n_beads,n_beads)
    Mqp = y[2*N+2*n_mmat:2*N+3*n_mmat].reshape(n_particles,n_dim,n_dim,n_beads,n_beads)
    Mqq = y[2*N+3*n_mmat:2*N+4*n_mmat].reshape(n_particles,n_dim,n_dim,n_beads,n_beads)   ## TAKE CARE (NEXT LINE) WHEN YOU DO MORE THAN 1D
    dd_mpp = -np.einsum('abcij,abcjk->abcik',ddpotential(q),Mqp)#-np.matmul(ddpotential(q),Mqp)
    dd_mpq = -np.einsum('abcij,abcjk->abcik',ddpotential(q),Mqq)#-np.matmul(ddpotential(q),Mqq)
    
    #print('ddpot', Mpp, np.cos(t))
    #print('matmul',-np.matmul(ddpotential(q)[1][0],Mqp[1,0,0]))
    #print('ddmpp', dpotential(q)[abs(dpotential(q))>1e4])

    dydt = np.concatenate((p.flatten()/m,-dpotential(q).flatten(),dd_mpp.flatten(),dd_mpq.flatten(),Mpp.flatten()/m, Mpq.flatten()/m))
    #print('count', count,t, (Mqq>1e7).any(), np.log(np.linalg.det(Mqq[0,0,:])))
    return dydt

def ode_instance_Matsubara(swarmobj,tarr,dpotential,ddpotential):
    n_particles = swarmobj.N
    n_dim = swarmobj.dimension
    n_beads = swarmobj.n_beads
    q = swarmobj.q.copy()
    p = swarmobj.p.copy()
    #print('Defining ODE instance with n_particles,n_dim,n_beads ',n_particles,n_dim,n_beads)
    Mpp= np.zeros((n_particles,n_dim,n_dim,n_beads,n_beads))
    Mpq= np.zeros_like(Mpp)
    Mqp= np.zeros_like(Mpp)
    Mqq= np.zeros_like(Mpp)

    Mpp = np.array([[[np.identity(n_beads) for i in range(n_dim)] for j in range(n_dim)] for k in range(n_particles)])
    Mqq = Mpp.copy()
        
    y0=np.concatenate((q.flatten(), p.flatten(),Mpp.flatten(),Mpq.flatten(),Mqp.flatten(),Mqq.flatten()))
    sol = scipy.integrate.odeint(func_Matsubara,y0,tarr,args = (swarmobj,dpotential,ddpotential),hmax=0.1,hmin=1e-9,mxstep=10000)
    
    return sol

def detmqq_Matsubara(sol,swarmobj): 
    n_particles = swarmobj.N
    n_beads = swarmobj.n_beads
    n_dim = swarmobj.dimension
    n_mmat = n_particles*n_beads**2*n_dim**2
    N = n_particles*n_beads*n_dim
    sol_mqq = sol[:,2*N+3*n_mmat:2*N+4*n_mmat].reshape(len(sol),n_particles,n_dim,n_dim,n_beads,n_beads)
    ret1 = sol_mqq[:,:,0,0,n_beads//2,n_beads//2]**2#(np.linalg.det(sol_mqq))**2 
    return ret1#[:,:,0,0]

def detmpp_Matsubara(sol,swarmobj): 
    n_particles = swarmobj.N
    n_beads = swarmobj.n_beads
    n_dim = swarmobj.dimension
    n_mmat = n_particles*n_beads**2*n_dim**2
    N = n_particles*n_beads*n_dim
    sol_mpp = sol[:,2*N:2*N+n_mmat].reshape(len(sol),n_particles,n_dim,n_dim,n_beads,n_beads)
    ret1 = (np.linalg.det(sol_mpp))**2 
    #print('ret1', ret1.shape)
    return ret1[:,:,0,0]
 
def q_Matsubara(sol,swarmobj):  
    n_particles = swarmobj.N
    n_beads = swarmobj.n_beads
    n_dim = swarmobj.dimension
    n_mmat = n_particles*n_beads**2*n_dim**2
    N = n_particles*n_beads*n_dim
    sol_q = sol[:,:N].reshape(len(sol),n_particles,n_dim,n_beads)
    #print('q',np.shape(sol_q))
    return sol_q

def p_Matsubara(sol,swarmobj):  
    n_particles = swarmobj.N
    n_beads = swarmobj.n_beads
    n_dim = swarmobj.dimension
    n_mmat = n_particles*n_beads**2*n_dim**2
    N = n_particles*n_beads*n_dim
    sol_p = sol[:,N:2*N].reshape(len(sol),n_particles,n_dim,n_beads)
    #print('q',np.shape(sol_q))
    return sol_p

#-------------------------------------------------------------------------------------------------------------
if(0):
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

if(0):
    n_particles = 20
    n_dim = 3
    N = n_particles*n_dim
    m=4.0  #Has to be changed; It is very risky to define it as a global variable.

    def system_definition(n_p,n_d):
        global n_particles,n_dim,N
        n_particles = n_p
        n_dim = n_d
        N = n_particles*n_dim
        print('Now, n_particles, n_dim = ', n_particles,n_dim)

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

