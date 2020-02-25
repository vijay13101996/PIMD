#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 14:23:48 2019

@author: vgs23
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh,eigs
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
#matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import multiprocessing as mp
import time
import pickle
import importlib
import OTOC_f
import cProfile
import sys
sys.path.insert(1,'/home/vgs23/Python_files/')
import Potentials_1D
import Potentials_2D
import H_matrix_1D
import H_matrix_2D
import H_diagonalization
import h5py
import Two_point_tcf

importlib.reload(OTOC_f)
start_time= time.time()

"""
gfortran -c OTOC_fortran.f90
f2py -c --f90flags="-O3" -m OTOC_f OTOC_fortran.f90

These are the command line codes to compile and wrap the relevant FORTRAN functions for computing the OTOC.
"""
L = 3*np.pi
N = 50
dx = L/N
dy= L/N
a = -L
b = L
x= np.linspace(a,b,N+1)
y= np.linspace(a,b,N+1)
hbar = 1.0
mass = 1.0#0.5

#--------------------------------------------------------
    
if(0):
        """
        This part of the code computes the eigenvalues and eigenvectors
        for a given 1D Hamiltonian
        """
        H_matrix_1D.hbar=hbar
        H_matrix_1D.mass=mass
        
        vecs = np.zeros((N-1,N-1))
        vals = np.zeros(N-1)
        
        H_diagonalization.Diagonalize_1D(x,dx,a,b,N,Potentials_1D.potential_quartic,vals,vecs)
        print('vals',vals[:30])
        if(1):
            """
            This part of the code is meant to weed out unphysical 
            eigenvalues, which mostly occurs for singular potentials
            like that of 1D box.
            
            Update: Code is fixed to ensure that there are no unphysical
            eigenvalues
            """
        
            print(np.sum(vecs[:,0]**2*dx))
            plt.figure(1)
            plt.plot(x[1:len(x)-1],vecs[:,0])
            plt.show()
#-----------------------------------------------------
"""
This part of the code contains the necessary tools to compute the 
OTOC for a 2D system. 
"""

potential_2D = Potentials_2D.potential_coupled_quartic
pot_key = 'Stadium_billiards'#'Coupled_quartic'#'Quartic'#'Harmonic'#

if(1):
    #wf = eigenstate(vecs[:,26])
    #print('norm',np.sum(vecs[:,2]*vecs[:,2]*dx*dy))
    #print('vals',vals[:40])
    #plt.imshow(wf.T,origin='lower')
    #plt.show()
    
    X,Y =np.meshgrid(x,y)
    potential_2D = np.vectorize(potential_2D)
    
    plt.imshow(potential_2D(X,Y))
    plt.show()
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, potential_2D(X,Y))
    plt.show()
    #print(E_psi[400],vecs[400,1])
    
    #plt.plot(x,vecs[:,4])
    #plt.show()

if(1):
    #H_diagonalization.Diagonalize_2D(x,dx,y,dy,a,b,N,potential_2D,vals,vecs)
    H_matrix_1D.hbar=hbar
    H_matrix_1D.mass=mass
    vecs = np.zeros(((N+1)**2,(N+1)**2))
    vals = np.zeros((N+1)**2)
    H_diagonalization.Diagonalize_2D(x,dx,y,dy,a,b,N,potential_2D,vals,vecs)
    
    print(np.shape(vecs),np.shape(vals))
    print('vals first few',vals[:40])
    
#    f = open("/home/vgs23/Pickle_files/Eigen_basis_{}_grid_N_{}_Ls_{}_rc_{}_n_barrier_{}.dat".format(pot_key,N,Potentials_2D.Ls,Potentials_2D.r_c,Potentials_2D.n_barrier),'wb')        
#    pickle.dump(vecs,f)
#    pickle.dump(vals,f)
#    f.close()
    
    h5f = h5py.File("/home/vgs23/Pickle_files/Eigen_basis_{}_grid_N_{}_Ls_{}_rc_{}_n_barrier_{}.h5".format(pot_key,N,Potentials_2D.Ls,Potentials_2D.r_c,Potentials_2D.n_barrier), 'w')
    h5f.create_dataset('vecs',data=vecs)
    h5f.create_dataset('vals',data=vals)
    h5f.close()


    def eigenstate(vector):
        temp = H_matrix_2D.tuple_index(N)
        wf = np.zeros((N+1,N+1))
        for p in range(len(temp)):
            i = temp[p][0]
            j = temp[p][1]
            wf[i][j] = vector[p]
            
        return wf
    

#------------------------------------------------
print('time 1',time.time()-start_time)
   
if(1):
    
    h5f = h5py.File("/home/vgs23/Pickle_files/Eigen_basis_{}_grid_N_{}_Ls_{}_rc_{}_n_barrier_{}.h5".format(pot_key,N,Potentials_2D.Ls,Potentials_2D.r_c,Potentials_2D.n_barrier), 'r')
    vecs = h5f['vecs'][:]
    vals = h5f['vals'][:]
    h5f.close()
    if(1):
        temp = H_matrix_2D.tuple_index(N)
        x_arr = np.zeros(len(temp))
        for p in range(len(temp)):
            i = temp[p][0]
            x_arr[p] = x[i]
    
    tarr = np.linspace(0,20,100)
    tcf_tarr = np.zeros_like(tarr)
    print('first',Two_point_tcf.two_point_pos_tcf(vals,vecs,x_arr,dx,dy,1,0.0,50))
    for i in range(len(tarr)):
        tcf_tarr[i] = Two_point_tcf.two_point_pos_tcf(vals,vecs,x_arr,dx,dy,1,tarr[i],50)
    plt.plot(tarr,tcf_tarr)
    plt.show()
    
if(0):   
    
    """
    This part of the code shall remain the same, even when other
    forms of time correlation functions are computed. The code 
    following this is specifically meant for OTOC.
    
    """
    
    basis_N = 30
    n_eigen = 30
    basis_x = np.zeros((N,N))
#    f = open("/home/vgs23/Pickle_files/Eigen_basis_{}_grid_N_{}_Ls_{}_rc_{}_n_barrier_{}.dat".format(pot_key,N,Potentials_2D.Ls,Potentials_2D.r_c,Potentials_2D.n_barrier),'rb')   
#    vecs = pickle.load(f)
#    vals = pickle.load(f)
#    f.close()
    
    h5f = h5py.File("/home/vgs23/Pickle_files/Eigen_basis_{}_grid_N_{}_Ls_{}_rc_{}_n_barrier_{}.h5".format(pot_key,N,Potentials_2D.Ls,Potentials_2D.r_c,Potentials_2D.n_barrier), 'r')
    vecs = h5f['vecs'][:]
    vals = h5f['vals'][:]
    h5f.close()
    
    vecs = vecs[:,:40]
    vals = vals[:40]
    #x_arr = x[1:len(x)-1]
    """
    The code below reassigns the position coordinates of a 2D system 
    on a 1D grid. This is the array to be passed for obtaining the 
    position matrix elements of a 2D system. 
    """
    if(1):
        temp = H_matrix_2D.tuple_index(N)
        x_arr = np.zeros(len(temp))
        for p in range(len(temp)):
            i = temp[p][0]
            x_arr[p] = x[i]
        
    k_arr = np.arange(basis_N)
    m_arr = np.arange(basis_N)
    
    if(1):   
        
        """
        This part of the code computes the OTOC for various potentials. 
        Except for the way in which the position matrix elements are 
        calculated, the code is the same for 1D and 2D. 
        
        Input:
            basis_N : Number of eigenvectors used to compute the 
                      microcanonical OTOC
            n_eigen : Number of energy levels used to compute the 
                      thermal OTOC.
            
        
        The code is currently very slow and has to be optimized using
        embedded FORTRAN code. 
        
        UPDATE: This code has been made atleast 10-15 times faster, by using 
        FORTRAN wrapped functions instead of numpy vectorized functions.
        """

        beta = 1.0/100
        t_final = 200.0
        t_arr = np.arange(0,t_final,0.5)
        print('beta,n_basis,n_eigen',beta,basis_N,n_eigen)
        
        #c_mc_arr = np.zeros_like(t_arr)
        #c_mc_arr = OTOC_f.position_matrix.compute_c_mc_arr_t(vecs,x_arr,dx,dy,k_arr,vals,0,m_arr,t_arr,c_mc_arr)
        OTOC_arr =np.zeros_like(t_arr)
        
        if(1):
            print('vals,vecs',np.shape(vecs),np.shape(vals))
            OTOC_arr = OTOC_f.position_matrix.compute_otoc_arr_t(vecs,x_arr,dx,dy,k_arr,vals,m_arr,t_arr,beta,n_eigen,OTOC_arr) 
            #OTOC_arr = OTOC(t_arr,beta)
            print('Pot details',Potentials_2D.n_barrier,Potentials_2D.r_c)
            f = open("/home/vgs23/Pickle_files/OTOC_{}_n_barrier_{}_r_c_{}_beta_{}_basis_{}_n_eigen_{}_tfinal_{}.dat".format(pot_key,Potentials_2D.n_barrier,Potentials_2D.r_c,beta,basis_N,n_eigen,t_final),'wb')
            pickle.dump(t_arr,f)
            pickle.dump(OTOC_arr,f)
            f.close()
        
        print('time 2',time.time()-start_time)
        
        fig = plt.figure(1)
        ax = fig.add_subplot(1,1,1)
        plt.yscale('log')
        #ax.set_ylim([0.1,100])
        
        print('Pot details',Potentials_2D.n_barrier,Potentials_2D.r_c)
        f = open("/home/vgs23/Pickle_files/OTOC_{}_n_barrier_{}_r_c_{}_beta_{}_basis_{}_n_eigen_{}_tfinal_{}.dat".format(pot_key,Potentials_2D.n_barrier,Potentials_2D.r_c,beta,basis_N,n_eigen,t_final),'rb')
        t_arr = pickle.load(f)
        OTOC_arr = pickle.load(f)
        f.close()
        
        print('OTOC_arr',OTOC_arr)
        ax.plot(t_arr,OTOC_arr,color='b')
        
        if(0):
            f = open("/home/vgs23/Pickle_files/OTOC_{}_n_barrier_{}_r_c_{}_beta_{}_basis_{}_n_eigen_{}_tfinal_{}.dat".format(pot_key,9,Potentials_2D.r_c,beta,30,30,t_final),'rb')
            t_arr = pickle.load(f)
            OTOC_arr = pickle.load(f)
            f.close()
            
            #ax.plot(t_arr,OTOC_arr,color='m')
            
            f = open("/home/vgs23/Pickle_files/OTOC_{}_n_barrier_{}_r_c_{}_beta_{}_basis_{}_n_eigen_{}_tfinal_{}.dat".format(pot_key,25,Potentials_2D.r_c,beta,30,30,t_final),'rb')
            t_arr = pickle.load(f)
            OTOC_arr = pickle.load(f)
            f.close()
        
            #ax.plot(t_arr,OTOC_arr,color='g')
        plt.show()
        if(0):
            fig = plt.figure(2)
            ax = fig.add_subplot(1,1,1)
            plt.yscale('log')
            ax.set_ylim([0.1,100])
            
            f = open("/home/vgs23/Pickle_files/OTOC_stadium_billiards_beta_{}_basis_{}_n_eigen_{}_tfinal_{}.dat".format(beta,80,80,t_final),'rb')
            t_arr = pickle.load(f)
            OTOC_arr1 = pickle.load(f)
            f.close()
            
            #ax.plot(t_arr,(OTOC_arr1),color='g',label='80,80')
            
            f = open("/home/vgs23/Pickle_files/OTOC_stadium_billiards_beta_{}_basis_{}_n_eigen_{}_tfinal_{}.dat".format(beta,90,90,t_final),'rb')
            t_arr = pickle.load(f)
            OTOC_arr2 = pickle.load(f)
            f.close()
            
            #ax.plot(t_arr,OTOC_arr2,color='r',label='90,90')
            
            f = open("/home/vgs23/Pickle_files/OTOC_stadium_billiards_beta_{}_basis_{}_n_eigen_{}_tfinal_{}.dat".format(beta,100,100,t_final),'rb')
            t_arr = pickle.load(f)
            OTOC_arr3 = pickle.load(f)
            f.close()
            
            ax.plot(t_arr,(OTOC_arr3),color='b',label='100,100')
            
            f = open("/home/vgs23/Pickle_files/OTOC_stadium_billiards_beta_{}_basis_{}_n_eigen_{}_tfinal_{}.dat".format(beta,50,40,t_final),'rb')
            t_arr = pickle.load(f)
            OTOC_arr4 = pickle.load(f)
            f.close()
            
            #ax.plot(t_arr,(OTOC_arr4),color='m',label='50,40')
            
            f = open("/home/vgs23/Pickle_files/OTOC_stadium_billiards_beta_{}_basis_{}_n_eigen_{}_tfinal_{}.dat".format(beta,50,100,t_final),'rb')
            t_arr = pickle.load(f)
            OTOC_arr5 = pickle.load(f)
            f.close()
            
            #ax.plot(t_arr,(OTOC_arr5),color='c',label='50,100')
            plt.title('OTOC for Stadium billiards at 400K')
            ax.legend()
            plt.show()

        
