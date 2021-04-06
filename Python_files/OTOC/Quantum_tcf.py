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
#import OTOC_f
import OTOC_f_1D
import OTOC_f_2D
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
import imp
from matplotlib import rc
from matplotlib.ticker import ScalarFormatter

imp.reload(OTOC_f_1D)
start_time= time.time()

"""
gfortran -c OTOC_fortran.f90
f2py -c --f90flags="-O3" -m OTOC_f OTOC_fortran.f90

These are the command line codes to compile and wrap the relevant FORTRAN functions for computing the OTOC.
"""

"""
Ensure that the OTOC normalized is adjusted for 1D systems
"""


L = 10.0
N = 100
a = -L
b = L
x= np.linspace(a,b,N+1)
y= np.linspace(0.5*a,0.5*b,N+1)        #### CHANGE the grid according to the potential, when computing OTOC
dx = x[1]-x[0]
dy= y[1]-y[0]
print('dx dy', dx, dy)
hbar = 1.0
mass = 0.5

plot_potential = 0
Diagonalize_2D = 0
#--------------------------------------------------------
    
if(1):
        """
        This part of the code computes the eigenvalues and eigenvectors
        for a given 1D Hamiltonian
        """
        H_matrix_1D.hbar=hbar
        H_matrix_1D.mass=mass
        
        vecs = np.zeros((N-1,N-1))
        vals = np.zeros(N-1)
        
        H_diagonalization.Diagonalize_1D(x,dx,a,b,N,Potentials_1D.potential_inv_harmonic,vals,vecs)
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

potential_2D = Potentials_2D.potential_coupled_harmonic
print('pot', potential_2D(2,2))
pot_key = 'Coupled_harmonic'#'Coupled_harmonic'#'Quartic'#'Harmonic'#'Stadium_billiards'#

if(plot_potential==1):
    #wf = eigenstate(vecs[:,26])
    #print('norm',np.sum(vecs[:,2]*vecs[:,2]*dx*dy))
    #print('vals',vals[:40])
    #plt.imshow(wf.T,origin='lower')
    #plt.show()
    
    X,Y =np.meshgrid(x,y)
    potential_2D = np.vectorize(potential_2D)
   
    plt.title('{}'.format(pot_key))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.imshow(potential_2D(X,Y),origin='lower',extent=[x[0],x[-1],y[0],y[-1]])
    plt.show()
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, potential_2D(X,Y))
    plt.show()
    #print(E_psi[400],vecs[400,1])
    
    #plt.plot(x,vecs[:,4])
    #plt.show()

if(Diagonalize_2D==1):
    #H_diagonalization.Diagonalize_2D(x,dx,y,dy,a,b,N,potential_2D,vals,vecs)
    H_matrix_1D.hbar=hbar
    H_matrix_1D.mass=mass
    vecs = np.zeros(((N+1)**2,(N+1)**2))
    vals = np.zeros((N+1)**2)
    H_diagonalization.Diagonalize_2D(x,dx,y,dy,a,b,N,potential_2D,vals,vecs)
    
    print(np.shape(vecs),np.shape(vals))
    print('vals first few',vals[:40])
    
    f = open("/home/vgs23/Pickle_files/Eigen_basis_{}_grid_N_{}.dat".format(pot_key,N),'wb')
    #open("/home/vgs23/Pickle_files/Eigen_basis_{}_grid_N_{}_Ls_{}_rc_{}_n_barrier_{}.dat".format(pot_key,N,Potentials_2D.Ls,Potentials_2D.r_c,Potentials_2D.n_barrier),'wb')        
    pickle.dump(vecs,f)
    pickle.dump(vals,f)
    f.close()
    
    h5f = h5py.File("/home/vgs23/Pickle_files/Eigen_basis_{}_grid_N_{}.h5".format(pot_key,N), 'w')
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
   
if(0):
    """
    This part of the code is meant for computing the 2 point
    quantum time correlation functions using DVR
    
    """

    #h5f = h5py.File("/home/vgs23/Pickle_files/Eigen_basis_{}_grid_N_{}_Ls_{}_rc_{}_n_barrier_{}.h5".format(pot_key,N,Potentials_2D.Ls,Potentials_2D.r_c,Potentials_2D.n_barrier), 'r')
    h5f = h5py.File("/home/vgs23/Pickle_files/Eigen_basis_{}_grid_N_{}.h5".format(pot_key,N), 'r')
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
    tcf_tarr = np.zeros_like(tarr) + 0j
    print('first',Two_point_tcf.two_point_pos_tcf(vals,vecs,x_arr,dx,dy,0.2,0.0,100))
    for i in range(len(tarr)):
        tcf_tarr[i] = Two_point_tcf.two_point_pos_tcf(vals,vecs,x_arr,dx,dy,0.2,tarr[i],100)
    plt.plot(tarr,np.real(tcf_tarr))#, np.imag(tcf_tarr))
    plt.show()
    
if(1):   
    
    """
    This part of the code shall remain the same, even when other
    forms of time correlation functions are computed. The code 
    following this is specifically meant for OTOC.
    
    """
    
    basis_N = 50
    n_eigen = 50
    basis_x = np.zeros((N,N))
    if(0):
        f = open("/home/vgs23/Pickle_files/Eigen_basis_{}_grid_N_{}.dat".format(pot_key,N),'rb')
        #open("/home/vgs23/Pickle_files/Eigen_basis_{}_grid_N_{}_Ls_{}_rc_{}_n_barrier_{}.dat".format(pot_key,N,Potentials_2D.Ls,Potentials_2D.r_c,Potentials_2D.n_barrier),'rb')   
        vecs = pickle.load(f)
        vals = pickle.load(f)
        print('len vals', len(vals))
        f.close()
   
    if(0):       ## Toggle for 2D potentials
        h5f = h5py.File("/home/vgs23/Pickle_files/Eigen_basis_{}_grid_N_{}.h5".format(pot_key,N), 'r')
        vecs = h5f['vecs'][:]
        vals = h5f['vals'][:]
        h5f.close()
    
    if(0):
        vecs = vecs[:,:20]
        vals = vals[:20]

    x_arr = x[1:len(x)-1]
    #print('xarr', np.shape(x_arr), np.shape(vecs)) 
    #plt.imshow(eigenstate(vecs[:,4]))
    #plt.show()

    print('normalized?', np.sum(vecs[:,0]**2*dx))#*dy))

    """
    The code below reassigns the position coordinates of a 2D system 
    on a 1D grid. This is the array to be passed for obtaining the 
    position matrix elements of a 2D system. 
    """
    if(0):
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
        t_final = 4.0
        t_arr = np.arange(0.0,t_final,0.001)
         
        #c_mc_arr = np.zeros_like(t_arr)
        #c_mc_arr = OTOC_f.position_matrix.compute_c_mc_arr_t(vecs,x_arr,dx,dy,k_arr,vals,0,m_arr,t_arr,c_mc_arr)
        OTOC_arr =np.zeros_like(t_arr)
        
        beta_arr = 1.0/np.array([1,2,3,4,5])   ### This is for Coupled_harmonic
        beta_arr = 1.0/np.array([1,40,200,400])### This is for Stadium billiards
        beta_arr = 1.0/np.array([1,2,4,10])   ### This is for Coupled_quartic
        beta_arr = 1.0/np.array([1,20,40,100]) ### This is for 1D_Box
        beta_arr = [1.0/5]      
        print('beta_arr,n_basis,n_eigen',beta_arr,basis_N,n_eigen)
        print('Boltzmann factor', np.exp(-beta_arr[0]*vals[n_eigen-1]), np.exp(-beta_arr[0]*vals[0]))

        if(1):   
            for betaa in beta_arr: 
                print('vals,vecs',np.shape(vecs),np.shape(vals))
                OTOC_arr = OTOC_f_1D.position_matrix.compute_otoc_arr_t(vecs,x_arr,dx,dy,k_arr,vals,m_arr,t_arr,betaa,n_eigen,OTOC_arr) 
                #OTOC_arr = OTOC.OTOC(t_arr,beta)
                f =  open("/home/vgs23/Pickle_files/OTOC_{}_beta_{}_basis_{}_n_eigen_{}_tfinal_{}.dat".format('inv_harmonic',betaa,basis_N,n_eigen,t_final),'wb')
                #f = open("/home/vgs23/Pickle_files/Kubo_OTOC_{}_beta_{}_basis_{}_n_eigen_{}_tfinal_{}.dat".format(pot_key,betaa,basis_N,n_eigen,t_final),'wb')
                pickle.dump(t_arr,f)
                pickle.dump(OTOC_arr,f)
                plt.plot(t_arr,np.log(OTOC_arr))
                plt.show()
                f.close()
        
        print('time 2',time.time()-start_time)
      
        if(0):
		rc('text', usetex=True)
		rc('font', family='serif')
		matplotlib.rcParams.update({'font.size': 17})
		fig = plt.figure(1)
		ax = fig.add_subplot(1,1,1)
		#plt.yscale('log')
		#ax.set_ylim([0.005,500])
		#ax.text(20,30, r'$\beta = 1/1000$')
		plt.xlabel(r'$t$')
		plt.ylabel(r'$C(t)$')
		plt.gcf().subplots_adjust(bottom=0.12,left=0.12)
		for axis in [ax.xaxis, ax.yaxis]:
		    axis.set_major_formatter(ScalarFormatter())
		#plt.title('OTOC for 1D Box')
				
		print('Pot details',Potentials_2D.r_c)
		color_arr = ['r','g','b','m','c']
		label_arr = [r'$\beta=1$',r'$\beta=\frac{1}{40}$',r'$\beta=\frac{1}{200}$', r'$\beta=\frac{1}{400}$'] # 'c', 'd']
		label_arr = [r'$T=1$', r'$T=20$', r'$T=40$', r'$T=100$']#,r'$T=5$'] 
		count = 0
		for (beta,colour,lab) in zip(beta_arr,color_arr,label_arr): 
		    f= open("/home/vgs23/Pickle_files/Kubo_OTOC_{}_beta_{}_basis_{}_n_eigen_{}_tfinal_{}.dat".format('1D_Box',beta,basis_N,n_eigen,t_final),'rb')
		    #f = open("/home/vgs23/Pickle_files/Kubo_OTOC_{}_beta_{}_basis_{}_n_eigen_{}_tfinal_{}.dat".format(pot_key,beta,basis_N,n_eigen,t_final),'rb')
		    #f = open("/home/vgs23/Pickle_files/Kubo_OTOC_{}_r_c_{}_beta_{}_basis_{}_n_eigen_{}_tfinal_{}.dat".format(pot_key,Potentials_2D.r_c,beta,50,50,3.0),'rb') 
		    t_arr = pickle.load(f)
		    OTOC_arr = pickle.load(f)
		    #print(OTOC_arr)
		    f.close()
		    print('beta','colour', beta, colour) 
		    #print('OTOC_arr',OTOC_arr)
		    ax.plot(t_arr,OTOC_arr,color=colour,label=lab)
	       
        if(0):
            f = open("/home/vgs23/Pickle_files/Kubo_OTOC_{}_beta_{}_basis_{}_n_eigen_{}_tfinal_{}.dat".format(pot_key,beta,basis_N,n_eigen,t_final),'rb')
            #f = open("/home/vgs23/Pickle_files/Kubo_OTOC_{}_r_c_{}_beta_{}_basis_{}_n_eigen_{}_tfinal_{}.dat".format(pot_key,Potentials_2D.r_c,beta,50,50,3.0),'rb') 
            t_arr = pickle.load(f)
            OTOC_arr = pickle.load(f)
            #print(OTOC_arr)
            f.close()
            print('beta','colour', beta, colour) 
            #print('OTOC_arr',OTOC_arr)
            ax.plot(t_arr,OTOC_arr,color=colour,label=lab)

        if(0):

            f = open("/home/vgs23/Pickle_files/OTOC_{}_n_barrier_{}_r_c_{}_beta_{}_basis_{}_n_eigen_{}_tfinal_{}.dat".format(pot_key,Potentials_2D.n_barrier,Potentials_2D.r_c,beta,50,50,t_final),'rb')
            t_arr = pickle.load(f)
            OTOC_arr = pickle.load(f)
            f.close()
            
            #print('OTOC_arr',OTOC_arr)
            ax.plot(t_arr,OTOC_arr,color='g')

            f = open("/home/vgs23/Pickle_files/OTOC_{}_n_barrier_{}_r_c_{}_beta_{}_basis_{}_n_eigen_{}_tfinal_{}.dat".format(pot_key,Potentials_2D.n_barrier,Potentials_2D.r_c,beta,30,30,t_final),'rb')
            t_arr = pickle.load(f)
            OTOC_arr = pickle.load(f)
            f.close()
            
            ax.plot(t_arr,OTOC_arr,color='m')
            
            #f = open("/home/vgs23/Pickle_files/OTOC_{}_n_barrier_{}_r_c_{}_beta_{}_basis_{}_n_eigen_{}_tfinal_{}.dat".format(pot_key,Potentials_2D,Potentials_2D.r_c,beta,30,30,t_final),'rb')
            #t_arr = pickle.load(f)
            #OTOC_arr = pickle.load(f)
            #f.close()
        
            #ax.plot(t_arr,OTOC_arr,color='g')
        ax.legend(loc=1)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
          fancybox=False, shadow=False, ncol=4,prop={'size': 10})
        
        print(ax.get_xticks()) 
        if(0):  # For coupled harmonic oscillator
            xt = ax.xaxis.get_majorticklocs()
            print(xt)
            xt=np.append(xt,6.5)

            xtl=xt.tolist()
            xtl[-1]=r'$t_E$'
            print(xtl)
            ax.set_xticks(xt)
            ax.set_xticklabels(xtl)
            ax.axvline(x=6.5,ymin=0.0, ymax = 0.7,linestyle='--')
            ax.set_xlim([-0.5,15.5])
        
        if(0):  # For 1D box
            xt = ax.xaxis.get_majorticklocs()
            print(xt)
            xt=np.append(xt,1/np.pi)

            xtl=xt.tolist()
            xtl[-1]=r'$\frac{1}{\pi}$'
            print(xtl)
            ax.set_xticks(xt)
            ax.set_xticklabels(xtl)
            ax.axvline(x=1/np.pi,ymin=0.0, ymax = 0.05,linestyle='--')
            ax.set_xlim([0.0,1.0])
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
        
            plt.savefig('/home/vgs23/Images/OTOC_1DBOX.eps', format='eps',dpi=96)
            plt.show()
        
        
        
        if(0):
            fig = plt.figure(2)
            ax = fig.add_subplot(1,1,1)
            plt.yscale('log')
            ax.set_ylim([0.1,100])
            beta = 1/400.0
            
            f = open("/home/vgs23/Pickle_files/OTOC_stadium_billiards_beta_{}_basis_{}_n_eigen_{}_tfinal_{}.dat".format(beta,80,80,t_final),'rb')
            t_arr = pickle.load(f)
            OTOC_arr1 = pickle.load(f)
            f.close()
            
            ax.plot(t_arr,(OTOC_arr1),color='g',label='80,80')
            
            f = open("/home/vgs23/Pickle_files/OTOC_stadium_billiards_beta_{}_basis_{}_n_eigen_{}_tfinal_{}.dat".format(beta,90,90,t_final),'rb')
            t_arr = pickle.load(f)
            OTOC_arr2 = pickle.load(f)
            f.close()
            
            ax.plot(t_arr,OTOC_arr2,color='r',label='90,90')
            
            f = open("/home/vgs23/Pickle_files/OTOC_stadium_billiards_beta_{}_basis_{}_n_eigen_{}_tfinal_{}.dat".format(beta,100,100,t_final),'rb')
            t_arr = pickle.load(f)
            OTOC_arr3 = pickle.load(f)
            f.close()
            
            ax.plot(t_arr,(OTOC_arr3),color='b',label='100,100')
            
            f = open("/home/vgs23/Pickle_files/OTOC_stadium_billiards_beta_{}_basis_{}_n_eigen_{}_tfinal_{}.dat".format(beta,50,40,t_final),'rb')
            t_arr = pickle.load(f)
            OTOC_arr4 = pickle.load(f)
            f.close()
            
            ax.plot(t_arr,(OTOC_arr4),color='m',label='50,40')
            
            f = open("/home/vgs23/Pickle_files/OTOC_stadium_billiards_beta_{}_basis_{}_n_eigen_{}_tfinal_{}.dat".format(beta,50,100,t_final),'rb')
            t_arr = pickle.load(f)
            OTOC_arr5 = pickle.load(f)
            f.close()
            
            ax.plot(t_arr,(OTOC_arr5),color='c',label='50,100')
            plt.title('OTOC for Stadium billiards at 400K')
            ax.legend()
            plt.show()

        
