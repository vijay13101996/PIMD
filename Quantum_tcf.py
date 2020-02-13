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
from matplotlib.ticker import NullFormatter
importlib.reload(OTOC_f)
start_time= time.time()

"""
gfortran -c OTOC_fortran.f90
f2py -c --f90flags="-O3" -m OTOC_f OTOC_fortran.f90

These are the command line codes to compile and wrap the relevant FORTRAN functions for computing the OTOC.
"""
L = 12#3*np.pi
N = 100
dx = L/N
dy= L/N
a = -L
b = L 
x= np.linspace(a,b,N+1)
y= np.linspace(a,b,N+1)
hbar = 1.0
mass = 0.5

"""
This part of the code below was a primitive basis set, that was used
to diagonalize the Hamiltonian. It is kept in the module just for 
future reference. 
"""
#T = -0.5*diags([-2., 1., 1.], [0,-1, 1], shape=(N, N))/dx**2
#U_vec = 0.0*x#0.5*x**2
#U = diags([U_vec], [0])
#
#H= T+U
#vals, vecs = eigsh(H,k=10, which='SM')
#
#vecs*=-1.0
##print(np.round(vals, 6))
##print(np.round([(n*np.pi/L)**2/2 for n in range(1,6)], 6))
#
##print(vals[:10])
#plt.plot(x,-1*vecs[:,0])
#print(np.sum(vecs[:,4]*vecs[:,4]*dx))
#
#grid_N= 10
#grid_x = np.zeros((N,N))
#
#def pos_matrix_elts(n,k):
#    return np.sum(vecs[:,n]*vecs[:,k]*x*dx)
#
#print(vals)
#print(pos_matrix_elts(2,1))

#--------------------------------------------------------

if(1):
    
    """
    This part of the code provides the necessary tools for computing
    OTOC for a 1D sytem. It has been well tested for 1D box and 
    Harmonic oscillator. One just has to be a bit careful when using
    singular potentials; The eigen value spectrum might contain some
    erroneous eigenvalues for such potentials.
    """
    
    def potential(x):
        if(x<0 or x>L):
            return 1e5
        else:
            return 0.0#1/(1+np.exp(k*x)) + 1/(1+np.exp(-k*(x-L)))#0.0#0.5*x**2
    
    def Kin_matrix_elt(x,dx,i,j,a,b,N):
        const = (hbar**2/(2*mass))
        prefactor = (np.pi**2/2)*( (-1)**(i-j)/(b-a)**2 )
        
        if(i!=j):
            sin_term1 = 1/(np.sin(np.pi*(i-j)/(2*N)))**2
            sin_term2 = 1/(np.sin(np.pi*(i+j)/(2*N)))**2
            
            return const*prefactor*(sin_term1 - sin_term2)
            #return const*(1/dx**2)*(-1)**(i-j)*2/(i-j)**2
        else:
            if(i!=0 and i!=N):
                const_term = (2*N**2 +1)/3
                sin_term3 = 1/(np.sin(np.pi*i/N))**2
                
                return const*prefactor*(const_term - sin_term3)
                #return const*(1/dx**2)*np.pi**2/3
            else:
                return 0.0#const*(1/dx**2)*np.pi**2/3#
    
    def Kin_matrix(x,dx,a,b,N):
        #Kin_mat = np.zeros((N+1,N+1))
        Kin_mat = np.zeros((N-1,N-1))
        for i in range(0,N-1):
            for j in range(0,N-1):
                    Kin_mat[i][j] = Kin_matrix_elt(x,dx,i+1,j+1,a,b,N)
                    #print('1d',i,j,Kin_mat[i][j])
        return Kin_mat
    
    def Pot_matrix(x,dx,a,b,N):
        Pot_mat = np.zeros((N-1,N-1))
        for i in range(0,N-1):
            for j in range(0,N-1):
                if(i==j):
                    Pot_mat[i][j] = potential(x[i])
                    
        return Pot_mat
    
    if(0):
        T = Kin_matrix(x,dx,a,b,N)
        V = Pot_matrix(x,dx,a,b,N)
        
        #print(T)
        H = T+V
        vals, vecs = np.linalg.eigh(H)#,k=6, which='SM')
        
        
        """
        The eigen vectors obtained from the above equation is not 
        necessarily normalized, so they are explicitly normalized in
        the lines below. 
        """
        norm = 1/(np.sum(vecs[:,2]**2*dx))**0.5
        vecs*=norm
        
        #plt.plot(x,potential(x)*np.ones_like(x))
        #plt.show()
        if(1):
            """
            This part of the code is meant to weed out unphysical 
            eigenvalues, which mostly occurs for singular potentials
            like that of 1D box.
            """
            #vals = vals[2:]
            #vecs = np.delete(vecs,0,axis=1)
            #vecs = np.delete(vecs,0,axis=1)
        
            print(np.sum(vecs[:,0]**2*dx))
            plt.figure(1)
            plt.plot(x[1:len(x)-1],vecs[:,0])
            plt.show()
#------------------------------------------------
"""
This part of the code contains the necessary tools to compute the 
OTOC for a 2D system. 
"""


Le = 2*(1/(np.pi+4)**0.5)  
s= 20.0
r_c = 1/(np.pi+4)**0.5
K = 10.0
print('Area',np.pi*r_c**2 + Le*2*r_c)


Ls = 10.0
r_c = 1.0
n_barrier = 0
"""
The potential below is the 'smooth' stadium billiards potential that
is generically used for understanding quantum chaos. Here,

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

potential_2D = potential_lg
pot_key = 'Lorenz_gas'#'Quartic'#

#-----------------------------------------------------
 
def Kin_matrix_2D_elt(x,dx,y,dy,i,i_d,j,j_d,a,b,N):
    if(j==j_d and i!=i_d):
        return Kin_matrix_elt(x,dx,i,i_d,a,b,N)
    elif(i==i_d and j!=j_d):
        #print('hereh',Kin_matrix_elt(y,dy,j,j_d,a,b,N),i,j,j_d)
        return Kin_matrix_elt(y,dy,j,j_d,a,b,N)
    elif(i==i_d and j==j_d):
        return Kin_matrix_elt(x,dx,i,i_d,a,b,N) + Kin_matrix_elt(y,dy,j,j_d,a,b,N)
    else:
        return 0.0
    
def Pot_matrix_2D(x,dx,y,dy,a,b,N):
    Pot_mat = np.zeros((N+1,N+1,N+1,N+1))
    for i in range(0,N+1):
        for i_d in range(0,N+1):
            for j in range(0,N+1):
                for j_d in range(0,N+1):
                    if(i==i_d and j==j_d):
                        Pot_mat[i][i_d][j][j_d] = potential_2D(x[i],y[j])
    return Pot_mat

def Kin_matrix_2D(x,dx,y,dy,a,b,N):
    Kin_mat = np.zeros((N+1,N+1,N+1,N+1))
    for i in range(0,N+1):
        for i_d in range(0,N+1):
            for j in range(0,N+1):
                for j_d in range(0,N+1):
                        Kin_mat[i][i_d][j][j_d] =Kin_matrix_2D_elt(x,dx,y,dy,i,i_d,j,j_d,a,b,N)
    return Kin_mat

def tuple_index(N):
    temp = []#np.zeros((N**2,2),dtype='int')
    count = 0
    for i in range(N+1):
        for j in range(N+1):
            temp.append((i,j))
            count+=1
    
    return temp

def Kinetic_matrix_2D(x,dx,y,dy,a,b,N):
    
    tuple_arr = tuple_index(N)
    length = len(tuple_arr)
    Kin_mat = np.zeros((length,length))
    for p in range(length):
        for p_d in range(length):
            i = tuple_arr[p][0]
            j = tuple_arr[p][1]
            i_d = tuple_arr[p_d][0]
            j_d = tuple_arr[p_d][1]
            Kin_mat[p][p_d] = Kin_matrix_2D_elt(x,dx,y,dy,i,i_d,j,j_d,a,b,N)
            #if(i==i_d):
               # print(Kin_mat[p][p_d],i,j,j_d)
    return Kin_mat

def Potential_matrix_2D(x,dx,y,dy,a,b,N):
    
    tuple_arr = tuple_index(N)
    length = len(tuple_arr)
    Pot_mat = np.zeros((length,length))
    for p in range(length):
        for p_d in range(length):
            i = tuple_arr[p][0]
            j = tuple_arr[p][1]
            i_d = tuple_arr[p_d][0]
            j_d = tuple_arr[p_d][1]
            if(i==i_d and j==j_d):
                Pot_mat[p][p_d] = potential_2D(x[i],y[j])
    return Pot_mat

"""
The basis used for diagonalization is the same as above: DVR Basis. 
In order to ensure that the basic implementation is the same, the 
basis set is reordered into a 1D array, which can expanded into the 
full 2D set using the tuple_index function above.
"""

if(1):
    T = Kinetic_matrix_2D(x,dx,y,dy,a,b,N)
    V = Potential_matrix_2D(x,dx,y,dy,a,b,N)
    
    
    H = T+V
    vals, vecs = np.linalg.eigh(H)#eigsh(H,k=20, which='SM')
    norm = np.sum(vecs[:,17]**2*dx*dy)**0.5
    vecs/=norm
    
    
    f = open("Eigen_basis_{}_grid_N_{}_Ls_{}_rc_{}.dat".format(pot_key,N,Ls,r_c),'wb')
        
    pickle.dump(vecs,f)
    pickle.dump(vals,f)
    f.close()
    
    E_psi = np.matmul(H,vecs[:,1])  # NOTE: vecs[:,n] is the nth eigenvector
    
    def eigenstate(vector):
        temp = tuple_index(N)
        wf = np.zeros((N+1,N+1))
        for p in range(len(temp)):
            i = temp[p][0]
            j = temp[p][1]
            wf[i][j] = vector[p]
            
        return wf
    
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
#------------------------------------------------
print('time 1',time.time()-start_time)
   
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
    
    UPDATE: This code has made atleast 10-15 times faster, by using 
    FORTRAN wrapped functions instead of numpy vectorized functions.
    """
    basis_N = 30
    n_eigen = 30
    basis_x = np.zeros((N,N))
    f = open("Eigen_basis_{}_grid_N_{}_Ls_{}_rc_{}.dat".format(pot_key,N,Ls,r_c),'rb')   
    vecs = pickle.load(f)
    vals = pickle.load(f)
    f.close()
    
    #x_arr = x[1:len(x)-1]
    """
    The code below reassigns the position coordinates of a 2D system 
    on a 1D grid. This is the array to be passed for obtaining the 
    position matrix elements of a 2D system. 
    """
    if(1):
        temp = tuple_index(N)
        x_arr = np.zeros(len(temp))
        for p in range(len(temp)):
            i = temp[p][0]
            x_arr[p] = x[i]
        
    
    def pos_matrix_elts(n,k):
        return np.sum(vecs[:,n]*vecs[:,k]*x_arr*dx*dy)
        #return np.sum(vecs[:,n]*vecs[:,k]*x*dx)#np.sum(temp1*temp2*x*dx)#
    
    x_matrix = np.vectorize(pos_matrix_elts)
    
    pos_mat_elt = 0.0
    n_arr = np.arange(5)
    k_arr = np.arange(basis_N)
    pos_mat = np.zeros_like(k_arr)
    #cProfile.run('OTOC_f.position_matrix.compute_pos_mat_arr_k(vecs,x_arr,dx,dy,n,k_arr,pos_mat)')
    #cProfile.run('x_matrix(n,k_arr)')
    #print('Normalised?', np.sum(vecs[:,2]*vecs[:,2]*dx))
    #print('Norm:', np.sum(vecs[:,2]*vecs[:,2]*dx*dy))
    #tarr = np.arange(0,50,1e-1)
    
    def b_matrix(n,m,t):
        karr = np.arange(0,basis_N)
        E_km = vals[karr] - vals[m]
        E_nk = vals[n] - vals[karr]
        x_nk= x_matrix(n,karr)
        x_km = x_matrix(karr,m)
        return 0.5*np.sum(x_nk*x_km*(E_km*np.exp(1j*E_nk*t) - E_nk*np.exp(1j*E_km*t)))
    
    b_matrix = np.vectorize(b_matrix)
    #cProfile.run('OTOC_f.position_matrix.b_matrix_elts(vecs,x_arr,dx,dy,k_arr,vals,2,2,1.0,b_mat_elt)')
    #cProfile.run('b_matrix(2,2,1.0)')
    
    m_arr = np.arange(0,basis_N)
    b_mat = np.zeros_like(m_arr)
    #cProfile.run('OTOC_f.position_matrix.compute_b_mat_arr_m(vecs,x_arr,dx,dy,k_arr,vals,2,m_arr,1.0,b_mat)')
    #cProfile.run('b_matrix(2,m_arr,1.0)')

    def c_mc(n,t):
        marr = np.arange(0,basis_N)
        b_nm = b_matrix(n,marr,t)
        return np.sum(b_nm*np.conj(b_nm))
    
    c_mc = np.vectorize(c_mc)
    
    #t_arr = np.arange(0,5,0.1)
    #c_mc_elt = np.zeros_like(t_arr)
    #c_mc_elt = OTOC_f.position_matrix.compute_c_mc_arr_n(vecs,x_arr,dx,dy,k_arr,vals,np.arange(10),m_arr,3.0,c_mc_elt)
    #c_mc_elt = OTOC_f.position_matrix.compute_c_mc_arr_t(vecs,x_arr,dx,dy,k_arr,vals,1,m_arr,t_arr,c_mc_elt)
  
    #plt.plot(t_arr,c_mc_elt)
    #plt.show()

    def OTOC(t,beta):
        val = vals[:n_eigen]
        Z = np.sum(np.exp(-beta*val))
        n_arr = np.arange(0,len(val))
        return (1/Z)*np.sum(np.exp(-beta*val)*c_mc(n_arr,t)  )
    
    OTOC = np.vectorize(OTOC)

    #cProfile.run('OTOC_f.position_matrix.compute_otoc_arr_t(vecs,x_arr,dx,dy,k_arr,vals,m_arr,t_arr,0.001,n_eigen,OTOC_mat)')
    #cProfile.run('OTOC(t_arr,0.001)')
    if(1):
        beta = 1.0/100
        t_final = 200.0
        t_arr = np.arange(0,t_final,0.5)
        print('beta,n_basis,n_eigen',beta,basis_N,n_eigen)
        
        #c_mc_arr = np.zeros_like(t_arr)
        #c_mc_arr = OTOC_f.position_matrix.compute_c_mc_arr_t(vecs,x_arr,dx,dy,k_arr,vals,0,m_arr,t_arr,c_mc_arr)
        OTOC_arr =np.zeros_like(t_arr)
        
        if(1):
            OTOC_arr = OTOC_f.position_matrix.compute_otoc_arr_t(vecs,x_arr,dx,dy,k_arr,vals,m_arr,t_arr,beta,n_eigen,OTOC_arr) 
            #OTOC_arr = OTOC(t_arr,beta)
            print('Pot details',n_barrier,r_c)
            f = open("OTOC_{}_n_barrier_{}_r_c_{}_beta_{}_basis_{}_n_eigen_{}_tfinal_{}.dat".format(pot_key,n_barrier,r_c,beta,basis_N,n_eigen,t_final),'wb')
            pickle.dump(t_arr,f)
            pickle.dump(OTOC_arr,f)
            f.close()
        
        print('time 2',time.time()-start_time)
        
        fig = plt.figure(1)
        ax = fig.add_subplot(1,1,1)
        plt.yscale('log')
        ax.set_ylim([0.1,100])
        
        print('Pot details',n_barrier,r_c)
        f = open("OTOC_{}_n_barrier_{}_r_c_{}_beta_{}_basis_{}_n_eigen_{}_tfinal_{}.dat".format(pot_key,n_barrier,r_c,beta,30,30,t_final),'rb')
        t_arr = pickle.load(f)
        OTOC_arr = pickle.load(f)
        f.close()
        
        ax.plot(t_arr,OTOC_arr,color='b')
        
        if(0):
            f = open("OTOC_{}_n_barrier_{}_r_c_{}_beta_{}_basis_{}_n_eigen_{}_tfinal_{}.dat".format(pot_key,9,r_c,beta,30,30,t_final),'rb')
            t_arr = pickle.load(f)
            OTOC_arr = pickle.load(f)
            f.close()
            
            #ax.plot(t_arr,OTOC_arr,color='m')
            
            f = open("OTOC_{}_n_barrier_{}_r_c_{}_beta_{}_basis_{}_n_eigen_{}_tfinal_{}.dat".format(pot_key,25,r_c,beta,30,30,t_final),'rb')
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
            
            f = open("OTOC_stadium_billiards_beta_{}_basis_{}_n_eigen_{}_tfinal_{}.dat".format(beta,80,80,t_final),'rb')
            t_arr = pickle.load(f)
            OTOC_arr1 = pickle.load(f)
            f.close()
            
            #ax.plot(t_arr,(OTOC_arr1),color='g',label='80,80')
            
            f = open("OTOC_stadium_billiards_beta_{}_basis_{}_n_eigen_{}_tfinal_{}.dat".format(beta,90,90,t_final),'rb')
            t_arr = pickle.load(f)
            OTOC_arr2 = pickle.load(f)
            f.close()
            
            #ax.plot(t_arr,OTOC_arr2,color='r',label='90,90')
            
            f = open("OTOC_stadium_billiards_beta_{}_basis_{}_n_eigen_{}_tfinal_{}.dat".format(beta,100,100,t_final),'rb')
            t_arr = pickle.load(f)
            OTOC_arr3 = pickle.load(f)
            f.close()
            
            ax.plot(t_arr,(OTOC_arr3),color='b',label='100,100')
            
            f = open("OTOC_stadium_billiards_beta_{}_basis_{}_n_eigen_{}_tfinal_{}.dat".format(beta,50,40,t_final),'rb')
            t_arr = pickle.load(f)
            OTOC_arr4 = pickle.load(f)
            f.close()
            
            #ax.plot(t_arr,(OTOC_arr4),color='m',label='50,40')
            
            f = open("OTOC_stadium_billiards_beta_{}_basis_{}_n_eigen_{}_tfinal_{}.dat".format(beta,50,100,t_final),'rb')
            t_arr = pickle.load(f)
            OTOC_arr5 = pickle.load(f)
            f.close()
            
            #ax.plot(t_arr,(OTOC_arr5),color='c',label='50,100')
            plt.title('OTOC for Stadium billiards at 400K')
            ax.legend()
            plt.show()

        
