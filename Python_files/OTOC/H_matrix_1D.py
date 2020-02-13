#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 14:20:04 2020

@author: vgs23
"""


import numpy as np

"""
    This part of the code provides the necessary tools for computing
    OTOC for a 1D sytem. It has been well tested for 1D box and 
    Harmonic oscillator. One just has to be a bit careful when using
    singular potentials; The eigen value spectrum might contain some
    erroneous eigenvalues for such potentials.
"""    
global hbar,mass

def Kin_matrix_elt(x,dx,i,j,a,b,N):
        global hbar,mass
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

def Pot_matrix(x,dx,a,b,N,potential):
    Pot_mat = np.zeros((N-1,N-1))
    for i in range(0,N-1):
        for j in range(0,N-1):
            if(i==j):
                Pot_mat[i][j] = potential(x[i])
    return Pot_mat        
                
                
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
                
   