#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 14:45:48 2020

@author: vgs23
"""
import H_matrix_1D
import numpy as np

""" 
The function Diagonalize in this module computes the eigenvectors
and eigenvalues for the given Hamiltonian matrix

"""

def Diagonalize(x,dx,a,b,N,potential,vals,vecs):
    T = H_matrix_1D.Kin_matrix(x,dx,a,b,N)
    V = H_matrix_1D.Pot_matrix(x,dx,a,b,N,potential)

    H = T+V
    vals[:], vecs[:] = np.linalg.eigh(H)#,k=6, which='SM')
    
    """
    The eigen vectors obtained from the above equation is not 
    necessarily normalized, so they are explicitly normalized in
    the lines below. 
    """
    norm = 1/(np.sum(vecs[:,2]**2*dx))**0.5
    vecs[:]*=norm
    
    #plt.plot(x,potential(x)*np.ones_like(x))
    #plt.show()
