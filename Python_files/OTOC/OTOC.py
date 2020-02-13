#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:34:01 2020

@author: vgs23
"""

import numpy as np

global vals,vecs,x_arr,dx,dy,basis_N,n_eigen

def pos_matrix_elts(n,k):
    global vals,vecs,x_arr,dx,dy,basis_N
    return np.sum(vecs[:,n]*vecs[:,k]*x_arr*dx*dy)
    #return np.sum(vecs[:,n]*vecs[:,k]*x*dx)#np.sum(temp1*temp2*x*dx)#
    
x_matrix = np.vectorize(pos_matrix_elts)

#pos_mat_elt = 0.0
#n_arr = np.arange(5)
k_arr = np.arange(basis_N)
#pos_mat = np.zeros_like(k_arr)
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
    global vals,n_eigen
    val = vals[:n_eigen]
    Z = np.sum(np.exp(-beta*val))
    n_arr = np.arange(0,len(val))
    return (1/Z)*np.sum(np.exp(-beta*val)*c_mc(n_arr,t)  )

OTOC = np.vectorize(OTOC)

#cProfile.run('OTOC_f.position_matrix.compute_otoc_arr_t(vecs,x_arr,dx,dy,k_arr,vals,m_arr,t_arr,0.001,n_eigen,OTOC_mat)')
#cProfile.run('OTOC(t_arr,0.001)')