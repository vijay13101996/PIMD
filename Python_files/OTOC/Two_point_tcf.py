#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 13:33:37 2020

@author: vgs23
"""

import numpy as np
#from OTOC import pos_matrix_elts


def pos_matrix_elts(vals,vecs,x_arr,dx,dy,n,k):
    #global vals,vecs,x_arr,dx,dy,basis_N
    return np.sum(vecs[:,n]*vecs[:,k]*x_arr*dx*dy)

def two_point_pos_tcf(vals,vecs,x_arr,dx,dy,beta,t,n_eigen):
    tcf = 0.0
    #print('norm', np.sum(vecs[:,0]*vecs[:,0]*dx))
    for n in range(n_eigen):
        for k in range(n_eigen):
            tcf+= np.exp(-beta*vals[n])*pos_matrix_elts(vals,vecs,x_arr,dx,dy,n,k)\
                *np.exp(1j*(vals[k]-vals[n])*t)*pos_matrix_elts(vals,vecs,x_arr,dx,dy,k,n)
    
    Z_partition = 0.0
    for n in range(n_eigen):
        Z_partition += np.exp(-beta*vals[n])
    tcf/=Z_partition                
    return tcf

