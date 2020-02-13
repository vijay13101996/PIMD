#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 15:20:12 2020

@author: vgs23
"""

import H_matrix_1D
import numpy as np


"""
The basis used for diagonalization is the same as above: DVR Basis. 
In order to ensure that the basic implementation is the same, the 
basis set is reordered into a 1D array, which can expanded into the 
full 2D set using the tuple_index function above.
"""


def Kin_matrix_2D_elt(x,dx,y,dy,i,i_d,j,j_d,a,b,N):
    if(j==j_d and i!=i_d):
        return H_matrix_1D.Kin_matrix_elt(x,dx,i,i_d,a,b,N)
    elif(i==i_d and j!=j_d):
        #print('hereh',Kin_matrix_elt(y,dy,j,j_d,a,b,N),i,j,j_d)
        return H_matrix_1D.Kin_matrix_elt(y,dy,j,j_d,a,b,N)
    elif(i==i_d and j==j_d):
        return H_matrix_1D.Kin_matrix_elt(x,dx,i,i_d,a,b,N) + H_matrix_1D.Kin_matrix_elt(y,dy,j,j_d,a,b,N)
    else:
        return 0.0
    
def Pot_matrix_2D(x,dx,y,dy,a,b,N,potential_2D):
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

def Potential_matrix_2D(x,dx,y,dy,a,b,N,potential_2D):
    
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