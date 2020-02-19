#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 14:15:55 2019

@author: vgs23
"""

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy,copy
from numpy import linalg as LA
from numpy.fft import fft,ifft
from multiprocessing import Process
import cProfile

A =[[1,1,3],[4,3,6]]
B = [[1,2,3],[4,5,6]]

A = np.sum(A,axis=0)

arr = np.arange(0,np.pi,0.01)
exp_arr = np.tanh(arr)

arr_tilde = np.fft.fft(exp_arr)

arr_dtilde = np.fft.ifft(arr_tilde)

#A = np.sin(np.arange(0,np.pi,0.01)

#plt.plot(arr,exp_arr,color='b')
#plt.plot(arr,arr_dtilde,color='r')


#A = np.array([[0],[1],[2],[0],[0]])
#B = np.array([[0],[1],[2],[3],[4]])
#
##A = A.T
##B = B.T
#print(A)
##A = np.array([0,1,2,0,0])
##B = np.array([0,1,2,3,4])
#
#A_tilde = np.fft.fft(A,axis=0)
#B_tilde = np.fft.fft(B,axis=0)
#
#conv_tilde = np.conj(A_tilde)*B_tilde
#conv = np.fft.ifft(conv_tilde,axis=0)
#print(conv)

A = np.array([[[1,2,3],[1,4,7]],[[1,3,5],[1,5,9]]])
#print(A[:,1])
#
#B = [A[:,i]-A[:,i-1] for i in range(len(A[0]))]
#print(B)

#A =np.array([0,1,2,3])

#print(np.roll(A,-1,axis=2)-A)

A = np.array([[1,2],[2,3]])
B = np.array([[0,1],[3,4]])

A = np.array([0,1,2,3,4,5])
#print(np.fft.fftfreq(len(A)))

A = np.array([[[-40,-30,-20,-10,0,10,20,30,40]]])
#rearr = [0]
#temp1 = -np.array(range(1,int(len(A)/2) +1))
#temp2 = -temp1
#temp = list(zip(temp1,temp2))
#temp = np.array(temp)
#temp = np.reshape(temp,2*len(temp1))
#print(temp)
#rearr.extend(temp)
#rearr = np.array(rearr)
#rearr -= int(len(A)/2)
#print(rearr)

#print(len(A[0][0]))
#n_beads = 9

#
#def rearrange(Q):
#    return Q[:,:,rearr]
#
#B= rearrange(A)
#print(B)
#
#rearr1 = np.zeros(n_beads,dtype=int)
#
#for i in range(int(n_beads/2)):
#    rearr1[i]=-(2*(i+1))
#    rearr1[i+int(n_beads/2)+1] =2*(i+1)
#
#print(rearr1)    
#
#def reverse_rearrange(Q):
#    return Q[:,:,rearr1]
#
#A = reverse_rearrange(B)
#print(A)
#  
#n_beads = 3  
#Transf = np.zeros((n_beads,n_beads))
#
#for l in range(n_beads):
#    for n in range(int(n_beads/2)+1):
#        if(n==0):
#            Transf[l,n] = 1/n_beads**0.5
#        else:
#            Transf[l,2*n-1] = (2/n_beads)**0.5\
#                    *np.cos(-2*np.pi*n*(l+1)/n_beads)
#            #print(-2*np.pi*n*(l-int(n_beads/2))/n_beads)
#            Transf[l,2*n] = (2/n_beads)**0.5\
#                    *np.sin(2*np.pi*n*(l+1)/n_beads)
#
#print(Transf)
#
#rearr = np.zeros(n_beads,dtype= int)
#for i in range(1,int(n_beads/2)+1):
#    rearr[2*i-1]=-i
#    rearr[2*i]=i
#
##print(rearr)
#rearr+=int(n_beads/2)
#
#temp= np.array(range(-int(n_beads/2), int(n_beads/2)+1 ))*np.pi/n_beads
#print(temp)
#A = np.sin(temp)
##A = np.ones(n_beads)
#print(A)
#
##A = np.fft.fft(A)
##A = np.fft.ifft(A)
##print(A)
##A = A[rearr]
#print(A)
#print(np.matmul(Transf.T,A))

A = np.array([[[0,1,2],[3,4,5]]])
B = np.array([1,2,3])
#print(A*w)

#A = np.random.rand(200,4)
#B =np.random.rand(4)
#print(np.shape(A),np.shape(B))
#print(np.shape(A*B))

#A = 0.0
#print(np.shape(A))


import h5py
#a = np.random.random(size=(100,20))
h5f = h5py.File('data.h5', 'w')
h5f.create_dataset('A',data=A)
h5f.create_dataset('B',data=B)
h5f.close()

h5f = h5py.File('data.h5','r')
A = h5f['A'][:]
B=  h5f['B'][:]
h5f.close()

print('b',B,'A',A)

#for i in range(int(len(A)/2)):
#    rearr[i]=-(2*(i+1))
#    rearr[i+int(len(A)/2)] =2*i
#    
#print(rearr)
#    
#
#
#print(B[rearr])
#print(A*B)
#print(np.sum(A*B,axis=1))


