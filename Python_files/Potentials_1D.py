#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 14:13:18 2020

@author: vgs23
"""
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
def potential_1D_box(x):
        L=1.0
        if(x<0 or x>L):
            return 1e5
        else:
            return 0.0#1/(1+np.exp(k*x)) + 1/(1+np.exp(-k*(x-L)))#0.0#0.5*x**2
        
def potential_quartic(x):
    return 0.25*x**4

def sigm(x):
    return 2/(1+np.exp(-1.5*x+5))

def dpotential_cb_1D(x):
    D0 = 0.18748
    alpha = 1.1605
    x_c = 1.8324

    ret =  2*D0*alpha*(1 - np.exp(-alpha*(x - x_c)))*np.exp(-alpha*(x - x_c))
    return ret



if(0):
    matplotlib.rcParams.update({'font.size': 22})
    x = np.arange(0,10,0.1)
    fig, ax = plt.subplots()
    y  = sigm(x) 
    ax.plot(x, y)
    ax.axvline(x=6.3,ymin=0.0, ymax = sigm(6.3)/2.1,linestyle='--')
    ax.axvline(x=0.2,ymin=0.0, ymax = sigm(0.2)*2.1,linestyle='--')
    ax.axhline(y=2.0,xmin=0.0, xmax = 1.0,linestyle='--')
    ax.annotate('Exponential growth',xy=(2.1,0.2))
    ax.annotate('Post Scrambling \n saturation', xy=(7.0,1.8))
    plt.xlabel('$t$')
    plt.ylabel('$C(t)$')
    plt.xticks([0.2,6.3],[r'$t_d$',r'$t_*$'])
    plt.yticks([0.0,2.0])
    plt.suptitle('Generic OTOC behaviour')
    plt.show()
