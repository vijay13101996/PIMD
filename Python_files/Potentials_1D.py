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


def dpotential_cb_1D(x):
    D0 = 0.18748
    alpha = 1.1605
    x_c = 1.8324

    ret =  2*D0*alpha*(1 - np.exp(-alpha*(x - x_c)))*np.exp(-alpha*(x - x_c))
    return ret




