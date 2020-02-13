#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 13:43:36 2019

@author: vgs23
"""

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy,copy
from numpy import linalg as LA
from numpy.fft import fft,ifft
from multiprocessing import Process
from Correlation_function import corr_function,corr_function_upgrade,corr_function_matsubara
#from MD_System import swarm,pos_op,beta,dimension,n_beads,w_n
import MD_System
import multiprocessing as mp
from functools import partial
import scipy
#from Langevin_thermostat import thermalize
#from Centroid_mean_force import CMF
#from External_potential import dpotential
#import Centroid_mean_force
import Velocity_verlet
import importlib
import time
import pickle
start_time = time.time()
from Matsubara_potential_hh import dpot_hh

deltat = 1e-2
M=3 
dimension = 2
MD_System.system_definition(8,M,dimension,MD_System.dpotential)
importlib.reload(Velocity_verlet)

swarmobject = MD_System.swarm(1)
swarmobject.q = np.zeros((swarmobject.N,MD_System.dimension,MD_System.n_beads))

print(np.shape(swarmobject.q))
#swarmobject.p = rand_boltz.normal(0.0,swarmobject.m/MD_System.beta,np.shape(swarmobject.q))