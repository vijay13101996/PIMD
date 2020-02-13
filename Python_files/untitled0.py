#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 12:18:21 2019

@author: vgs23
"""

#To create a flexible simulation environment. 
#Should contain the following
#    1. System definition (Beta, beads etc.)
#    2. Simulation specifications (Method, Method parameters etc.)
#    3. Post processing (Plotting, comparing etc. )
#
#Define a simulation class:
#    
#A simulation object should allow for the use of various path integral 
#methods to compute various quantities, but the object as such should 
#be concerned with the propagation alone. One should be able to define
#each method using functions/variables from the simulation object, but 
#the object itself should be method independent. 
#
#Define a plotter class:
#
#A plotter object should allow for robust plotting of physical quantities
#obtained from various methods, and should help in convergence analysis. 
#
#Define a system class:
#    
#System object would be roughly similar to the current swarm object. It
#should be used to initiate a system with system specific parameters. 
#It should be callable from a simulation object, and should provide ways
#to compute system dependent quantities at any given time. 

