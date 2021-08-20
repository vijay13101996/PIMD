#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 14:43:00 2019

@author: vgs23
"""

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy,copy
from numpy import linalg as LA
from numpy.fft import fft,ifft
from multiprocessing import Process
import Correlation_function
import Langevin_thermostat
from Correlation_function import corr_function,corr_function_upgrade,corr_function_matsubara
#from MD_System import swarm,pos_op,beta,dimension,n_beads,w_n
import MD_System
import multiprocessing as mp
from functools import partial
import scipy
import Velocity_verlet
import importlib
import time
import pickle
start_time = time.time()
import Matsubara_potential
#import Centroid_mean_force
import Ring_polymer_dynamics
#import Centroid_dynamics
import Quasicentroid_dynamics
import BondAngle_QC_dynamics
import Matsubara_dynamics
import Classical_dynamics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import Potentials_2D
#import Potentials_1D
import csv
import Centroid_dynamics


if __name__ == '__main__':
        if(0):
            ### 3D Morse potential
            potential = Potentials_2D.potential_cb_r
            dpotential = Potentials_2D.dpotential3_cb

        if(0):
            ### Quartic potential
            potential = Potentials_2D.potential_quartic
            dpotential = Potentials_2D.dpotential_quartic

        if(1):
            ### Quartic(Matsubara) potential
            potential = Matsubara_potential.pot_inv_harmonic_M3
            dpotential = Matsubara_potential.dpot_inv_harmonic_M3
            ddpotential = Matsubara_potential.ddpot_inv_harmonic_M3

        deltat = 0.01
        N = 1000
        T = 4.0#5.0*deltat#5000.0*deltat#3000*deltat

        n_tp = 1000 
        dt_tp = T/n_tp
        tcf_tarr = np.arange(0,T+0.0001,dt_tp)
        tcf = np.zeros_like(tcf_tarr) 

        fs =  41.341374575751

        beads=32
        T_K = 800
        T_au = 5
        beta= 1/T_au#1.0/5#1.0/5#315773/T_K#500.0    # Take care of redefinition of beta, when using QCMD!
        n_instance =1
        print('N',N)#*100*n_instance)
        print('Beta T_au',beta, T_au)
        print('deltat',deltat)
        #print('beads',beads)

        ACMD = 0
        CMD = 0
        QCMD = 0 
        AQCMD =0
        RPMD =0
        BAQCMD=0
        Classical=0
        Matsubara=1

        if(0):
            x_ar = np.arange(0.1,6.0,0.1)
            y_ar = np.arange(0.1,6.0,0.1)
            X,Y = np.meshgrid(x_ar,y_ar)
            #plt.imshow(potential(X,Y))
            
            #fig = plt.figure()
            #ax = fig.add_subplot(111, projection='3d')
            plt.plot(x_ar,Potentials_2D.dpotential_cb_x(x_ar,0*y_ar))
            #Z = potential(X,Y)
            #Z[Z>1] = 1
            #ax.plot_surface(X,Y,Z)
            plt.show()

        if(BAQCMD==1):
            Q = np.linspace(1.0,2.5,50)
            rforce = BondAngle_QC_dynamics.BAQCMD_instance(beads,dpotential,beta,100,1,Q,deltat)
            #rforce16 = BondAngle_QC_dynamics.BAQCMD_instance(16,dpotential,beta,100,1,Q,deltat)
            print('time',time.time() - start_time)
              
            def force3_baqcmd(Q):
                r = np.sum(Q**2,axis=1)**0.5
                r = r[:,np.newaxis]
                r = np.repeat(r,3,axis=1)
                #print('Q,r',Q,r)
                return Q*rforce(r)/r

            def force3_baqcmd16(Q):
                r = np.sum(Q**2,axis=1)**0.5
                r = r[:,np.newaxis]
                r = np.repeat(r,3,axis=1)
                #print('Q,r',Q,r)
                return Q*rforce16(r)/r
         
            def force2_baqcmd(Q):
                r = np.sum(Q**2,axis=1)**0.5
                r = r[:,np.newaxis]
                r = np.repeat(r,2,axis=1)
                #print('Q,r',Q,r)
                return Q*rforce(r)/r

            def forceorig_baqcmd(Q):
                r = np.sum(Q**2,axis=1)**0.5
                r = r[:,np.newaxis]
                r = np.repeat(r,3,axis=1)
                #print('Q,r',Q,r)
                return Q*Potentials_2D.dpotential_cb_r(r)/r

            def forceharm_baqcmd(Q):
                r = np.sum(Q**2,axis=1)**0.5
                r = r[:,np.newaxis]
                r = np.repeat(r,3,axis=1) 
                return Q*Potentials_2D.dpotential_cb_harm(r)/r
            
            plt.plot(Q,Potentials_2D.dpotential_cb_harm(Q))
            plt.show()

            tcf1 = BondAngle_QC_dynamics.compute_tcf_BAQCMD(n_instance,N,force3_baqcmd,beta,T,n_tp,deltat)
            #tcf2 = BondAngle_QC_dynamics.compute_tcf_BAQCMD(n_instance,N,force3_baqcmd16,beta,T,n_tp,deltat)
            plt.plot(tcf_tarr,tcf1)
            #plt.plot(tcf_tarr,tcf2)
            plt.show()
        if(QCMD==1):
            rforce = Quasicentroid_dynamics.QCMD_instance(32,dpotential,beta,100,5,128,1.0)
            print('time',time.time() - start_time)
            
            def force_xy(Q):
                x = Q[:,0,...]
                y = Q[:,1,...]
                r = (x**2+y**2)**0.5
                force_r = rforce(r)
                #force_xy = force_r*abs(np.array([x/r,y/r]))
                dpotx = force_r*(x/r)
                dpoty = force_r*(y/r)
                ret = np.transpose(np.array([dpotx,dpoty]),(1,0,2))
                return ret#np.array([dpotx,dpoty])
            #force_xy = np.vectorize(force_xy)
             
            tcf = Quasicentroid_dynamics.compute_tcf_QCMD(n_instance,N,force_xy,beta,T,n_tp,deltat)
               
            plt.plot(tcf_tarr,tcf)
            plt.show()
        if(AQCMD==1):
            """
            This code is for the Adiabatic implementation of Quasicentroid 
            Molecular dynamics.
            """
            Quasicentroid_dynamics.compute_tcf_AQCMD(n_instance,N,beads,dpotential,beta,T,deltat)
            
            for i in range(n_instance):
                f = open('/home/vgs23/Pickle_files/AQCMD_tcf_N_{}_B_{}_inst_{}_T_{}_dt_{}_NB_{}.dat'.format(N*100,beta,i,T,deltat,beads),'rb')
                tcf+= pickle.load(f)
                plt.plot(tcf_tarr,tcf,color='r')
                #plt.plot(tcf_tarr,np.cos(tcf_tarr),color='g')
                f.close()
                print(i,'completed')
                print('time',time.time() - start_time)
            
            tcf/=(n_instance)
            plt.plot(tcf_tarr,tcf)

            tcf*=0.0
            for i in range(n_instance):
                f = open('/home/vgs23/Pickle_files/QCMD_tcf_N_{}_B_{}_inst_{}_T_{}_dt_{}_SAMP1.dat'.format(N*100,beta,i,T,0.5),'rb')
                tcf += pickle.load(f)
                f.close()
                print(i,'completed')
            #plt.plot(tcf_tarr,tcf,color='g')
            plt.show()
        if(ACMD==1): 
            """
            This code is for Adiabatic implementation of Centroid Molecular 
            dynamics.
            """
            Centroid_dynamics.compute_tcf_ACMD(n_instance,N,beads,dpotential,beta,T,deltat)

            for i in range(n_instance):
                f = open('/home/vgs23/Pickle_files/ACMD_vv_tcf_N_{}_B_{}_inst_{}_T_{}_dt_{}_NB_{}_SAMP3.dat'.format(N*100,beta,i,T,deltat,beads),'rb')
                tcf += pickle.load(f)
                f.close()
                print(i,'completed')
            
            tcf/=(n_instance)
            print(tcf[0],tcf[3],tcf[7])
            plt.plot(tcf_tarr,tcf,color='r')
            #plt.plot(tcf_tarr,np.cos(tcf_tarr)/(MD_System.beta*MD_System.n_beads),color='g')
            plt.show()

            if(1):
                tcf*=0.0
                for i in range(n_instance):
                        f = open('/home/vgs23/Pickle_files/CMD_vv_tcf_N_{}_B_{}_inst_{}_T_{}_dt_{}_SAMP2.dat'.format(N*100,beta,i,20000.0,2.0),'rb')
                        tcf = pickle.load(f)
                        print(tcf[0])
                        f.close()
                        print(i,'completed')
                    
                tcf/=(n_instance)
                #tcf*=beads
                print(tcf[0], tcf[5]*beads,tcf[45]*beads,tcf[89]*beads)
        if(RPMD==1):
            Ring_polymer_dynamics.compute_tcf(n_instance,N,beads,dpotential,beta,T,n_tp,deltat)
            #Ring_polymer_dynamics.compute_static_avg(N,beads,dpotential,beta,deltat)
            for i in range(n_instance):
                f = open('/home/vgs23/Pickle_files/RPMD_Andersen_tcf_N_{}_B_{}_inst_{}_dt_{}_NB_{}_S1.dat'.format(N*100,beta,i,deltat,beads),'rb')
                tcf += pickle.load(f)
                f.close()
                print(i,'completed')
            
            tcf/=(n_instance)
        if(CMD==1):
            xforce = Centroid_dynamics.CMD_instance(beads,dpotential,beta)
            Q=np.arange(0.5,3,0.1) 
            plt.plot(Q,xforce(Q))
            #plt.plot(Q,dpotential_morse(Q))
            plt.show()
           
            def force_xy(Q):
                x = Q[:,0,...]
                y = Q[:,1,...]
                r = (x**2+y**2)**0.5
                force_r = xforce(r)

                #force_xy = force_r*abs(np.array([x/r,y/r]))
                dpotx = force_r*(x/r)
                dpoty = force_r*(y/r)
                ret = np.transpose(np.array([dpotx,dpoty]),(1,0,2))
                return ret#np.array([dpotx,dpoty])

            Centroid_dynamics.compute_tcf_CMD(n_instance,N,force_xy,beta,T,deltat)
            for i in range(n_instance):
                f =  open('/home/vgs23/Pickle_files/CMD_tcf_N_{}_B_{}_inst_{}_T_{}_dt_{}_SAMP1.dat'.format(N*100,beta,i,T,deltat),'rb')
                #open('/home/vgs23/Pickle_files/CMD_tcf_N_{}_B_{}_inst_{}_dt_{}_NB_{}.dat'.format(N*100,MD_System.beta*MD_System.n_beads,i,deltat,MD_System.n_beads),'rb')
                tcf += pickle.load(f)
                f.close()
                print(i,'completed')
                
            tcf/=(n_instance)
            #tcf*=beads
            #print(tcf[0]*beads, tcf[5]*beads,tcf[45]*beads,tcf[89]*beads)
            print(tcf[0],tcf[3],tcf[7])
            plt.plot(tcf_tarr,tcf,color='r')
            #plt.plot(tcf_tarr,np.cos(tcf_tarr)/(MD_System.beta*MD_System.n_beads),color='g')
            plt.show()
        if(Classical==1):
            potential = Matsubara_potential.pot_inv_harmonic_M1
            dpotential = Matsubara_potential.dpot_inv_harmonic_M1
            ddpotential = Matsubara_potential.ddpot_inv_harmonic_M1

            thermtime = 100.0 #a.u. 5 for T=5
            tcf_tarr = np.linspace(0,4.0,200)
            fprefix = 'Classical_OTOC_T_{}_N_{}'.format(T_au,N)
            
            instance = [90]#range(100001,100101)#[10000,10001]
            
            ctx = mp.get_context('spawn')
            start_time = time.time() 
            Classical_dynamics.Classical_OTOC_instance(N,beta,thermtime,deltat,dpotential,ddpotential,tcf_tarr, fprefix,instance,ctx)
             
 
        # TO DO:
        # 1. Delete all the unnecessary/redundant comments.
        # 2. Assign file name format in this module.
        # 3. Save data in txt files, no pickle files anymore. Create a function that does that, and store that in utils.py
        # 4. Plot data in interactive mode or in gnuplot.
        # 5. Remove random parameters in other modules.
        # 6. Call thermalization in this file here, so that you have control over it.  

         
        if(Matsubara==1):
            M=3    
            ntheta = 301
            theta_arr =  np.linspace(0.0,75.0,ntheta)
            #print('theta',theta_arr)
            potential = Matsubara_potential.pot_inv_harmonic_M3
            dpotential = Matsubara_potential.dpot_inv_harmonic_M3
            ddpotential = Matsubara_potential.ddpot_inv_harmonic_M3

            thermtime = 20.0 #a.u. 5 for T=5
            tcf_tarr = np.linspace(0,4.0,200)
            theta = 0.0
            fprefix = 'Matsubara_theta_OTOC_T_{}_M_{}_N_{}_thermtime_{}_thrange_75.0'.format(T_au,M,N,thermtime)

            #Matsubara_dynamics.Matsubara_phase_coverage(N,M,beta,1000,20.0,deltat,dpotential,fprefix,1001,0)
            
            instance = range(100001,100101)#[10000,10001]
            
            ctx = mp.get_context('spawn')
            start_time = time.time() 
            #Matsubara_dynamics.Phase_dep_OTOC_instance(N,M,beta,thermtime,deltat,dpotential,ddpotential,tcf_tarr, fprefix, 0.0,instance,ctx)
            
            for i in range(len(theta_arr)):
                Matsubara_dynamics.Phase_dep_OTOC_instance(N,M,beta,thermtime,deltat,dpotential,ddpotential,tcf_tarr, fprefix, theta_arr[i],instance,ctx)
                print('time', time.time()-start_time)
            print('time', time.time()-start_time)
                 
 
            if(0):
                tcf = np.zeros((len(tcf),ntheta)) + 0j
                color_arr = ['r','g','b','m','c','k']
                count = 0
                #for beta in beta_arr:
                for i in [220]:#range(4000,4500):#Matsubara_instance:
                        f = open('/home/vgs23/Pickle_files/Matsubara_phase_dep_OTOC_N_{}_B_{}_inst_{}_dt_{}_M_{}_ntheta_{}.dat'.format(N,beta,i,deltat,M,ntheta),'rb')
                        tcf_curr = pickle.load(f)
                        if(np.isnan(np.sum(tcf_curr))!=True):
                                tcf+= tcf_curr
                                print(tcf_curr[:,0])
                                #plt.plot(tcf_tarr,np.log(abs(tcf_curr[:,0])), label=i) 
                                count+=1
                                plt.plot(tcf_tarr,np.log(abs(tcf[:,0])/count), label=i)
                        f.close()
                        print(i,count,'completed', tcf_curr[20])
                tcf/=count#len(Matsubara_instance)
                #tcf = np.nan_to_num(tcf)
                print('tcf', tcf)
                plt.show()
                if(0):
                        f = open('/home/vgs23/Pickle_files/Matsubara_phase_dep_OTOC_N_{}_B_{}_inst_{}_dt_{}_M_{}_ntheta_{}.dat'.format(N,beta,i,deltat,3,ntheta),'rb')
                        tcf_curr = pickle.load(f)
                        plt.plot(tcf_tarr,np.log(abs(tcf_curr[:,0])), label='M=3')        
                        plt.legend()
                        plt.show()
                print('tcf', tcf.shape)
                plt.plot(tcf_tarr,np.log(abs(tcf[:,0])), label = 'Average')
                plt.legend()
                plt.show()
                #print('tcf',tcf,np.log(np.real(tcf)), np.log(-np.imag(tcf)))
                #theta_arr = np.linspace(-10.0,10.0,ntheta)
                plt.title(r'$C_T^{Mats}$ vs $\theta$')
                for i in [0,1,2,3,4,5,6,7,8,9,10]:#range(ntheta):#,10,15,20]:#,25,30,35,40]:#range(20):
                    #print('i',i, max(np.log(tcf[:,i])))
                    plt.plot(tcf_tarr,np.log(abs(tcf[:,i])), label=r'$\theta$={}'.format(theta_arr[i]),linewidth=1.5)
                    #plt.plot(tcf_tarr,np.log(abs(np.real(tcf[:,i]))), label=theta_arr[i],linewidth=1.5) 
                    #plt.plot(tcf_tarr,np.log(abs(np.imag(tcf[:,i]))), label=theta_arr[i],linewidth=1.5)
                    
                    #w = (np.fft.fftfreq(tcf_tarr.shape[-1], d = dt_tp ))[:n_tp//2]
                    #fft = np.real(np.fft.fft(tcf[:,i]))[:n_tp//2]
                    #plt.plot(w,fft, label=theta_arr[i],linewidth=1.5) 
                    #plt.show()
                #plt.plot(tcf_tarr,((tcf[:,20])),color ='g')
                #plt.plot(tcf_tarr,((tcf[:,40])),color='b')
                #plt.xlim([0,10])
                
                plt.ylabel(r'$C_T(t)$')
                plt.xlabel(r'$t$')

                f =  open("/home/vgs23/Pickle_files/OTOC_{}_beta_{}_basis_{}_n_eigen_{}_tfinal_{}.dat".format('inv_harmonic',0.2,50,50,4.0),'rb+')
                t_arr = pickle.load(f,encoding='latin1')
                OTOC_arr = pickle.load(f,encoding='latin1')
                plt.plot(t_arr,np.log(OTOC_arr), label='Quantum OTOC')
                f.close()

                plt.legend()
                plt.show()
                
            if(0):
                # Quantum OTOC
                f =  open("/home/vgs23/Pickle_files/OTOC_{}_beta_{}_basis_{}_n_eigen_{}_tfinal_{}.dat".format('inv_harmonic',0.2,50,50,4.0),'rb+')
                t_arr = pickle.load(f,encoding='latin1')
                OTOC_arr = pickle.load(f,encoding='latin1')
                plt.plot(t_arr,np.log(OTOC_arr), label='Quantum OTOC')
                f.close()
                
                # Classical OTOC
                f = open('/home/vgs23/Pickle_files/Classical_OTOC_N_{}_B_{}_inst_{}_dt_{}.dat'.format(10000,beta,100,deltat),'rb')
                tcf_curr = pickle.load(f)
                plt.plot(tcf_tarr,np.log(abs(tcf_curr)), label='Classical OTOC')
                f.close()
               
                tcf = np.zeros((len(tcf),ntheta)) + 0j
                count = 0 
                # M=3 for theta = -10.0
                for i in [220]:
                        f = open('/home/vgs23/Pickle_files/Matsubara_phase_dep_OTOC_N_{}_B_{}_inst_{}_dt_{}_M_{}_ntheta_{}.dat'.format(10000,beta,i,deltat,3,ntheta),'rb')
                        tcf_curr = pickle.load(f)
                        if(np.isnan(np.sum(tcf_curr))!=True):
                                tcf+= tcf_curr
                                print(tcf_curr[:,0])
                                #plt.plot(tcf_tarr,np.log(abs(tcf_curr[:,0])), label=i) 
                                count+=1
                                #plt.plot(tcf_tarr,np.log(abs(tcf[:,0])/count), label=i)
                        f.close()
                        print(i,count,'completed', tcf_curr[20])
                tcf/=count#len(Matsubara_instance)
                #tcf = np.nan_to_num(tcf)
                plt.plot(tcf_tarr,np.log(abs(tcf)), label = r'$\theta = -10.0$, M=3') 

                # M=5 for theta = -10.0
                for i in range(4000,4500):
                        f = open('/home/vgs23/Pickle_files/Matsubara_phase_dep_OTOC_N_{}_B_{}_inst_{}_dt_{}_M_{}_ntheta_{}.dat'.format(100,beta,i,deltat,5,ntheta),'rb')
                        tcf_curr = pickle.load(f)
                        if(np.isnan(np.sum(tcf_curr))!=True):
                                tcf+= tcf_curr
                                print(tcf_curr[:,0])
                                #plt.plot(tcf_tarr,np.log(abs(tcf_curr[:,0])), label=i) 
                                count+=1
                                #plt.plot(tcf_tarr,np.log(abs(tcf[:,0])/count), label=i)
                        f.close()
                        print(i,count,'completed', tcf_curr[20])
                tcf/=count#len(Matsubara_instance)
                #tcf = np.nan_to_num(tcf)
                plt.plot(tcf_tarr,np.log(abs(tcf)), label = r'$\theta = -10.0$, M=5') 
                plt.title(r'$C_T$ : Classical, Quantum, Matsubara') 
                plt.ylabel(r'$C_T(t)$')
                plt.xlabel(r'$t$')
                plt.legend()
                plt.show()
