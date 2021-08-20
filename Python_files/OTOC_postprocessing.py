import numpy as np
import utils
from matplotlib import pyplot as plt
import glob
import pickle
theta_arr =  np.linspace(0.0,75.0,301)#np.linspace(0.0,500.0,501)
tcf_tarr = np.linspace(0,4.0,200)
from scipy.stats import norm
import scipy

Classical=0
Matsubara=1
M=3
T=5
N=1000

def Gauss(x,b0,bM):
        return np.exp(b0-bM*(x)**2)

if(Classical==1):
        if(1):
                tcf = np.zeros(len(tcf_tarr))
                fig = plt.figure()      
                fprefix = 'Classical_OTOC_T_{}_N_{}'.format(T,N)
                fname = '{}_S'.format(fprefix)
                files = glob.glob("/home/vgs23/Pickle_files/{}_*100[0-1][0-9][0-9]*".format(fname)) 
                for f in files:
                      #print('f',f)
                      tcf_temp = abs(utils.read_1D_plotdata(f)) 
                      tcf += tcf_temp[:,1]
                      
                tcf/=len(files) 
                fprefix = 'Classical_OTOC_T_{}_N_{}'.format(T,N)
                fname = '{}'.format(fprefix) 
                utils.store_1D_plotdata(tcf_tarr,tcf,fname)
        
        fprefix = 'Classical_OTOC_T_{}_N_{}'.format(T,N)
        fname = '{}'.format(fprefix)
        f = "/home/vgs23/Pickle_files/{}.txt".format(fname)
        tcf_temp = utils.read_1D_plotdata(f)
        tcf_tarr = tcf_temp[:,0]
        tcf = tcf_temp[:,1]
        plt.plot(tcf_tarr,np.log(tcf),label='Classical')

        fprefix = 'Matsubara_theta_OTOC_T_{}_M_{}_N_{}_thermtime_20.0'.format(T,3,N)
        fname = '{}_theta_{}'.format(fprefix,0.0) 
        f = "/home/vgs23/Pickle_files/{}.txt".format(fname) 
        tcf_temp = utils.read_1D_plotdata(f)
        tcf_tarr = tcf_temp[:,0]
        tcf = tcf_temp[:,1]
        plt.plot(tcf_tarr,np.log(tcf),label='M=3, theta=0.0')

        fprefix = 'Matsubara_theta_OTOC_T_{}_M_{}_N_{}_thermtime_20.0'.format(T,5,N)
        fname = '{}_theta_{}'.format(fprefix,0.0) 
        f = "/home/vgs23/Pickle_files/{}.txt".format(fname)        
        tcf_temp = utils.read_1D_plotdata(f)
        tcf_tarr = tcf_temp[:,0]
        tcf = tcf_temp[:,1]
        plt.plot(tcf_tarr,np.log(tcf),label='M=5, theta=0.0')



if(Matsubara==1):
        if(0): # Read theta=0.0 OTOCs from files
                tcf = np.zeros(len(tcf_tarr))
                fig = plt.figure()      
               
                for i in range(1,101): 
                        fprefix = 'Matsubara_theta_OTOC_T_{}_M_{}_N_{}_thermtime_100.0'.format(T,M,N)
                        fname = '/home/vgs23/Pickle_files/{}_theta_{}_S_{}.txt'.format(fprefix,0.0,i)
                        
                        tcf_temp = abs(utils.read_1D_plotdata(fname)) 
                        tcf += tcf_temp[:,1]
                              
                tcf/=100 
                fprefix = 'Matsubara_theta_OTOC_T_{}_M_{}_N_{}_thermtime_100.0'.format(T,M,N)
                fname = '{}_theta_{}'.format(fprefix,0.0) 
                utils.store_1D_plotdata(tcf_tarr,tcf,fname)

        if(0): # Read OTOCs from files 
                
                tcf = np.zeros((len(tcf_tarr),len(theta_arr)))
                fig = plt.figure()      
                for i in range(len(theta_arr)):
                        fprefix = 'Matsubara_theta_OTOC_T_{}_M_{}_N_{}_thermtime_20.0_thrange_75.0'.format(T,M,N)
                        fname = '{}_theta_{}_S'.format(fprefix,theta_arr[i])
                        files = glob.glob("/home/vgs23/Pickle_files/{}_*100[0-1][0-9][0-9]*".format(fname)) 
                        for f in files:
                              #print('f',f)
                              tcf_temp = abs(utils.read_1D_plotdata(f)) 
                              tcf[:,i] += tcf_temp[:,1]
                              
                        print('theta', theta_arr[i], len(files))
                        tcf[:,i]/=len(files) 
                        fprefix = 'Matsubara_theta_OTOC_T_{}_M_{}_N_{}_thermtime_20.0_thrange_75.0'.format(T,M,N)
                        fname = '{}_theta_{}'.format(fprefix,theta_arr[i]) 
                        utils.store_1D_plotdata(tcf_tarr,tcf[:,i],fname)

        
        if(0): # OTOC Slope
                for i in range(0,301):
                        fprefix = 'Matsubara_theta_OTOC_T_{}_M_{}_N_{}_thermtime_20.0_thrange_75.0'.format(5,3,1000)
                        fname = '{}_theta_{}'.format(fprefix,theta_arr[i])
                        f = "/home/vgs23/Pickle_files/{}.txt".format(fname)
                        tcf_temp = utils.read_1D_plotdata(f)
                        tcf_tarr = np.real(tcf_temp[:,0])
                        logtcf = np.log(np.real(tcf_temp[:,1]))
                        #print('t', tcf_tarr[35],tcf_tarr[100])
                        lin_t = tcf_tarr[35:100]
                        lin_otoc = logtcf[35:100]
                        m,c = np.polyfit(lin_t,lin_otoc,1)
                        print('theta,m', theta_arr[i],m)
                        plt.scatter(theta_arr[i],m,color='r')
                        #plt.plot(lin_t,lin_otoc)
                        #plt.plot(lin_t,m*lin_t + c)
                plt.show() 
        
        if(0): # Phase Histogram
                fprefix = 'Matsubara_theta_OTOC_T_{}_M_{}_N_{}_thermtime_{}'.format(T,M,N,20.0)
                fname = '{}_theta_histogram_S_{}'.format(fprefix,0)
                f = "/home/vgs23/Pickle_files/{}.txt".format(fname)
                temp = utils.read_1D_plotdata(f)
                thetagrid = np.real(temp[:,0])
                hist = np.real(temp[:,1])
                print('hist', hist) 
                pt = 425
                popt,pcov = scipy.optimize.curve_fit(Gauss,thetagrid[pt:-pt],hist[pt:-pt])
                
                print('popt', popt,thetagrid[pt:-pt])    
                #plt.plot(thetagrid,hist) 
                plt.plot(thetagrid[pt:-pt],Gauss(thetagrid[pt:-pt],popt[0],popt[1]))
                plt.plot(thetagrid[pt:-pt],hist[pt:-pt])
                plt.show()        

        if(1): # Plot OTOC
                for i in np.arange(0,11,1):
                        fprefix = 'Matsubara_theta_OTOC_T_{}_M_{}_N_{}_thermtime_20.0_thrange_75.0'.format(5,3,1000)
                        fname = '{}_theta_{}'.format(fprefix,theta_arr[i])
                        f = "/home/vgs23/Pickle_files/{}.txt".format(fname)
                        tcf_temp = utils.read_1D_plotdata(f)
                        tcf_tarr = tcf_temp[:,0]
                        tcf = tcf_temp[:,1]#*hist[500+i]
                        #print('tehta', thetagrid[500+i],theta_arr[i])
                        plt.plot(tcf_tarr,np.log(tcf),label=theta_arr[i])

                f =  open("/home/vgs23/Pickle_files/OTOC_{}_beta_{}_basis_{}_n_eigen_{}_tfinal_{}.dat".format('inv_harmonic',0.2,50,50,4.0),'rb+')
                t_arr = pickle.load(f,encoding='latin1')
                OTOC_arr = pickle.load(f,encoding='latin1')
                plt.plot(t_arr,np.log(OTOC_arr), linewidth=1,label='Quantum OTOC')
                f.close()


                plt.legend()
                plt.show()
 
