import numpy as np
import MD_System

# Comment out this code. Not very self-explantory.

def proj(U,V,axg):
    return (np.sum(U*V,axis=axg)/np.sum(U*U,axis=axg))

def norm(U,axg):
    return np.sum(U*U,axis=axg)**0.5

def Gram_Schmidt(T):
    i=0
    j=0
    N = len(T.T)
    #print('T',T,N)
    T_gram_prim = T.copy()
    T_gram = np.zeros_like(T)
    while(i<N):
        j=0
        while(j<i):
            #print('i j', i,j)
            T_gram_prim[...,i]-=  proj(T_gram_prim[...,j],T[...,i],1)[:,None]*T_gram_prim[...,j]
            j+=1
        T_gram[...,i]=T_gram_prim[...,i]/norm(T_gram_prim[...,i],1)[:,None]  
        i+=1

    return T_gram

def CHANGED_theta_constrained_randomize(swarmobj,theta,rng): ## FOR ANY DIMENSION: Code to be fixed!
    M = swarmobj.n_beads
    w_marr = 2*np.arange(-int((M-1)/2),int((M-1)/2)+1)*np.pi\
            /(swarmobj.beta)

    # Sampling the position distribution
    swarmobj.q = rng.normal(0,1,np.shape(swarmobj.q))

    # Scaling factor for Pi_0 
    R = np.sum(w_marr**2*swarmobj.q[...,::-1]**2,axis=2)**0.5

    # First column of the unitary transformation matrix
    T0 = w_marr*swarmobj.q[...,::-1]/R   ### To be modified for multidimensional case!

    dim = swarmobj.dimension
    T = rng.rand(len(swarmobj.q),dim,M,M)
    T[...,0] = T0
    
    N = swarmobj.N
    Pi = rng.normal(0,swarmobj.m/swarmobj.beta,(N,dim,M))
    
    for i in range(dim):
        T[:,i] = Gram_Schmidt(T[:,i])                               # Obtain the unitary transformation matrix
        Pi[:,i,0] = (theta/dim)/R[:,i]                              # Setting Pi_0 to theta/R, so that the resultant distribution has the required theta
        swarmobj.p[:,i,:] =np.einsum('ijk,ik->ij', T[:,i],Pi[:,i])  # Solving for the momentum distribution
    
def theta_constrained_randomize(swarmobj,theta,rng):
    # Sampling the position distribution
    #swarmobj.q = rng.normal(0,1,np.shape(swarmobj.q))

    # Scaling factor for Pi_0 
    R = np.sum(swarmobj.w_marr**2*swarmobj.q[...,::-1]**2,axis=2)**0.5

    # First column of the unitary transformation matrix
    T0 = swarmobj.w_marr*swarmobj.q[...,::-1]/R   

    M = swarmobj.M
    
    T = rng.rand(len(swarmobj.q),M,M)
    T[...,0] = T0[:,0,:]

    # Obtain the unitary transformation matrix
    T = Gram_Schmidt(T)

    N = swarmobj.N
    Pi = rng.normal(0,(swarmobj.m/swarmobj.beta)**0.5,(N,M))
    
    # Setting Pi_0 to theta/R, so that the resultant distribution has the required theta
    Pi[:,0] = theta/R[:,0]
   
    # Solving for the momentum distribution
    swarmobj.p[:,0,:] =np.einsum('ijk,ik->ij', T,Pi) #np.matmul(T,Pi)
    #print('theta',np.sum(swarmobj.p*w_marr*swarmobj.q[...,::-1],axis=2))
    for i in range(1000):
        if(R[i]<2.0):
                print('i q', i, swarmobj.q[i],R[i])
    #print('T pi p R', T[413], Pi[413],swarmobj.q[413], R[413], R.shape) 
    #print('p', swarmobj.p[413])

def centrifugal_term(swarmobj,theta):
    Rsq = np.sum(swarmobj.w_marr**2*swarmobj.q[...,::-1]**2,axis=2)[:,None]
    const = theta**2/swarmobj.m 
    dpot = (-const/Rsq**2)*swarmobj.w_marr[::-1]**2*swarmobj.q  ### Check the force term here. There is some unseemly behaviour
    #print('R', Rsq[951]**0.5,const/Rsq[951,0])
    return dpot
 
if(0):
    # Comment out this section too, and rewrite it as a unit test procedure.
    T = np.random.random((1,3,3))
    print(T)

    T = Gram_Schmidt(T)
    print('T After',T[0], np.transpose(T,axes=[0,2,1]))
    print(np.matmul(T[0],T[0].T))

    N=10
    M=3
    beta=1.0
    swarmobj = MD_System.swarm(N,M,beta,1)
    swarmobj.q = np.random.random((N,1,M))
    swarmobj.p = np.zeros_like(swarmobj.q)
    theta_constrained_randomize(swarmobj,-10,1)
