import Stadium_billiards
import numpy as np
#import matplotlib.pyplot as plt
import cProfile

N = 10000
X = np.linspace(-2,2,N)
Y = np.linspace(-2,2,N)

Le = 1.0
r_c = 1.0

def potential_2D(x,y):
    if(abs(x)<Le/2 and abs(y)<r_c):
        return 0.0
    elif(abs(x)<(Le/2+r_c) and abs(y)<r_c):
        if(x>0):
            if(((x-Le/2)**2+y**2)<r_c**2):
                return 0.0
            else:
                return 1e5
        else:
            if(((x+Le/2)**2+y**2)<r_c**2):
                return 0.0
            else:
                return 1e5
    else:
        return 1e5

pot = np.zeros((N,N))

potential = np.vectorize(potential_2D)
XX,YY =np.meshgrid(X,Y)

def assignment(X,Y,pot):
	X[:] = np.array(X[:],order = 'F')
	Y[:] = np.array(Y[:],order = 'F')
	pot[:] = np.array(pot[:], order = 'F')

#for i in range(20):
def fortran_potential_assign(X,Y,pot):
	#X = np.array(X,order = 'F')
	#Y = np.array(Y,order = 'F')
	#pot = np.array(pot, order = 'F')
	pot = Stadium_billiards.stadium_billiards.potential_assignment(X,Y,1.0,1.0,pot)
	#plt.imshow(pot,origin='lower')
	#plt.show()
	#pot = potential(X,Y)

def potential_assign(pot):
	pot = potential(X,Y)

#cProfile.run('Stadium_billiards.stadium_billiards.potential_assignment(X,Y,1.0,1.0,pot)')
#cProfile.run('potential_assign(pot)')

#fortran_potential_assign(X,Y,pot)
#cProfile.run('fortran_potential_assign(X,Y,pot)')

#plt.imshow(pot,origin='lower')
#plt.show()
