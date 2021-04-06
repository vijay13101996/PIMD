import itertools
import sympy
from sympy import *

def A(i,x):
        if(i==0):
            return 1.0
        elif(i<0):
            return 2**0.5*cos(2*i*pi*x)
        else:
            return 2**0.5*sin(2*i*pi*x)

x = sympy.Symbol('x')
M = 5
degree = 4
iterate = list(itertools.product(range(M),repeat=degree))
print('iterate', (iterate))
potential = sympy.Symbol('')
Q_tilde_m = [sympy.Symbol('Q_m{}'.format(i)) for i in range(1,M//2+1)]
Q_tilde_m = Q_tilde_m[::-1]
Q_tilde_p = [sympy.Symbol('Q_{}'.format(i)) for i in range(M//2+1)]
Q_tilde_m.extend(Q_tilde_p)
Q_tilde = Q_tilde_m
print('here',Q_tilde_m, Q_tilde)

potential = sympy.Symbol('')
count = 0
for combi in iterate:
        print('term {} of'.format(count), len(iterate))
        count+=1
        term = sympy.Symbol('')
        coeff = sympy.Symbol('1')
        for i in range(degree):
                term*= Q_tilde[combi[i]]
                #print('term',i, term,combi[i])
                coeff*= A(combi[i]-M//2,x)
                #print('coeff', coeff, combi[i]) 
        coeff = integrate(coeff, (x,0,1))
        print('coeff', coeff, term)
        potential+= term*coeff#Q_tilde[combi[0]]*Q_tilde[combi[1]]*coeff_harmonic(combi[0]-1,combi[1]-1)
        
print('pot', potential)

deriv=1
if(deriv==1):
        #print(0.5*potential) 
        for i in range(M):
                print('grad w.r.t k={}  '.format(i),diff(potential,Q_tilde[i]))
                print('\n')
        iterate = list(itertools.product(range(M),repeat=2))  

        for combi in iterate:
                print('dd of k=',combi[0], combi[1], diff(potential,Q_tilde[combi[0]],Q_tilde[combi[1]]))
                print('\n')
        
