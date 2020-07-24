from __future__ import print_function, division
import numpy as np
import sys

from qclpysim.utils.xyzio import read_first as read_xyz, append_frame, write_frame


qarray = np.array([[0.0000000000e+00, 1.7737081512e+00, -0.5262282166e+00], [1.5360751082e+00,-0.8868540756e+00, -0.5262282166e+00],
   [-1.5360751082e+00, -0.8868540756e+00, -0.5262282166e+00], [0.0000000000e+00, 0.0000000000e+00, 0.2255274584e+00]])#/1.8897259885789 
labels = np.array(['H','H','H','N'])

op_file = open('ammonia.xyz','w+')
write_frame(op_file,labels,qarray,cmnt='')
op_file.close()

config_file = "ammonia.xyz"
nbeads = 32
labels, qinit = read_xyz(config_file)
ntot = len(qinit)
natoms_original = ntot
natoms = 32*4
qinit.shape = (natoms_original, 1, 3)
qcart = np.transpose(qinit, [0,2,1]).reshape((-1,1))
labels = labels[::1]

print(labels)
print(qinit.shape)

qinit_beads = np.zeros((natoms,nbeads,3))
label_beads = np.chararray(natoms*nbeads)

i=0
while(i<natoms):#i<len(labels)):
    
    bond_length = ( (qinit[3]-qinit[2]))/3
    #print(bond_length.shape,'bl')
    avg_position = np.random.normal(0.0,15*abs(bond_length),bond_length.shape)  #np.random.uniform(-10*bond_length,10*bond_length, bond_length.shape)
    #print('avg', avg_position)
    
    H1_position = qinit[0]+avg_position 
    H2_position = qinit[1]+avg_position
    H3_position = qinit[2]+avg_position
    N_position = qinit[3]+avg_position
    
    NH1_distance = N_position - H1_position
    NH2_distance = N_position - H2_position
    NH3_distance = N_position - H3_position
    print('Bondlength', np.linalg.norm(NH1_distance), np.linalg.norm(NH2_distance), np.linalg.norm(NH3_distance))

    H1_arr = np.random.normal(H1_position, abs(NH1_distance)/10.0, (nbeads,3))
    H2_arr = np.random.normal(H2_position, abs(NH2_distance)/10.0, (nbeads,3))
    H3_arr = np.random.normal(H3_position, abs(NH3_distance)/10.0, (nbeads,3))
    N_arr = np.random.normal(N_position, (abs(NH1_distance)+abs(NH2_distance)+abs(NH3_distance))/30.0, (nbeads,3))

    qinit_beads[i] = H1_arr
    qinit_beads[i+1] = H2_arr
    qinit_beads[i+2] = H3_arr
    qinit_beads[i+3] = N_arr
   
    label_beads[i*nbeads:(i+1)*nbeads] = labels[0]
    label_beads[(i+1)*nbeads: (i+2)*nbeads] = labels[1]
    label_beads[(i+2)*nbeads: (i+3)*nbeads] = labels[2]
    label_beads[(i+3)*nbeads: (i+4)*nbeads] = labels[3]

    if(i<10):
        print(N_position, qinit_beads[i], labels[0],labels[1], labels[2])
    
    i+=4

#print(qinit_beads)
append_frame("init.xyz",label_beads, qinit_beads.reshape(-1,3))


