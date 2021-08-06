import numpy as np

def store_1D_plotdata(x,y,fname):
        data = np.column_stack([x, y])
        datafile_path = "/home/vgs23/Pickle_files/{}.txt".format(fname)
        np.savetxt(datafile_path , data)#,fmt = ['%f','%f'])

def read_1D_plotdata(fname):
        data = np.loadtxt("/home/vgs23/Pickle_files/{}.txt".format(fname),dtype=complex)
        return data

def chunks(L, n): 
        return [L[x: x+n] for x in range(0, len(L), n)]

