import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3

def animate_2D(framedata, xliml,xlimu, yliml,ylimu):
    """
    Make a 2D animation from the 2D array 'framedata'.
    The first dimension has to be time or frame index, 
    the second dimension consists of x and y coordinates 
    of a given frame.
    """

    framedata= np.array(framedata)
    fig, ax = plt.subplots()
    xdata, ydata = [], []
    ln, = plt.plot([], [],'ro')
    
    def init():
        ax.set_xlim(xliml,xlimu)
        ax.set_ylim(yliml,ylimu)
        return ln,

    def update(frame,framedata):    
        xdata = framedata[frame,:,0]
        ydata = framedata[frame,:,1] 
        ln.set_data(xdata, ydata)
        return ln,

    ani = FuncAnimation(fig, update, frames=len(framedata), fargs = [framedata],
                        init_func=init, blit=True)
    plt.show()

