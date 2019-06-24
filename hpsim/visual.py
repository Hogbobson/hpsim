#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 19:46:43 2019

@author: hogbobson
"""

import numpy as np
import h5py
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import pyplot as plt
from matplotlib import animation as anim

    
def no_plot(ensemble):
    pass


def standard_plot(ensemble):
    n = ensemble['number of objects']
    plt.figure(num = 0, figsize = (8,8))
    for i in range(n):
        plt.plot(ensemble['r data'][i,0,:], ensemble['r data'][i,1,:], '-', \
                 label = ensemble['label'][i], \
                 color = (abs(np.sin(i)), 0 + i/n, 1-i/n))
    plt.xlabel('x')
    plt.ylabel('y')
    xlim = np.max(ensemble['r data'])*1.1
    xlim = 5*1e11
    plt.axis((-xlim, xlim, -xlim, xlim))
    plt.legend()
    plt.show()


def mkfile(data, name):
    fname = name + '.hdf5'
    f = h5py.File(fname, 'w')
    f.create_dataset(name, data = data)
    f.close()

    
def simple_anim(everything):
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    
    line, = ax.plot([], [], [], 'o')
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    
    iteras = int(everything['current time'] / everything['dt'] + 1)
    
    def init():
        line.set_data([], [], [])
        time_text.set_text('')
        return line, time_text
    
    
    def animate(i):
        thisx = everything['ensemble']['r data'][:, 0, i]
        thisy = everything['ensemble']['r data'][:, 1, i]
        thisz = everything['ensemble']['r data'][:, 2, i]
    
        line.set_data(thisx, thisy, thisz)
        time_text.set_text(time_template % (i*everything['dt']))
        return line, time_text
    
    ani = anim.FuncAnimation(fig, animate, np.arange(1, iteras),
                                  interval=25, blit=True, init_func=init)
            
    fig.show()