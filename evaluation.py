#!/usr/bin/env python

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np

class Evaluation:
    def __init__(self):
        print('Evaluation Initialized')


    def get_pts(self, pc):
        x = [e[0] for e in pc]
        y = [e[1] for e in pc]
        z = [e[2] for e in pc]
        return x, y, z

    fig = None
    ax = None
    scat = None
    def plot_pcl(self, pc, dimension=2):
        if self.fig == None:
            plt.ion()
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d') if dimension == 3 else self.fig.add_subplot(111)
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z') if dimension == 3 else None
            self.ax.set_xlim([-150, 150])
            self.ax.set_ylim([-100, 100])
            x, y, z = self.get_pts(pc)
            self.scat = self.ax.scatter(x, y, z, c='r', marker='o') if dimension == 3 else self.ax.scatter(x, y, c='black', s=0.5,                                                                                        marker='o')
            self.ax.add_patch(Rectangle(xy=(-0.562, 1.256/2), width=3.974, height=1.256, linewidth=1, color='blue', fill=True))
            plt.show()
        else:
            x, y, z = self.get_pts(pc)
            self.scat.set_offsets(np.c_[x,y])
            self.fig.canvas.draw_idle()
            plt.pause(0.01)