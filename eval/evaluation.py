#!/usr/bin/env python

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np

from transforms3d import quaternions, affines

class Evaluation:

    fig = None
    ax = None
    ax2 = None
    ax3 = None
    scat = None
    imshow = None
    line = None

    def __init__(self):
        print('Evaluation Initialized')
        self.tightened = False
        self.lastRotation = None
        self.lastTranslation = None
        self.xdata = []
        self.ydata = []

        plt.ion()
        self.fig = plt.figure(figsize=(12, 8))


    def plotTrajectory(self, pose):
        if self.fig == None:
            plt.ion()
            self.fig = plt.figure()
        if self.ax3 == None:
            self.ax3 = self.fig.add_subplot(223)

        if self.line == None:
            self.line, = self.ax3.plot(self.xdata, self.ydata, marker='.')
            plt.show()

        H = None
        if self.lastRotation is not None:
            R_last = self.lastRotation
            T_last = self.lastTranslation
            T_cur = np.asarray(pose['translation'])
            H = affines.compose(T_cur - T_last, np.eye(3), np.ones(3))

        self.lastRotation = np.asarray(quaternions.quat2mat(pose['rotation']))
        self.lastTranslation = np.asarray(pose['translation'])

        if H is None:
            self.xdata.append(self.lastTranslation[0])
            self.ydata.append(self.lastTranslation[1])
        else:
            res = np.dot(H, np.append([self.xdata[-1], self.ydata[-1], 0], 1))[0:3]
            self.xdata.append(res[0])
            self.ydata.append(res[1])

        if self.xdata[0] == self.lastTranslation[0] and self.ydata[0] == self.lastTranslation[1]:
            self.xdata = [self.lastTranslation[0]]
            self.ydata = [self.lastTranslation[1]]
        self.line.set_xdata(self.xdata)
        self.line.set_ydata(self.ydata)

        self.ax3.set_xlim([np.asarray(pose['translation'])[0] - 100, np.asarray(pose['translation'])[0] + 100])
        self.ax3.set_ylim([np.asarray(pose['translation'])[1] - 100, np.asarray(pose['translation'])[1] + 100])

    def plotGrid(self, grid):
        img = []
        i = 0
        for row in grid:
            img.append([])
            for e in row:
                img[i].append(len(e.locations))
            i = i + 1
        if self.fig == None:
            plt.ion()
            self.fig = plt.figure()
        if self.ax2 == None:
            self.ax2 = self.fig.add_subplot(222)
            self.imshow = self.ax2.imshow(img, origin='centre', vmax=2)
        else:
            self.imshow.set_data(img)

    def get_pt_coordinates(self, pc):
        x = [e[0] for e in pc]
        y = [e[1] for e in pc]
        z = [e[2] for e in pc]
        return x, y, z

    def plotPcl(self, pc, dimension=2):
        if self.fig == None:
            plt.ion()
            self.fig = plt.figure()
        if self.ax == None:
            self.ax = self.fig.add_subplot(221, projection='3d') if dimension == 3 else self.fig.add_subplot(221)
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z') if dimension == 3 else None
            self.ax.set_xlim([-150, 150])
            self.ax.set_ylim([-100, 100])
            x, y, z = self.get_pt_coordinates(pc)
            self.scat = self.ax.scatter(x, y, z, c='r', marker='o') if dimension == 3 else self.ax.scatter(x, y, c='black', s=0.5,                                                                                        marker='o')
            self.ax.add_patch(Rectangle(xy=(-0.562, 1.256/2), width=3.974, height=1.256, linewidth=1, color='blue', fill=True))
            plt.show()
        else:
            x, y, z = self.get_pt_coordinates(pc)
            self.scat.set_offsets(np.c_[x,y])

    def draw(self):
        if self.tightened == False:
            self.tightened = True
            plt.tight_layout()
        if self.fig != None:
            self.fig.canvas.draw_idle()
            plt.pause(0.001)

    def reset(self):
        self.xdata = []
        self.ydata = []
        self.lastRotation = None
        self.lastTranslation = None