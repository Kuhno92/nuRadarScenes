#!/usr/bin/env python

import data_management.tfrecord_manager as tfrecord_manager
import numpy as np
from transforms3d import quaternions, affines, euler

class GridElement:

    def __init__(self):
        self.locations = []

class Preprocessing:

    def __init__(self):
        print('Preprocessing Initialized')
        self.grid = []
        self.grid_size_x = 100  # 100m
        self.grid_size_y = 100  # 100m
        self.grid_element_size_x = 0.2  # 50cm
        self.grid_element_size_y = 0.2  # 50cm
        self.grid_elements_x = self.grid_size_x / self.grid_element_size_x + 1  # 100.5m needs to be uneven to have a centered car location
        self.grid_elements_y = self.grid_size_y / self.grid_element_size_y + 1  # 100.5m needs to be uneven to have a centered car location

        self.lastRotation = None
        self.lastTransformation = None
        self.counter = 0

        #init grid
        for i in range(int(self.grid_elements_y)):
            self.grid.append([GridElement() for i in range(int(self.grid_elements_x))])

    def getRadarGrid(self, pc, pose):
        self.updateGrid(pc, pose)
        return self.grid

    def updateGrid(self, pc, pose):
        #calc trans. matrix
        H = None
        if self.lastRotation is not None:
            R_last = self.lastRotation
            R_cur = np.asarray(pose['rotation'])
            R_diff = np.asarray(euler.quat2euler(R_cur)) - np.asarray(euler.quat2euler(R_last))
            R = euler.euler2mat(R_diff[0], R_diff[1], R_diff[2])

            T = np.asarray(pose['translation']) - self.lastTransformation
            T = np.dot(quaternions.quat2mat(R_cur).transpose(), T)

            H = affines.compose(-T, R.transpose(), np.ones(3))

        self.lastRotation = np.asarray(pose['rotation'])
        self.lastTransformation = np.asarray(pose['translation'])


        old_points = []
        #update existing grid points
        for row in self.grid:
            for e in row:
                i = 0
                for _ in e.locations:
                    old_points.append(e.locations.pop(i))
                    i = i + 1

        for p in old_points:
            p[0:3] = np.dot(H, np.append(p[0:3], 1))[0:3]
            self.gridInsert(p)

        # add new pc to the grid
        if self.counter < 2 or True:
            for p in pc:
                self.gridInsert(p)
        self.counter = self.counter + 1

        return self.grid

    def gridInsert(self, p):
        if p[0] < self.grid_size_x/2 and p[1] < self.grid_size_y/2 and p[0] > -self.grid_size_y/2 and p[1] > -self.grid_size_y/2:
            y_index = int(self.grid_elements_y / 2 + p[1] // self.grid_element_size_y)
            x_index = int(self.grid_elements_x / 2 + p[0] // self.grid_element_size_x)
            self.grid[y_index][x_index].locations.append(p)

    def addGrid2File(self):
        img = []
        i = 0
        for row in self.grid:
            img.append([])
            for e in row:
                img[i].append(len(e.locations))
            i = i + 1

        tfrecord_manager.addFrame(img, img)

    def closeFile(self):
        tfrecord_manager.writeFile()

    def readFile(self):
        tfrecord_manager.readFile()