#!/usr/bin/env python

import numpy as np
import random

class Preprocessing:
    grid = []
    grid_element_x = 0.5 #50cm
    grid_element_y = 0.5 #50cm
    grid_size_x = grid_element_x*100*2*100 #100m
    grid_size_y = grid_element_y*100*2*100 #100m

    feature_vector = (0,0,0,0) # (x,y,z,rcs)

    def __init__(self):
        print('Preprocessing Initialized')

        #init grid
        for i in range(int(self.grid_size_y)):
            print(i)
            self.grid.append([[] for i in range(int(self.grid_size_x))])

    def getRadarGrid(self, pc):
        self.updateGrid(pc)
        return self.grid

    def updateGrid(self, pc):
        print('Updating Grid ...')
        for p in pc:
            #[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('dyn_prop', 'i1'), ('id', '<i2'), ('rcs', '<f4'), ('vx', '<f4'), ('vy', '<f4'), ('vx_comp', '<f4'), ('vy_comp', '<f4'), ('is_quality_valid', 'i1'), ('ambig_state', 'i1'), ('x_rms', 'i1'), ('y_rms', 'i1'), ('invalid_state', 'i1'), ('pdh0', 'i1'), ('vx_rms', 'i1'), ('vy_rms', 'i1')]
            self.grid[int(p[1]//self.grid_element_y)][int(p[0]//self.grid_element_x)].append((p[0],p[1],p[2],p[5]))
        return self.grid
