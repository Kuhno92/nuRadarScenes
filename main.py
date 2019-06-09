#!/usr/bin/env python

from nuscenes.nuscenes import NuScenes
from pypcd import pypcd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
from transforms3d import quaternions, affines
import os
import time
import itertools

def get_pts(pc):
    x = [e[0] for e in pc]
    y = [e[1] for e in pc]
    z = [e[2] for e in pc]
    return x, y, z

fig = None
ax = None
scat = None
def plot_pcl(pc, dimension=2):
    global fig, ax, scat
    if fig == None:
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d') if dimension == 3 else fig.add_subplot(111)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z') if dimension == 3 else None
        ax.set_xlim([-150, 150])
        ax.set_ylim([-100, 100])
        x, y, z = get_pts(pc)
        scat = ax.scatter(x, y, z, c='r', marker='o') if dimension == 3 else ax.scatter(x, y, c='black', s=0.5,                                                                                        marker='o')
        ax.add_patch(Rectangle(xy=(-0.562, 1.256/2), width=3.974, height=1.256, linewidth=1, color='blue', fill=True))
        plt.show()
    else:
        x, y, z = get_pts(pc)
        scat.set_offsets(np.c_[x,y])
        fig.canvas.draw_idle()
        plt.pause(0.01)

def getRadarPCL(current_sample):
    radar_front_data = nusc.get('sample_data', current_sample['data']['RADAR_FRONT'])
    radar_front_left_data = nusc.get('sample_data', current_sample['data']['RADAR_FRONT_LEFT'])
    radar_front_right_data = nusc.get('sample_data', current_sample['data']['RADAR_FRONT_RIGHT'])
    radar_back_left_data = nusc.get('sample_data', current_sample['data']['RADAR_BACK_LEFT'])
    radar_back_right_data = nusc.get('sample_data', current_sample['data']['RADAR_BACK_RIGHT'])

    radar_front_calib = nusc.get('calibrated_sensor',
                                nusc.get('sample_data', current_sample['data']['RADAR_FRONT'])[
                                    'calibrated_sensor_token'])
    radar_front_left_calib = nusc.get('calibrated_sensor',
                                 nusc.get('sample_data', current_sample['data']['RADAR_FRONT_LEFT'])[
                                     'calibrated_sensor_token'])
    radar_front_right_calib = nusc.get('calibrated_sensor',
                                 nusc.get('sample_data', current_sample['data']['RADAR_FRONT_RIGHT'])[
                                     'calibrated_sensor_token'])
    radar_back_left_calib = nusc.get('calibrated_sensor',
                                 nusc.get('sample_data', current_sample['data']['RADAR_BACK_LEFT'])[
                                     'calibrated_sensor_token'])
    radar_back_right_calib = nusc.get('calibrated_sensor',
                                 nusc.get('sample_data', current_sample['data']['RADAR_BACK_RIGHT'])[
                                     'calibrated_sensor_token'])

    pc_f = pypcd.PointCloud.from_path(os.path.normpath('../data/sets/nuscenes/' + radar_front_data['filename']))
    pc_fl = pypcd.PointCloud.from_path(os.path.normpath('../data/sets/nuscenes/' + radar_front_left_data['filename']))
    pc_fr = pypcd.PointCloud.from_path(os.path.normpath('../data/sets/nuscenes/' + radar_front_right_data['filename']))
    pc_bl = pypcd.PointCloud.from_path(os.path.normpath('../data/sets/nuscenes/' + radar_back_left_data['filename']))
    pc_br = pypcd.PointCloud.from_path(os.path.normpath('../data/sets/nuscenes/' + radar_back_right_data['filename']))

    def toVecCoord(*x):
        y = np.asarray(x)
        y[0:3] = np.dot(H, np.append(y[0:3], 1))[0:3]
        return y

    R = quaternions.quat2mat(radar_front_calib['rotation'])
    T = radar_front_calib['translation']
    H = affines.compose(T, R, np.ones(3))
    pc_f = list(itertools.starmap(toVecCoord, pc_f.pc_data))

    R = quaternions.quat2mat(radar_front_left_calib['rotation'])
    T = radar_front_left_calib['translation']
    H = affines.compose(T, R, np.ones(3))
    pc_fl = list(itertools.starmap(toVecCoord, pc_fl.pc_data))

    R = quaternions.quat2mat(radar_front_right_calib['rotation'])
    T = radar_front_right_calib['translation']
    H = affines.compose(T, R, np.ones(3))
    pc_fr = list(itertools.starmap(toVecCoord, pc_fr.pc_data))
    R = quaternions.quat2mat(radar_back_left_calib['rotation'])
    T = radar_back_left_calib['translation']
    H = affines.compose(T, R, np.ones(3))
    pc_bl = list(itertools.starmap(toVecCoord, pc_bl.pc_data))
    R = quaternions.quat2mat(radar_back_right_calib['rotation'])
    T = radar_back_right_calib['translation']
    H = affines.compose(T, R, np.ones(3))
    pc_br = list(itertools.starmap(toVecCoord, pc_br.pc_data))

    pc = np.concatenate((pc_f, pc_fl, pc_fr, pc_bl, pc_br))

    return pc

if __name__== "__main__":
    nusc = NuScenes(version='v1.0-mini', dataroot='../data/sets/nuscenes', verbose=False)
    scene = nusc.scene[1]

    current_sample = nusc.get('sample', scene['first_sample_token'])
    while True:
        pc = getRadarPCL(current_sample)
        plot_pcl(pc, 2)
        if current_sample['next'] == "":
            current_sample = nusc.get('sample', scene['first_sample_token'])
            #break
        else:
            current_sample = nusc.get('sample', current_sample['next'])