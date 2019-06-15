#!/usr/bin/env python

from nuscenes.nuscenes import NuScenes
from pypcd import pypcd
import numpy as np
from transforms3d import quaternions, affines
import os
import itertools

class DataReader:
    def __init__(self):
        self.nusc = NuScenes(version='v1.0-mini', dataroot='../data/sets/nuscenes', verbose=False)
        self.sceneID = 0
        self.scene = self.nusc.scene[self.sceneID]
        self.current_sample = self.nusc.get('sample', self.scene['first_sample_token'])
        print('Data Reader Initialized')

    def getNextRadarPCL(self):
        radar_front_data = self.nusc.get('sample_data', self.current_sample['data']['RADAR_FRONT'])
        radar_front_left_data = self.nusc.get('sample_data', self.current_sample['data']['RADAR_FRONT_LEFT'])
        radar_front_right_data = self.nusc.get('sample_data', self.current_sample['data']['RADAR_FRONT_RIGHT'])
        radar_back_left_data = self.nusc.get('sample_data', self.current_sample['data']['RADAR_BACK_LEFT'])
        radar_back_right_data = self.nusc.get('sample_data', self.current_sample['data']['RADAR_BACK_RIGHT'])

        radar_front_calib = self.nusc.get('calibrated_sensor',
                                    self.nusc.get('sample_data', self.current_sample['data']['RADAR_FRONT'])[
                                        'calibrated_sensor_token'])
        radar_front_left_calib = self.nusc.get('calibrated_sensor',
                                     self.nusc.get('sample_data', self.current_sample['data']['RADAR_FRONT_LEFT'])[
                                         'calibrated_sensor_token'])
        radar_front_right_calib = self.nusc.get('calibrated_sensor',
                                     self.nusc.get('sample_data', self.current_sample['data']['RADAR_FRONT_RIGHT'])[
                                         'calibrated_sensor_token'])
        radar_back_left_calib = self.nusc.get('calibrated_sensor',
                                     self.nusc.get('sample_data', self.current_sample['data']['RADAR_BACK_LEFT'])[
                                         'calibrated_sensor_token'])
        radar_back_right_calib = self.nusc.get('calibrated_sensor',
                                     self.nusc.get('sample_data', self.current_sample['data']['RADAR_BACK_RIGHT'])[
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

        ego_pose = self.nusc.get('ego_pose', radar_front_data['ego_pose_token'])

        if self.current_sample['next'] == "":
            self.current_sample = self.nusc.get('sample', self.scene['first_sample_token'])
            pc = None
        else:
            self.current_sample = self.nusc.get('sample', self.current_sample['next'])

        return pc, ego_pose

    def nextScene(self):
        self.sceneID = self.sceneID + 1
        if self.sceneID >= len(self.nusc.scene):
            self.sceneID = 0
        self.scene = self.nusc.scene[self.sceneID]
        self.current_sample = self.nusc.get('sample', self.scene['first_sample_token'])
