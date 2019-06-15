#!/usr/bin/env python

from nuscenes.nuscenes import NuScenes

import evaluation
import data_ingest
import preprocessing

if __name__== "__main__":
    dr = data_ingest.DataReader()
    eval = evaluation.Evaluation()
    preproc = preprocessing.Preprocessing()

    while True:
        pc, pose = dr.getNextRadarPCL()
        if pc is None:
            eval.reset()
            dr.nextScene()
            preproc = preprocessing.Preprocessing()
            pc, pose = dr.getNextRadarPCL()
        grid = preproc.getRadarGrid(pc, pose)
        eval.plotPcl(pc, 2)
        eval.plotGrid(grid)
        eval.plotTrajectory(pose)
        eval.draw()