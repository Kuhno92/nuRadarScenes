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
        pc = dr.getNextRadarPCL()
        grid = preproc.getRadarGrid(pc)
        eval.plot_pcl(pc, 2)
