#!/usr/bin/env python

from eval import evaluation
from data_management import data_ingest, preprocessing
import argparse
from model import model, train
import tensorflow as tf
import time

def writeDataset():
    dr = data_ingest.DataReader()
    eval = evaluation.Evaluation()
    preproc = preprocessing.Preprocessing()

    while True:
        start = time.time()
        pc, pose = dr.getNextRadarPCL()

        if pc is None:
            eval.reset()
            if dr.nextScene() == 0:
                preproc.closeFile()
                exit(0)
            preproc = preprocessing.Preprocessing()
            pc, pose = dr.getNextRadarPCL()

        grid = preproc.getRadarGrid(pc, pose)
        preproc.addGrid2File()

        #eval.plotPcl(pc, 2)
        #eval.plotGrid(grid)
        #eval.plotTrajectory(pose)
        #eval.draw()

        end = time.time()
        print("Time per frame: {:1.4f}s".format(end - start))

def trainModel():

    preproc = preprocessing.Preprocessing()
    preproc.readFile()


def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--write-dataset', action='store_true')
    parser.add_argument('--train-model', action='store_true')

    return parser.parse_args()

if __name__== "__main__":

    args = parse_args()

    if args.write_dataset:
        writeDataset()

    if args.train_model:
        trainModel()


    model = model.MyModel()

    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Add a channels dimension
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

    train.train(model, train_ds, test_ds)

    exit(0)