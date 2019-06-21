import tensorflow as tf
import numpy as np
import pickle

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

train_filename = 'train.tfrecord'  # address to save the TFRecords file
# open the TFRecords file
writer = None

def addFrame(img, label):
    global writer

    if writer == None:
        writer = tf.io.TFRecordWriter(train_filename)

    feature = {
        'train/label': _bytes_feature(tf.compat.as_bytes(pickle.dumps(label))),
        'train/image': _bytes_feature(tf.compat.as_bytes(pickle.dumps(img)))
    }
    tf_example = tf.train.Example(
        features=tf.train.Features(feature=feature)
    )
    writer.write(tf_example.SerializeToString())


def writeFile():
    writer.close()

def readFile():

    # 1-2. Check if the data is stored correctly
    # open the saved file and check the first entries
    for serialized_example in tf.data.TFRecordDataset('train.tfrecord').take(10):
        features = tf.io.parse_single_example(
            serialized_example,
            features={
                'train/label': tf.io.FixedLenFeature([], tf.string),
                'train/image': tf.io.FixedLenFeature([], tf.string)
            }
        )
        features['train/image'] = pickle.loads(features['train/image'].numpy())
        features['train/label'] = pickle.loads(features['train/label'].numpy())
