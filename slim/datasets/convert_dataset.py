# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Converts Deepfashion data to TFRecords of TF-Example protos.

This scripts creates two TFRecord datasets: one for train
and one for val. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import glob

import tensorflow as tf
import numpy as np

from collections import OrderedDict, defaultdict
from datasets import dataset_utils

from classification_config import cfg

# The number of images in the validation set.
_NUM_VALIDATION_RATIO = cfg.DATASET.TRAIN_RATIO

# Seed for repeatability.
_RANDOM_SEED = 0

# The number of shards per dataset split.
_NUM_SHARDS = cfg.DATASET.SHARDS

class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _get_sub_dirs(path):
  # return last level directories list
  dirs = []
  for child in os.listdir(path):
    child_full = os.path.join(path, child)
    if os.path.isdir(child_full):
      dirs += _get_sub_dirs(child_full)
    else:
      return [path]
  return dirs


def _get_filenames_and_classes(dataset_dir):
  """Returns a list of filenames and inferred class names.

  Args:
    dataset_dir: A directory containing a set of subdirectories representing
      first class names. Each subdirectory representing second class names and within it should be PNG or JPG encoded images.

  Returns:
    A list of image file paths, relative to `dataset_dir`
    a list of class name from first class
  """
  images = []
  class_names = []

  sub_dirs = _get_sub_dirs(dataset_dir)
  for sub_dir in sub_dirs:
    filenames = glob.glob(os.path.join(sub_dir, "*.jpg"))
    if len(filenames):
      images += filenames
      # get dirname path relative to root path
      filenames_relative = map(lambda x: os.path.dirname(x[len(dataset_dir)+1:]), filenames)
      # intermediate directory names are class label names
      cur_class_names = map(lambda x: x.split("/"), set(filenames_relative))
      class_names += cur_class_names
  return images, np.squeeze(class_names)


def _get_dataset_filename(outputdir, split_name, shard_id):
  output_filename = '%s_%s_%05d-of-%05d.tfrecord' % (
      cfg.DATASET.DATASET_PREFIX, split_name, shard_id, _NUM_SHARDS)
  return os.path.join(outputdir, output_filename)


def _convert_dataset(split_name, filenames, class_names_to_ids, outputdir, dataset_dir):
  """Converts the given filenames to a TFRecord dataset.

  Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    filenames: A list of absolute paths to jpg images.
    class_names_to_ids: A dictionary from class names (strings) to ids
      (integers).
    outputdir: The directory where the converted datasets are stored.
    dataset_dir: The directory where the original datasets stored
  """
  assert split_name in ['train', 'validation']

  num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))
  hierarchy_levels = len(class_names_to_ids.keys())
  with tf.Graph().as_default():
    image_reader = ImageReader()

    with tf.Session('') as sess:

      for shard_id in range(_NUM_SHARDS):
        output_filename = _get_dataset_filename(
            outputdir, split_name, shard_id)

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
          for i in range(start_ndx, end_ndx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                i+1, len(filenames), shard_id))
            sys.stdout.flush()

            # Read the filename:
            image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
            height, width = image_reader.read_image_dims(sess, image_data)
            labels = os.path.dirname(filenames[i][len(dataset_dir)+1:]).split("/")
            label_ids = [class_names_to_ids['class'+str(level+1)][labels[level]] for level in range(hierarchy_levels)]
            example = dataset_utils.image_to_tfexample_hdcnn(
                image_data, b'jpg', height, width, label_ids)
            tfrecord_writer.write(example.SerializeToString())
            print("\nimage", filenames[i])
            print("Hierarchical class names: %s" %(",".join(labels)))
            print("Hierarchical class ids: %s" %(",".join(np.array(label_ids).astype(str))))
  sys.stdout.write('\n')
  sys.stdout.flush()


def _dataset_exists(outputdir):
  for split_name in ['train', 'validation']:
    for shard_id in range(_NUM_SHARDS):
      output_filename = _get_dataset_filename(
          outputdir, split_name, shard_id)
      if not tf.gfile.Exists(output_filename):
        return False
  return True


def run(dataset_dir, outputdir):
  """Runs the download and conversion operation.

  Args:
    dataset_dir: The dataset directory where the dataset is stored.
    outputdir: The tf record file output dir
  """
  if not tf.gfile.Exists(outputdir):
    tf.gfile.MakeDirs(outputdir)

  if _dataset_exists(outputdir):
    print('Dataset files already exist. Exiting without re-creating them.')
    return

  photo_filenames, class_names = _get_filenames_and_classes(dataset_dir)
  print("Get jpg image filenames and its class labels done!")
  # get class hierarchy levels
  hierarchy_levels = class_names.shape[1]
  class_names_to_ids = defaultdict(dict)
  for level in range(hierarchy_levels):
    class_level = "class"+str(level+1)
    unique_class_names = list(OrderedDict.fromkeys([i[level] for i in class_names]))
    class_names_to_ids[class_level] = dict(zip(unique_class_names, range(len(unique_class_names))))
  # write the labels file:
  dataset_utils.write_label_file_hdcnn(class_names, outputdir)
  print("Saving label file done!")

  # Divide into train and test:
  random.seed(_RANDOM_SEED)
  random.shuffle(photo_filenames)
  # get number of training images
  train_count = int(len(photo_filenames)*_NUM_VALIDATION_RATIO)
  training_filenames = photo_filenames[:train_count]
  validation_filenames = photo_filenames[train_count:]

  # First, convert the training and validation sets.
  _convert_dataset('train', training_filenames, class_names_to_ids, outputdir, dataset_dir)
  _convert_dataset('validation', validation_filenames, class_names_to_ids, outputdir, dataset_dir)

  print('\nFinished converting the Deepfashion dataset!')
