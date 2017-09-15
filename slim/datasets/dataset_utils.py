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
"""Contains utilities for downloading and converting datasets."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tarfile

from collections import defaultdict, OrderedDict
from six.moves import urllib
import tensorflow as tf
import numpy as np

LABELS_FILENAME = 'labels.txt'

# used for 2 layers hierarchical classification
FIRST_LABELS_FILENAME = 'first_class_labels.txt'
SECOND_LABELS_FILENAME = 'second_class_labels.txt'

def int64_feature(values):
  """Returns a TF-Feature of int64s.

  Args:
    values: A scalar or list of values.

  Returns:
    A TF-Feature.
  """
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
  """Returns a TF-Feature of bytes.

  Args:
    values: A string.

  Returns:
    A TF-Feature.
  """
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def float_feature(values):
  """Returns a TF-Feature of floats.

  Args:
    values: A scalar of list of values.

  Returns:
    A TF-Feature.
  """
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def image_to_tfexample(image_data, image_format, height, width, class_id):
  return tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': bytes_feature(image_data),
      'image/format': bytes_feature(image_format),
      'image/class/label': int64_feature(class_id),
      'image/height': int64_feature(height),
      'image/width': int64_feature(width),
  }))


def image_to_tfexample_layer2(image_data, image_format, height, width, first_class_id, second_class_id):
  return tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': bytes_feature(image_data),
      'image/format': bytes_feature(image_format),
      'image/class/first_class_label': int64_feature(first_class_id),
      'image/class/second_class_label': int64_feature(second_class_id),
      'image/height': int64_feature(height),
      'image/width': int64_feature(width),
  }))


def image_to_tfexample_hdcnn(image_data, image_format, height, width, class_ids):
  return tf.train.Example(features=tf.train.Features(feature={
    'image/encoded': bytes_feature(image_data),
    'image/format': bytes_feature(image_format),
    'image/height': int64_feature(height),
    'image/width': int64_feature(width),
    'image/class/class_labels': int64_feature(class_ids),
  }))

def download_and_uncompress_tarball(tarball_url, dataset_dir):
  """Downloads the `tarball_url` and uncompresses it locally.

  Args:
    tarball_url: The URL of a tarball file.
    dataset_dir: The directory where the temporary files are stored.
  """
  filename = tarball_url.split('/')[-1]
  filepath = os.path.join(dataset_dir, filename)

  def _progress(count, block_size, total_size):
    sys.stdout.write('\r>> Downloading %s %.1f%%' % (
        filename, float(count * block_size) / float(total_size) * 100.0))
    sys.stdout.flush()
  filepath, _ = urllib.request.urlretrieve(tarball_url, filepath, _progress)
  statinfo = os.stat(filepath)
  print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dataset_dir)


def write_label_file(labels_to_class_names, dataset_dir,
                     filename=LABELS_FILENAME):
  """Writes a file with the list of class names.

  Args:
    labels_to_class_names: A map of (integer) labels to class names.
    dataset_dir: The directory in which the labels file should be written.
    filename: The filename where the class names are written.
  """
  labels_filename = os.path.join(dataset_dir, filename)
  with tf.gfile.Open(labels_filename, 'w') as f:
    for label in labels_to_class_names:
      class_name = labels_to_class_names[label]
      f.write('%d:%s\n' % (label, class_name))


def write_label_file_layer2(first_second_class_names, dataset_dir, first_class_filename=FIRST_LABELS_FILENAME, second_class_filename=SECOND_LABELS_FILENAME):
  """Writes a file with the list of first and second class names in hierarchical classification.

  Args:
    first_second_class_names: A list of first class name and second class tuples.
    dataset_dir: The directory in which the labels file should be written.
    first_class_filename: The filename where the first class names are written.
    second_class_filename: The filename where the second class names are written together with corresponding first class names
  """
  first_class_labels_filename = os.path.join(dataset_dir, first_class_filename)
  second_class_labels_filename = os.path.join(dataset_dir, second_class_filename)
  first_class_names = list(OrderedDict.fromkeys([i[0] for i in first_second_class_names]))
  with tf.gfile.GFile(first_class_labels_filename, 'w') as f:
    for idx, label in enumerate(first_class_names):
      f.write('%d:%s\n' %(idx, label))
  f.close()
  with tf.gfile.GFile(second_class_labels_filename, 'w') as of:
    for idx, label in enumerate(first_second_class_names):
      of.write('%d:%s:%s\n' %(idx, label[0], label[1]))


def write_label_file_hdcnn(class_names_list, dataset_dir, labels_filename=LABELS_FILENAME):
  """Writes a file with the list of hierarchical class names.

  Args:
    class_names_list: A list of classes class tuples from first class to last.
    dataset_dir: The directory in which the labels file should be written.
    labels_filename: The filename where the labels are written.
  """
  class_names = np.asarray(class_names_list).astype(str)
  labels_filename = os.path.join(dataset_dir, labels_filename)
  with tf.gfile.GFile(labels_filename, 'w') as of:
    for idx, label in enumerate(class_names):
      of.write('%d:%s\n' %(idx, ":".join(class_names[idx])))


def has_labels(dataset_dir, filename=LABELS_FILENAME):
  """Specifies whether or not the dataset directory contains a label map file.

  Args:
    dataset_dir: The directory in which the labels file is found.
    filename: The filename where the class names are written.

  Returns:
    `True` if the labels file exists and `False` otherwise.
  """
  return tf.gfile.Exists(os.path.join(dataset_dir, filename))


def has_labels_l2(dataset_dir, first_class_filename=FIRST_LABELS_FILENAME, second_class_filename=SECOND_LABELS_FILENAME):
  """Check if label files exist for hierarchical classification"""
  return tf.gfile.Exists(os.path.join(dataset_dir, first_class_filename)) and tf.gfile.Exists(os.path.join(dataset_dir, second_class_filename))


def read_label_file(dataset_dir, filename=LABELS_FILENAME):
  """Reads the labels file and returns a mapping from ID to class name.

  Args:
    dataset_dir: The directory in which the labels file is found.
    filename: The filename where the class names are written.

  Returns:
    A map from a label (integer) to class name.
  """
  labels_filename = os.path.join(dataset_dir, filename)
  with tf.gfile.Open(labels_filename, 'rb') as f:
    lines = f.read().decode()
  lines = lines.split('\n')
  lines = filter(None, lines)

  labels_to_class_names = {}
  for line in lines:
    index = line.index(':')
    labels_to_class_names[int(line[:index])] = line[index+1:]
  return labels_to_class_names


def read_label_file_hdcnn(dataset_dir, labels_filename=LABELS_FILENAME):
  """Reads the class label file and returns a mapping from ID to class name with hierarchy class as primary key.

  Args:
    dataset_dir: The directory in which the labels file is found.
    filename: The filename where the class names are written.

  Returns:
    A map from a label (integer) to class name. If hierarchical classes are specified, use hierarchy class as its primary key.
  """
  labels_filename = os.path.join(dataset_dir, labels_filename)
  with tf.gfile.Open(labels_filename, 'rb') as f:
    lines = f.read().decode()
  lines = lines.split('\n')
  lines = filter(None, lines)

  n_hierarchy = len(lines[0].split(":")) - 1
  hierarchy_keys = ['class'+str(i) for i in range(1, n_hierarchy+1)]
  labels_to_class_names = defaultdict(dict)

  for line in lines:
    items = line.split(":")
    for idx, hierarchy_key in enumerate(hierarchy_keys):
      labels_to_class_names[hierarchy_key][items[0]] = items[idx+1]
  return labels_to_class_names


def read_label_files_l2(dataset_dir, first_class_filename=FIRST_LABELS_FILENAME, second_class_filename=SECOND_LABELS_FILENAME):
  first_class_labels_to_class_names = read_label_file_hdcnn(dataset_dir, labels_filename=first_class_filename)
  second_class_labels_to_class_names = read_label_file_hdcnn(dataset_dir, labels_filename=second_class_filename)
  print("Layer 1 category id-to-name mapping: ", first_class_labels_to_class_names)
  print("Layer 2 category id-to-name mapping: ", second_class_labels_to_class_names)
  return first_class_labels_to_class_names, second_class_labels_to_class_names

