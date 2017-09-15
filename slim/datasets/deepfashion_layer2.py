# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides data for the deepfasion dataset.

The dataset scripts used to create the dataset can be found at:
tensorflow/models/slim/datasets/download_and_convert_flowers.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from collections import defaultdict
from datasets import dataset_utils

slim = tf.contrib.slim

_FILE_PATTERN = 'deepfashion_l2_%s_*.tfrecord'

SPLITS_TO_SIZES = {'train': 235150, 'validation': 26128}

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying size.',
    'first_class_label': 'A single integer between 0 and 2',
    'second_class_label': 'A single integer between 0 and 19'
}


def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
  """Gets a dataset tuple with instructions for reading flowers.

  Args:
    split_name: A train/validation split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources.
      It is assumed that the pattern contains a '%s' string so that the split
      name can be inserted.
    reader: The TensorFlow reader type.

  Returns:
    A `Dataset` namedtuple.

  Raises:
    ValueError: if `split_name` is not a valid train/validation split.
  """
  if split_name not in SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)

  if not file_pattern:
    file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

  # Allowing None in the signature so that dataset_factory can use the default.
  if reader is None:
    reader = tf.TFRecordReader

  keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
      'image/class/first_class_label': tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
      'image/class/second_class_label': tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
  }

  items_to_handlers = {
      'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
      'first_class_label': slim.tfexample_decoder.Tensor('image/class/first_class_label'),
      'second_class_label': slim.tfexample_decoder.Tensor('image/class/second_class_label'),
  }

  decoder = slim.tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  if dataset_utils.has_labels_l2(dataset_dir):
    first_class_labels_to_names, second_class_labels_to_names = dataset_utils.read_label_files_l2(dataset_dir)
    # get L1 class start and end index
    first_class_start_end_index = defaultdict(dict)
    first_class_names = first_class_labels_to_names['class1'].values()
    for first_class_name in first_class_names:
        class_index = [int(x[0]) for x in second_class_labels_to_names['class1'].items() if x[1] == first_class_name]
        first_class_start_end_index[first_class_name]["start"] = min(class_index)
        first_class_start_end_index[first_class_name]["end"] = max(class_index)
    first_class_count = len(first_class_labels_to_names['class1'].keys())
    second_class_count = len(second_class_labels_to_names['class2'].keys())
    return slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=reader,
      decoder=decoder,
      num_samples=SPLITS_TO_SIZES[split_name],
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      num_first_classes=first_class_count,
      num_second_classes=second_class_count,
      first_class_labels_to_names=first_class_labels_to_names,
      second_class_labels_to_names=second_class_labels_to_names,
      first_class_start_end_index=first_class_start_end_index,
    )
  else:
      raise ValueError("Cannot find first and second classes label files in {} directory!".format(dataset_dir))
