"""Converts a deepfashion dataset to tf record file.

Usage:
$ python convert_deepfashion_l2_data.py \
    --dataset_dir=/startdt_data/clothing_data/DeepFashion/Category_and_Attribute_Prediction_Benchmark/image4classification
    --outputdir=models/deepfashion
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from datasets import convert_dataset

FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string(
    'dataset_dir', None,
    'The root directory storing Deepfashion categories and images')
tf.app.flags.DEFINE_string(
    'outputdir',
    None,
    'The directory where the output TFRecords and temporary files are saved.')


def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')
  if not FLAGS.outputdir:
    raise ValueError('You must define the output directory with --outpurdir')
  convert_dataset.run(FLAGS.dataset_dir, FLAGS.outputdir)

if __name__ == '__main__':
  tf.app.run()