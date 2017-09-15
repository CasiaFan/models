from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import tensorflow as tf
import numpy as np

from collections import Counter, defaultdict


slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
    'batch_size', 1, 'The number of samples in each batch.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/home/arkenstone/tensorflow/workspace/models/slim/models/deepfashion_l2/frozen_graph.pb',
    'The directory where the model was written to or an absolute path to a '
    'frozen graph pb file.')

tf.app.flags.DEFINE_string(
    'image', '/home/arkenstone/Documents/img_00000064.jpg', 'The path where the test image files are stored.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_resnet_v2_layer2', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_string(
    'label_file', '/home/arkenstone/tensorflow/workspace/models/slim/models/deepfashion_l2/data/second_class_labels.txt', 'The label file with integer index as key and first/second/.. class name as value'
)

FLAGS = tf.app.flags.FLAGS

# input image size corresponding to model
input_image_size = {"inception_resnet_v2_layer2": [299, 299]}


def preprocess_inception(image_np, central_fraction=0.875):
    image_height, image_width, image_channel = image_np.shape
    if central_fraction:
        bbox_start_h = int(image_height * (1 - central_fraction) / 2)
        bbox_end_h = int(image_height - bbox_start_h)
        bbox_start_w = int(image_width * (1 - central_fraction) / 2)
        bbox_end_w = int(image_width - bbox_start_w)
        image_np = image_np[bbox_start_h:bbox_end_h, bbox_start_w:bbox_end_w]
    # normalize
    image_np = 2 * (image_np / 255.) - 1
    return image_np

preprocess_factory = {"inception_resnet_v2_layer2": preprocess_inception}

def _read_label_file(label_path):
  """Reads the class label file and returns a mapping from ID to class name with hierarchy class as primary key.

  Returns:
    A map from a label (integer) to class name. If hierarchical classes are specified, use hierarchy class as its primary key.
  """
  with tf.gfile.Open(label_path, 'rb') as f:
    lines = f.read().decode()
  lines = lines.split('\n')
  lines = filter(None, lines)

  n_hierarchy = len(lines[0].split(":")) - 1
  hierarchy_keys = ['class'+str(i) for i in range(1, n_hierarchy+1)]
  labels_to_class_names = defaultdict(dict)

  for line in lines:
    items = line.split(":")
    for idx, hierarchy_key in enumerate(hierarchy_keys):
      labels_to_class_names[hierarchy_key][int(items[0])] = items[idx+1]
  return labels_to_class_names


def _get_class_start_end_index(labels_to_class_names):
    class_start_end_index = defaultdict(dict)
    for i in range(1, len(labels_to_class_names.keys())):
        class_level = 'class' + str(i)
        class_start_end_index[class_level] = defaultdict(dict)
        for class_name in list(set(labels_to_class_names[class_level].values())):
            class_index = [int(x[0]) for x in labels_to_class_names[class_level].items() if x[1] == class_name]
            class_start_end_index[class_level][class_name]["start"] = min(class_index)
            class_start_end_index[class_level][class_name]["end"] = max(class_index)
    return class_start_end_index


def _statistic_category_count(labels_to_class_names):
    hierarchy_class_count_dict = defaultdict(dict)
    for class_name in labels_to_class_names.keys():
        hierarchy_class_count_dict[class_name] = dict(Counter(labels_to_class_names[class_name].values()))
    return hierarchy_class_count_dict


def _softmax(inputs, theta=1.0):
    y = np.array(inputs) * theta
    # subtract the max for numeric stability
    max_y = np.max(y)
    y -= max_y
    y = np.exp(y)
    sum_y = np.sum(y)
    return y/sum_y


class ClothingClassification():
    def __init__(self, FLAGS):
        try:
            self.image_height, self.image_width = input_image_size[FLAGS.model_name]
        except:
            raise ValueError("Your model must be within {}".format(",".join(input_image_size.keys())))
        # load model parameters from checkpoint pb file
        pb = tf.gfile.GFile(FLAGS.checkpoint_path, "rb")
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(pb.read())
        # set log verbosity level
        tf.logging.set_verbosity(tf.logging.INFO)
        # set gpu level
        gpu_ratio = 0.1
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_ratio, allow_growth=True)
        sess_config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=False)
        self.sess = tf.Session(graph=tf.import_graph_def(graph_def, name=''), config=sess_config)
        if FLAGS.label_file:
            self.labels_to_class_names = _read_label_file(FLAGS.label_file)
            self.class_start_end_index = _get_class_start_end_index(self.labels_to_class_names)
            self.hierarchy_class_count_dict = _statistic_category_count(self.labels_to_class_names)
        else:
            raise ValueError("You must supply the label file generated during tfrecords file generation for training")


    def close(self):
        self.sess.close()


    def classify(self, image, topk=1):
        """
        image: image file path
        topk: top k candidate categories
        """
        # read in image to ndarray
        image_np = cv2.imread(image)
        # preprocess as preprocessing factory doing
        preprocess_name = FLAGS.preprocessing_name or FLAGS.model_name
        preprocess_func = preprocess_factory[preprocess_name]
        image_np = preprocess_func(image_np)
        # resize
        image_np = cv2.resize(image_np, (self.image_height, self.image_width))
        # expand dims to [1, height, width, 3]
        image_np = np.expand_dims(image_np, 0)
        # input tensor
        input_tensor = self.sess.graph.get_tensor_by_name("input:0")
        # output tensor
        output = self.sess.graph.get_tensor_by_name("InceptionResnetV2/Logits/Outputs:0")
        logits = self.sess.run(output, feed_dict={input_tensor: image_np})
        # get first class count
        l1_class_names = list(set(self.labels_to_class_names['class1'].values()))
        l2_class_names = list(set(self.labels_to_class_names['class2'].values()))
        num_first_class = len(l1_class_names)
        logits_l1 = logits[0][0:num_first_class]
        logits_l2 = logits[0][num_first_class:]
        predictions_l1_scores = _softmax(logits_l1)
        for index, l1_class_name in enumerate(sorted(l1_class_names)):
            predictions_l2_scores = _softmax(logits_l2[self.class_start_end_index['class1'][l1_class_name]['start']:
            self.class_start_end_index['class1'][l1_class_name]['end'] + 1])
            predictions_l2_scores = predictions_l2_scores * (predictions_l1_scores[index])
            if index == 0:
                predictions_l2_cond_scores = predictions_l2_scores
            else:
                predictions_l2_cond_scores = np.concatenate([predictions_l2_cond_scores, predictions_l2_scores])
        l2_topk_pred_indices = np.argsort(predictions_l2_cond_scores)[-topk:][::-1]
        l2_topk_pred_names = np.array(l2_class_names)[l2_topk_pred_indices]
        predictions_l1_index = np.argmax(logits_l1)
        predictions_l1_names = np.array(l1_class_names)[predictions_l1_index]
        return predictions_l1_names, l2_topk_pred_names


def main(_):
  if not FLAGS.image:
    raise ValueError('You must supply the dataset directory with --dataset_dir')
  clothing_classifier = ClothingClassification(FLAGS)
  l1_names, l2_names = clothing_classifier.classify(FLAGS.image)
  print("Class1 name: ", l1_names)
  print("Class2 name: ", l2_names)

if __name__ == '__main__':
  tf.app.run()