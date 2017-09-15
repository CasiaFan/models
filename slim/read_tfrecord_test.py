import tensorflow as tf
import matplotlib.pyplot as plt

data_path = '/home/arkenstone/tensorflow/workspace/models/slim/models/deepfashion/data/deepfashion_train_00000-of-00005.tfrecord'  # address to save the hdf5 file
with tf.Session() as sess:
    feature = {'image/encoded': tf.FixedLenFeature([], tf.string),
           'image/format': tf.FixedLenFeature([], tf.string),
           'image/class/class_labels': tf.FixedLenFeature([2], tf.int64),
           'image/height': tf.FixedLenFeature([], tf.int64),
           'image/width': tf.FixedLenFeature([], tf.int64),
    }
    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)
    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)
    # Convert the image data from string back to the numbers
    image = tf.image.decode_jpeg(features['image/encoded'])
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # Cast label data into int32
    label = tf.cast(features['image/class/class_labels'], tf.int64)

    height = tf.cast(features['image/height'], tf.int32)
    width = tf.cast(features['image/width'], tf.int32)
    image = tf.reshape(image, [height, width, 3])
    # resize
    image = tf.image.resize_images(image, [224, 224])
    print image.get_shape()
    # images, labels = tf.train.batch([image, label], batch_size=1)
    images, labels = tf.train.shuffle_batch([image, label], batch_size=10, capacity=30, num_threads=1, min_after_dequeue=10)

    # Initialize all global and local variables
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    # Create a coordinator and run all QueueRunner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    # print images.get_shape()
    for i in range(10):
        img, lbl = sess.run([images, labels])
        print "Image_label: ", lbl
        plt.imshow(img[0, ...])
        plt.show()
    coord.request_stop()

    # Wait for threads to stop
    coord.join(threads)
    sess.close()
