import os

import tensorflow as tf
import numpy as np

import time
from datetime import datetime

from tensorflow.models.image.cifar10 import cifar10, cifar10_input

cifar10_input.IMAGE_SIZE = 32
cifar10_input.NUM_CLASSES = 4


types = ['spiral', 'lenticular', 'irregular', 'elliptical']

def read_example(value):
  record_defaults = [[""], [-1]]
  path, label = tf.decode_csv(
      value, record_defaults, " ")

  image_file = tf.read_file(path)
  image = tf.image.decode_jpeg(image_file, channels=3)
  image.set_shape([64, 64, 3])
  
  whitened_image = tf.image.per_image_whitening(image)

  return whitened_image, label

with tf.Graph().as_default():
  #sess = tf.InteractiveSession()
  # Start populating the filename queue.

  filenames = []
  for t in types:
    filenames.append(os.path.join('./images', t, 'scaled', 'example_map'))

  filename_queue = tf.train.string_input_producer(filenames)

  print filenames

  reader = tf.TextLineReader()
  key, value = reader.read(filename_queue)

  batch_size = 128
  min_fraction_of_examples_in_queue = 0.4
  num_examples_per_epoch = 50000
  min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)
  images_batch, label_batch =\
    tf.train.shuffle_batch_join([read_example(value) for _ in range(100)],
                                 batch_size=batch_size,
                                 capacity=min_queue_examples + 3*batch_size,
                                 min_after_dequeue=min_queue_examples)

  logits = cifar10.inference(images_batch)
  loss = cifar10.loss(logits, label_batch)

  global_step = tf.Variable(0, trainable=False)
  train_op = cifar10.train(loss, global_step)

  saver = tf.train.Saver(tf.all_variables())

  summary_op = tf.merge_all_summaries() 

  init = tf.initialize_all_variables()

  sess = tf.Session()
  sess.run(init)

  threads = tf.train.start_queue_runners(sess=sess)

  print 'starting training'

  for step in xrange(100000):
    print 'step: ', step
    start_time = time.time()
    _, loss_value = sess.run([train_op, loss])

    duration = time.time() - start_time

    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

    if step % 10 == 0:
      num_examples_per_step = 128
      examples_per_sec = num_examples_per_step / duration
      sec_per_batch = float(duration)

      format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '      
                    'sec/batch)')
      print (format_str % (datetime.now(), step, loss_value,
                           examples_per_sec, sec_per_batch))
