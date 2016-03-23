import Image

import os

import tensorflow as tf
import numpy as np

import time
from datetime import datetime

from tensorflow.models.image.cifar10 import cifar10, cifar10_input

import sys

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
    tf.train.shuffle_batch_join([read_example(value) for _ in range(9)],
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


  summary_writer = tf.train.SummaryWriter('./train',
                                          graph_def=sess.graph_def)


  if len(sys.argv) == 1:
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

      if step % 100 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      if step % 1000 == 0 or (step + 1) == 100000:
        checkpoint_path = os.path.join('./train', 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)

  else: # if len(sys.argv) == 1
    print 'skipping training'
    ckpt = tf.train.get_checkpoint_state('./train')
    if ckpt and ckpt.model_checkpoint_path:
      print 'restoring model'
      print ckpt.model_checkpoint_path
      saver.restore(sess, ckpt.model_checkpoint_path)
      print 'model restored'
    else:
      print "Error: no checkpoint"
      sys.exit()

    print 'running logits'   
    threads = tf.train.start_queue_runners(sess=sess)
    logits_evaled, images = sess.run([logits, images_batch])

    for x in logits_evaled:
      val = np.asarray(x)[0:4] 
      softmax = np.exp(val) / np.sum(np.exp(val), axis=0)
      print types[np.argmax(softmax)], softmax

    for n, i in enumerate(images):
      arr = (np.asarray(i)*10 + 100).astype(int)
      print i
      img = Image.fromarray(arr, "RGB")
      img.save(os.path.join('./results', '%s.jpg' % n))
