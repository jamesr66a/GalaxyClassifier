import os

import tensorflow as tf

types = ['spiral', 'lenticular', 'irregular', 'elliptical']

def read_example(value):
  record_defaults = [[""], [-1]]
  path, label = tf.decode_csv(
      value, record_defaults, " ")

  image_file = tf.read_file(path)
  image = tf.image.decode_jpeg(image_file, channels=3)

  return image, label


with tf.Session() as sess:
  # Start populating the filename queue.
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  filenames = []
  for t in types:
    filenames.append(os.path.join('./images', t, 'scaled', 'example_map'))

  filename_queue = tf.train.string_input_producer(filenames)

  print filenames

  reader = tf.WholeFileReader()
  key, value = reader.read(filename_queue)

  x = sess.run([value])
  print x

  #i, l = read_example(value)
  #print sess.run([i, l])

  coord.request_stop()
  coord.join(threads)
