import os

import tensorflow as tf

types = ['spiral', 'lenticular', 'irregular', 'elliptical']


def read_example_map(example_map, type):
  record_defaults=[tf.constant([], dtype=tf.string), tf.constant([], dtype=tf.int32)]
  fn, label = tf.decode_csv(example_map, record_defaults, " ")
  file_contents = tf.read_file(fn)
  example = tf.image.decode_jpeg(file_contents, channels=3)
  return example, label

def read_examples():
  map_filenames = []
  type_names = []

  for t in types:
    map_filenames.append(os.path.join('./images', t, 'scaled', 'example_map'))
    type_names.append(t)

  filename_queue = tf.train.string_input_producer(map_filenames)
  type_queue = tf.train.string_input_producer(type_names)

  reader = tf.TextLineReader()
  key, value = reader.read(filename_queue)

  images_tensor, label_tensor = read_example_map(value, type_queue.dequeue())

  #for t in next(iter(types)):
    #example, label = read_example_map(input_queue.dequeue())
    #images_tensor = tf.concat(1, images_tensor, example)
    #label_tensor = tf.concat(1, label_tensor, label)

  return images_tensor, label_tensor

sess = tf.Session()
with sess.as_default():
  images, labels = read_examples()
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  print images.eval(), labels.eval()

  coord.request_stop()
  coord.join(threads)