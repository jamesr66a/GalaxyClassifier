from PIL import *
from PIL import Image

import re
import os

import tensorflow as tf
import numpy as np

types = ['spiral', 'lenticular', 'irregular', 'elliptical']

jpeg_re = re.compile('.*\.jpg')

size = 300, 300

filenames = {"spiral": [], "lenticular": [], "irregular": [], "elliptical": []}

for t in types:
  for dirname, dirnames, filename in os.walk(os.path.join('./images', t, 'scaled')):
    for name in filter(jpeg_re.match, filename):
      filenames[t].append(os.path.join('./images', t, 'scaled', name))

images = {}
examples = {}

indices = {"spiral": 0, "lenticular": 1, "irregular": 2, "elliptical": 3}

sess = tf.Session()
with sess.as_default():
  for t in types:
    filename_queue = tf.train.string_input_producer(filenames[t])
    print 'meme'

    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    print 'lmao'

    images[t] = tf.image.decode_jpeg(value, channels=3)

    # Normalize pixels
    images[t] = tf.image.per_image_whitening(images[t])
    print 'dank'

    # Add labels
    #labels = tf.tile(tf.constant([indices[t]]), images[t].)

  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  print 'abc'

  for t in types:
    for i in range(len(filenames[t])):
      img = images[t].eval()
      print img.shape

  coord.request_stop()
  coord.join(threads)