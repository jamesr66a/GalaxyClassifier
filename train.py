from PIL import *
from PIL import Image

import re
import os

import tensorflow as tf

types = ['spiral', 'lenticular', 'irregular', 'elliptical']

jpeg_re = re.compile('.*\.jpg')

size = 300, 300

filenames = {"spiral": [], "lenticular": [], "irregular": [], "elliptical": []}

for t in types:
  for dirname, dirnames, filename in os.walk(os.path.join('./images', t)):
    for name in filter(jpeg_re.match, filename):
      filenames[t].append(os.path.join('./images', t, name))

images = {}

indices = {"spiral": 0, "lenticular": 1, "irregular": 2, "elliptical": 3}

for t in types:
  filename_queue = tf.train.string_input_producer(filenames[t])

  reader = tf.WholeFileReader()
  key, value = reader.read(filename_queue)

  images[t] = tf.image.decode_jpeg(value, channels=3)
  print images[t]

  # Normalize pixels
  images[t] = tf.image.per_image_whitening(images[t])