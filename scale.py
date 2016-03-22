from PIL import *
from PIL import Image

import re
import os

types = ['spiral', 'lenticular', 'irregular', 'elliptical']

jpeg_re = re.compile('.*\.jpg')

size = 64, 64

for t in types:
  for dirname, dirnames, filenames in os.walk(os.path.join('./images', t)):
    for name in filter(jpeg_re.match, filenames):
      image = Image.open(os.path.join('./images', t, name))
      image.thumbnail(size, Image.ANTIALIAS)
      crop = image.crop((0, 0, size[0], size[1]))
      crop.save(os.path.join('./images', t, 'scaled', name))
