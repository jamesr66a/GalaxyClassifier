import tensorflow as tf

sess = tf.Session()
with sess.as_default():

  filename = ['./images/spiral/scaled/potw1503a.jpg']
  filename_queue = tf.train.string_input_producer(filename)

  reader = tf.WholeFileReader()
  key, value = reader.read(filename_queue)

  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  value.eval()

  coord.request_stop()
  coord.join(threads)
