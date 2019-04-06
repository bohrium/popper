#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
#imgs, lbls = mnist.train.images, mnist.train.labels
#imgs = np.reshape(imgs, (-1, 28, 28))

import tensorflow as tf
import numpy as np
import imageio 

H = 28 
W = 28

yc = tf.constant(np.expand_dims(np.expand_dims(np.arange(float(H)), axis=1), axis=0), dtype=tf.float32)
xc = tf.constant(np.expand_dims(np.expand_dims(np.arange(float(W)), axis=0), axis=0), dtype=tf.float32)

#image = tf.placeholder(shape=[28, 28], dtype=tf.float32)
#
#wA = tf.Variable(shape=[28*28, 30], dtype=tf.float32) 


line_seg = tf.placeholder(shape=[6, 5], dtype=tf.float32)

ys = tf.expand_dims(tf.expand_dims(line_seg[: , 0], axis=1), axis=2)
xs = tf.expand_dims(tf.expand_dims(line_seg[: , 1], axis=1), axis=2)
ye = tf.expand_dims(tf.expand_dims(line_seg[: , 2], axis=1), axis=2)
xe = tf.expand_dims(tf.expand_dims(line_seg[: , 3], axis=1), axis=2)
t  = tf.expand_dims(tf.expand_dims(line_seg[: , 4], axis=1), axis=2)

canvas = tf.reduce_max((
        tf.sqrt(tf.square(ys-ye) + tf.square(xs-xe))
       -tf.sqrt(tf.square(ys-yc) + tf.square(xs-xc))
       -tf.sqrt(tf.square(yc-ye) + tf.square(xc-xe))
)/t, axis=0)
canvas = tf.exp(canvas)

def random():
    return [
        (np.random.randint(H), np.random.randint(W),
         np.random.randint(H), np.random.randint(W), 0.1 + 0.4*np.random.random())
        for i in range(6)
    ]

with tf.Session() as sess:
    r = random()
    c = sess.run(canvas, feed_dict = {line_seg:r})

imageio.imwrite('r.png', (255*c).astype(np.uint8))
