''' author: samtenka
    change: 2019-04-06
    create: 2019-04-02
    descrp: implement a variational autoencoder based on these provided building blocks:
                encoder: data space --> distributions over latent space 
                sampler: noise x distributions over latent space --> latent space 
                kl to target: distributions over latent space --> reals 
                decoder: latent space --> distributions over data space
                log likelihood: data space x distributions over data space --> reals  
'''

import tensorflow as tf
import numpy as np
import imageio

BATCH_SIZE = 64
NB_UPDATES = 2001
CONCISENESS = 50
LEARN_RATE = 0.001

H = 28 
W = 28
NB_SEGS = 6
DATA_DIM = H*W
LATENT_DIM = 20
NOISE_DIM = LATENT_DIM
LATENT_DISTR_DIM = 2*LATENT_DIM
DATA_DISTR_DIM = NB_SEGS*5

DATA_SHAPE = [BATCH_SIZE, DATA_DIM] 
LATENT_NOISE_SHAPE = [BATCH_SIZE, NOISE_DIM] 
DATA_NOISE_SHAPE = [BATCH_SIZE, DATA_DIM] 

def lrelu(layer):
    return tf.maximum(0.2*layer, layer)
def clip(layer):
    return tf.maximum(0.0, tf.minimum(1.0, layer))

class Encoder(object):
    def __init__(self, name='enc', HIDDEN_DIM=100):
        self.weighta = tf.get_variable('%s-weighta'%name, shape=[DATA_DIM, HIDDEN_DIM], dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())
        self.weightb = tf.get_variable('%s-weightb'%name, shape=[HIDDEN_DIM, LATENT_DISTR_DIM], dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())

    def __call__(self, data): 
        hidden = lrelu(tf.matmul(data, self.weighta))
        out = tf.matmul(hidden, self.weightb)
        means =        out[: , :LATENT_DIM]
        stdvs = tf.exp(out[: , LATENT_DIM:]) + 0.005
        return tf.concat([means, stdvs], axis=1) 

class Decoder(object):
    def __init__(self, name='dec', HIDDEN_DIM=100):
        self.weighta = tf.get_variable('%s-weighta'%name, shape=[LATENT_DIM, HIDDEN_DIM], dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())
        self.weightb = tf.get_variable('%s-weightb'%name, shape=[HIDDEN_DIM, DATA_DISTR_DIM], dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())
        self.weightc = tf.get_variable('%s-weightc'%name, shape=[DATA_DIM], dtype=tf.float32,
                initializer=tf.random_normal_initializer(stddev=0.01))

    def __call__(self, latent): 
        hidden = lrelu(tf.matmul(latent, self.weighta))
        out = tf.matmul(hidden, self.weightb)

        s = 0.1 + 0.1 * tf.exp(10.0*self.weightc)
        stdvs = tf.stack([s]*BATCH_SIZE, axis=0)

        line_segs = tf.reshape(out[: , :NB_SEGS*5], [-1, NB_SEGS, 5])
        ys = clip(0.5 + 0.5*tf.expand_dims(tf.expand_dims(line_segs[: , : , 0], axis=2), axis=2))
        xs = clip(0.5 + 0.5*tf.expand_dims(tf.expand_dims(line_segs[: , : , 1], axis=2), axis=2))
        ye = clip(0.5 + 0.5*tf.expand_dims(tf.expand_dims(line_segs[: , : , 2], axis=2), axis=2))
        xe = clip(0.5 + 0.5*tf.expand_dims(tf.expand_dims(line_segs[: , : , 3], axis=2), axis=2))
        t  = 0.02 + 0.2 * clip(tf.expand_dims(tf.expand_dims(line_segs[: , : , 4], axis=2), axis=2))

        yc = (1.0/H) * tf.constant(np.expand_dims(np.expand_dims(np.expand_dims(np.arange(float(H)), axis=1), axis=0), axis=0), dtype=tf.float32)
        xc = (1.0/W) * tf.constant(np.expand_dims(np.expand_dims(np.expand_dims(np.arange(float(W)), axis=0), axis=0), axis=0), dtype=tf.float32)

        means = tf.reduce_max((
                tf.sqrt(1e-4 + tf.square(ys-ye) + tf.square(xs-xe))
               -tf.sqrt(1e-4 + tf.square(ys-yc) + tf.square(xs-xc))
               -tf.sqrt(1e-4 + tf.square(yc-ye) + tf.square(xc-xe))
        )/t, axis=1)
        means = tf.reshape(tf.exp(means), [-1, DATA_DIM])

        return tf.concat([means, stdvs], axis=1) 


def latent_sampler(noise, latent_distr):
    return latent_distr[: , :LATENT_DIM] + tf.multiply(noise, latent_distr[: , LATENT_DIM:])  

def data_sampler(noise, data_distr):
    return data_distr[: , :DATA_DIM] #+ tf.multiply(noise, data_distr[: , DATA_DIM:])  

def kl_to_target(latent_distr):
    ''' In k dimensions, we have:
            KL(N(m,vI)||N(0, I)) = (trace(vI) + m^2 - k + log(det(vI))) / 2
    '''
    means = latent_distr[: , :LATENT_DIM] 
    stdvs = latent_distr[: , LATENT_DIM:] 
    return (
             tf.reduce_sum(tf.square(stdvs), axis=1)    # penalty for being too diffuse
            +tf.reduce_sum(tf.square(means), axis=1)    # penalty for off center
            -tf.reduce_sum(tf.log(stdvs)*2., axis=1)    # penalty for being too concentrated 
            -float(LATENT_DIM)
    ) / 2.0

def neg_log_likelihood(data, data_distr):
    ''' variable-scale L1 norm
    '''
    means = data_distr[: , :DATA_DIM] 
    stdvs = data_distr[: , DATA_DIM:] 
    return tf.reduce_sum(
         tf.abs(data - means)/stdvs # penalty for guessing wrong, weighted by confidence
         +tf.log(2*stdvs)           # penalty for hedging 
    , axis=1)
    
class VAE(object):
    def __init__(self, data_shape, latent_noise_shape, data_noise_shape, encoder, latent_sampler, data_sampler, kl_to_target, decoder, neg_log_likelihood):
        self.data = tf.placeholder(shape=data_shape, dtype=tf.float32) 
        self.latent_noise = tf.placeholder(shape=latent_noise_shape, dtype=tf.float32)   
        self.data_noise = tf.placeholder(shape=data_noise_shape, dtype=tf.float32)   

        latent_distribution = encoder(self.data)  
        sample = latent_sampler(self.latent_noise, latent_distribution) 
        self.recon_distribution = decoder(sample)
        self.recon = data_sampler(self.data_noise, self.recon_distribution)

        self.reconstruction_loss = neg_log_likelihood(self.data, self.recon_distribution) 
        self.regularization_loss = kl_to_target(latent_distribution)
        self.loss = tf.reduce_mean(self.reconstruction_loss + self.regularization_loss)

        self.update = tf.train.AdamOptimizer(LEARN_RATE).minimize(self.loss)

E = Encoder()
D = Decoder()
V = VAE(DATA_SHAPE, LATENT_NOISE_SHAPE, DATA_NOISE_SHAPE, E, latent_sampler, data_sampler, kl_to_target, D, neg_log_likelihood)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
imgs, lbls = mnist.train.images, mnist.train.labels
imgs = np.reshape(imgs, (-1, DATA_DIM))
eights = (lbls==8)
imgs = imgs[eights]
lbls = lbls[eights]

timgs, tlbls = mnist.test.images, mnist.test.labels
timgs = np.reshape(timgs, (-1, DATA_DIM))
teights = (tlbls==8)
timgs = timgs[teights]
tlbls = tlbls[teights]


def get_batch(dataset='train'):
    i = imgs if dataset=='train' else timgs
    indices = np.random.choice(len(i), size=BATCH_SIZE)
    return i[indices]
def get_latent_noise():
    return np.random.randn(*LATENT_NOISE_SHAPE)
def get_data_noise():
    mags = np.random.exponential(*DATA_NOISE_SHAPE)
    signs = 2*np.random.randint(0,2, DATA_NOISE_SHAPE) - 1
    return signs*mags

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) 

    for i in range(NB_UPDATES):
        b = get_batch()
        n = get_latent_noise()
        l, _ = sess.run([
            V.loss, V.update], feed_dict={
            V.data:b,
            V.latent_noise:n,
        })
        if i%CONCISENESS==0:
            print('step %4d \t loss %6.2f' % (i, l))
            b = get_batch('test')
            n = get_latent_noise()
            d = get_data_noise()
            r, s = sess.run([V.recon_distribution[0,:DATA_DIM],
                             V.recon_distribution[0,DATA_DIM:]], feed_dict={V.latent_noise:n, V.data:b, V.data_noise:d})
            d = np.maximum(0, np.minimum(1, 0.1*np.abs(r-b[0])/s))
            r = np.maximum(0, np.minimum(1, r))
            s = np.maximum(0, np.minimum(1, s))
            side_by_side = np.concatenate([
                np.reshape(b[0], (H, W)), 
                np.reshape(r, (H, W)),
                np.reshape(s, (H, W)),
                np.reshape(d, (H, W))
                ], axis=1) 
            imageio.imwrite('r%04d.png'%i, (255*side_by_side).astype(np.uint8))
