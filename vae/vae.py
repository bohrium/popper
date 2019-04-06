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
NB_UPDATES = 1001
CONCISENESS = 100
LEARN_RATE = 0.01

DATA_DIM = 28*28 
LATENT_DIM = 20
NOISE_DIM = LATENT_DIM
LATENT_DISTR_DIM = 2*LATENT_DIM
DATA_DISTR_DIM = 2*DATA_DIM

DATA_SHAPE = [BATCH_SIZE, DATA_DIM] 
LATENT_NOISE_SHAPE = [BATCH_SIZE, NOISE_DIM] 
DATA_NOISE_SHAPE = [BATCH_SIZE, DATA_DIM] 

def lrelu(layer):
    return tf.maximum(0.2*layer, layer)

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

    def __call__(self, latent): 
        hidden = lrelu(tf.matmul(latent, self.weighta))
        out = tf.matmul(hidden, self.weightb)
        means = tf.sigmoid(out[: , :DATA_DIM])
        stdvs = tf.exp(out[: , DATA_DIM:]) + 0.005
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
            r = np.maximum(0, np.minimum(1, r))
            s = np.maximum(0, np.minimum(1, s))
            side_by_side = np.concatenate([
                np.reshape(b[0], (28, 28)), 
                np.reshape(r, (28, 28)),
                np.reshape(s, (28, 28))
                ], axis=1) 
            imageio.imwrite('r%04d.png'%i, (255*side_by_side).astype(np.uint8))
