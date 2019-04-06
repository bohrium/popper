''' author: samtenka
    change: 2019-04-02
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

BATCH_SIZE = 64
NB_UPDATES = 1001
CONCISENESS = 50
LEARN_RATE = 0.001

DATA_DIM = 28*28 
LATENT_DIM = 10
NOISE_DIM = LATENT_DIM
LATENT_DISTR_DIM = 2*LATENT_DIM
DATA_DISTR_DIM = 2*DATA_DIM

DATA_SHAPE = [BATCH_SIZE, DATA_DIM] 
NOISE_SHAPE = [BATCH_SIZE, NOISE_DIM] 

class Encoder(object):
    def __init__(self, name='enc'):
        self.weight = tf.get_variable('%s-weight'%name, shape=[DATA_DIM, LATENT_DISTR_DIM], dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())
    def __call__(self, data): 
        hidden = tf.matmul(data, self.weight)
        means =        hidden[: , :LATENT_DIM]
        stdvs = tf.exp(hidden[: , LATENT_DIM:]) + 0.005
        return tf.concat([means, stdvs], axis=1) 

class Decoder(object):
    def __init__(self, name='dec'):
        self.weight = tf.get_variable('%s-weight'%name, shape=[LATENT_DIM, DATA_DISTR_DIM], dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())
    def __call__(self, latent): 
        hidden = tf.matmul(latent, self.weight)
        means =        hidden[: , :DATA_DIM]
        stdvs = tf.exp(hidden[: , DATA_DIM:]) + 0.005
        return tf.concat([means, stdvs], axis=1) 

def sampler(noise, latent_distr):
    return latent_distr[: , :LATENT_DIM] + tf.multiply(noise, latent_distr[: , LATENT_DIM:])  

def kl_to_target(latent_distr):
    ''' In k dimensions, we have:
            KL(N(m,vI)||N(0, I)) = (trace(vI) + m^2 - k + log(det(vI))) / 2
    '''
    means = latent_distr[: , :LATENT_DIM] 
    stdvs = latent_distr[: , LATENT_DIM:] 
    return (
             tf.reduce_sum(tf.square(stdvs), axis=1)    # penalty for being too diffuse
            +tf.reduce_sum(tf.square(means), axis=1)    # penalty for off center
            -tf.reduce_sum(tf.log(stdvs)/2., axis=1)    # penalty for being too concentrated 
            -float(LATENT_DIM)
    ) / 2.0

def log_likelihood(data, data_distr):
    ''' variable-scale L1 norm
    '''
    means = data_distr[: , :DATA_DIM] 
    stdvs = data_distr[: , DATA_DIM:] 
    return tf.reduce_mean(
         tf.abs(data - means)/stdvs # penalty for guessing wrong, weighted by confidence
         -tf.log(2*stdvs)           # penalty for hedging 
    , axis=1)
    
class VAE(object):
    def __init__(self, data_shape, noise_shape, encoder, sampler, kl_to_target, decoder, log_likelihood):
        self.data = tf.placeholder(shape=data_shape, dtype=tf.float32) 
        self.noise = tf.placeholder(shape=noise_shape, dtype=tf.float32)   

        latent_distribution = encoder(self.data)  
        sample = sampler(self.noise, latent_distribution) 

        self.reconstruction_loss = log_likelihood(self.data, decoder(sample)) 
        self.regularization_loss = kl_to_target(latent_distribution)
        self.loss = tf.reduce_mean(self.reconstruction_loss + self.regularization_loss)

        self.update = tf.train.AdamOptimizer(LEARN_RATE).minimize(self.loss)

E = Encoder()
D = Decoder()
V = VAE(DATA_SHAPE, NOISE_SHAPE, E, sampler, kl_to_target, D, log_likelihood)

def get_batch():
    return np.random.randn(BATCH_SIZE, DATA_DIM) * np.expand_dims(np.arange(float(DATA_DIM)), axis=0) / float(DATA_DIM)
def get_noise():
    return np.random.randn(BATCH_SIZE, NOISE_DIM) 

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) 

    for i in range(NB_UPDATES):
        b = get_batch()
        n = get_noise()
        l, _ = sess.run([
            V.loss, V.update], feed_dict={
            V.data:b,
            V.noise:n,
        })
        if i%CONCISENESS==0:
            print('step %4d \t loss %6.2f' % (i, l))
