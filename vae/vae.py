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
NB_UPDATES = 5001
CONCISENESS = 250
LEARN_RATE = 0.001

DIGIT = 8

H = 28 
W = 28
NB_SEGS = 6
DATA_DIM = H*W
LATENT_DIM = 16
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
    def __init__(self, name='enc', HIDDEN_DIM=32):
        self.weighta = tf.get_variable('%s-weighta'%name, shape=[DATA_DIM, HIDDEN_DIM], dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())
        self.biasa = tf.get_variable('%s-biasa'%name, shape=[1, HIDDEN_DIM], dtype=tf.float32,
                initializer=tf.random_normal_initializer(stddev=0.1))
        self.weightb = tf.get_variable('%s-weightb'%name, shape=[HIDDEN_DIM, LATENT_DISTR_DIM], dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())

    def __call__(self, data): 
        hidden = lrelu(tf.matmul(data, self.weighta) + self.biasa)
        out = tf.matmul(hidden, self.weightb)
        means =        out[: , :LATENT_DIM]
        stdvs = tf.exp(out[: , LATENT_DIM:]) + 0.005
        return tf.concat([means, stdvs], axis=1) 

class Decoder(object):
    def __init__(self, name='dec', HIDDEN_DIM=32):
        self.weighta = tf.get_variable('%s-weighta'%name, shape=[LATENT_DIM, HIDDEN_DIM], dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())
        self.biasa = tf.get_variable('%s-biasa'%name, shape=[1, HIDDEN_DIM], dtype=tf.float32,
                initializer=tf.random_normal_initializer(stddev=0.5))
        self.weightb = tf.get_variable('%s-weightb'%name, shape=[HIDDEN_DIM, DATA_DISTR_DIM], dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())

        self.weightc = tf.get_variable('%s-weightc'%name, shape=[LATENT_DIM, 1], dtype=tf.float32,
                initializer=tf.random_normal_initializer(stddev=0.01))
        self.biasc = tf.get_variable('%s-biasc'%name, shape=[1, 1], dtype=tf.float32,
                initializer=tf.random_normal_initializer(stddev=0.01))

    def prior_loss(self):
        return 0.0 #100.*(tf.reduce_mean(tf.abs(self.weighta)) + tf.reduce_mean(tf.abs(self.weightb)))

    def __call__(self, latent): 
        hidden = lrelu(tf.matmul(latent, self.weighta) + self.biasa)
        out = tf.matmul(hidden, self.weightb)

        line_segs = tf.reshape(out[: , :NB_SEGS*5], [-1, NB_SEGS, 5])
        ys = clip(0.5 + 0.5*tf.expand_dims(tf.expand_dims(line_segs[: , : , 0], axis=2), axis=2))
        xs = clip(0.5 + 0.5*tf.expand_dims(tf.expand_dims(line_segs[: , : , 1], axis=2), axis=2))
        ye = clip(0.5 + 0.5*tf.expand_dims(tf.expand_dims(line_segs[: , : , 2], axis=2), axis=2))
        xe = clip(0.5 + 0.5*tf.expand_dims(tf.expand_dims(line_segs[: , : , 3], axis=2), axis=2))
        t  = 0.005 + 0.1 * tf.exp(tf.expand_dims(tf.expand_dims(line_segs[: , : , 4], axis=2), axis=2))


        yc = (1.0/H) * tf.constant(np.expand_dims(np.expand_dims(np.expand_dims(np.arange(float(H)), axis=1), axis=0), axis=0), dtype=tf.float32)
        xc = (1.0/W) * tf.constant(np.expand_dims(np.expand_dims(np.expand_dims(np.arange(float(W)), axis=0), axis=0), axis=0), dtype=tf.float32)

        logth = (
                tf.sqrt(1e-4 + tf.square(ys-ye) + tf.square(xs-xe))
               -tf.sqrt(1e-4 + tf.square(ys-yc) + tf.square(xs-xc))
               -tf.sqrt(1e-4 + tf.square(yc-ye) + tf.square(xc-xe))
        )/t
        logth = -tf.square(logth)
        #logth = tf.maximum(-tf.square(logth), 10.0*logth)
        means = tf.exp(tf.reduce_max(logth, axis=1))
        means = tf.reshape(means, [-1, DATA_DIM])

        stdvs = 0.04 + 0.1 * tf.exp(0.1 * tf.matmul(latent, self.weightc)) + 0.0*means

        return tf.concat([means, stdvs], axis=1) 

    def colored(self, latent): 
        hidden = lrelu(tf.matmul(latent, self.weighta) + self.biasa)
        out = tf.matmul(hidden, self.weightb)

        line_segs = tf.reshape(out[: , :NB_SEGS*5], [-1, NB_SEGS, 5])
        ys = clip(0.5 + 0.5*tf.expand_dims(tf.expand_dims(line_segs[: , : , 0], axis=2), axis=2))
        xs = clip(0.5 + 0.5*tf.expand_dims(tf.expand_dims(line_segs[: , : , 1], axis=2), axis=2))
        ye = clip(0.5 + 0.5*tf.expand_dims(tf.expand_dims(line_segs[: , : , 2], axis=2), axis=2))
        xe = clip(0.5 + 0.5*tf.expand_dims(tf.expand_dims(line_segs[: , : , 3], axis=2), axis=2))
        t  = 0.005 + 0.1 * tf.exp(tf.expand_dims(tf.expand_dims(line_segs[: , : , 4], axis=2), axis=2))

        yc = (1.0/H) * tf.constant(np.expand_dims(np.expand_dims(np.expand_dims(np.arange(float(H)), axis=1), axis=0), axis=0), dtype=tf.float32)
        xc = (1.0/W) * tf.constant(np.expand_dims(np.expand_dims(np.expand_dims(np.arange(float(W)), axis=0), axis=0), axis=0), dtype=tf.float32)

        logth= (
             tf.sqrt(1e-4 + tf.square(ys-ye) + tf.square(xs-xe))
            -tf.sqrt(1e-4 + tf.square(ys-yc) + tf.square(xs-xc))
            -tf.sqrt(1e-4 + tf.square(yc-ye) + tf.square(xc-xe))
        )/t
        logth = -tf.square(logth)
        #logth = tf.maximum(-tf.square(logth), 10.0*logth)

        colors = tf.constant([
            [1.0, 0.6, 0.1],
            [1.0, 0.1, 0.6],
            [0.8, 0.9, 0.0],
            [0.0, 0.9, 0.8],
            [0.6, 0.1, 1.0],
            [0.1, 0.6, 1.0],
            #[0.00, 0.00, 1.00],
            #[0.00, 1.00, 0.00],
            #[1.00, 0.00, 0.00],
            #[0.25, 0.25, 0.75],
            #[0.25, 0.75, 0.25],
            #[0.75, 0.25, 0.25],
            #[0.75, 0.75, 0.00],
            #[0.75, 0.00, 0.75],
            #[0.00, 0.75, 0.75],
            #[0.5 , 0.5 , 0.5 ],
        ])

        indices = tf.argmax(logth, axis=1) 
        there =   tf.expand_dims(tf.exp(tf.reduce_max(logth, axis=1)), axis=3)
        colored = tf.gather_nd(colors, tf.expand_dims(indices, 3)) * there

        return tf.reshape(colored, [-1, DATA_DIM, 3])


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
    ''' variable-scale L2 norm
    '''
    means = data_distr[: , :DATA_DIM] 
    stdvs = data_distr[: , DATA_DIM:] 
    return tf.reduce_sum(
          tf.square((data - means)/stdvs)/2 # penalty for guessing wrong, weighted by confidence
         +tf.log(stdvs)                     # penalty for hedging 
         +tf.log(2*3.14159)/2
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
        self.colored = decoder.colored(sample)

        self.reconstruction_loss = neg_log_likelihood(self.data, self.recon_distribution) 
        self.regularization_loss = kl_to_target(latent_distribution)
        self.losses = self.reconstruction_loss + self.regularization_loss
        self.prior_loss = decoder.prior_loss()
        self.loss = tf.reduce_mean(self.losses) + self.prior_loss

        self.update = tf.train.AdamOptimizer(LEARN_RATE).minimize(self.loss)

E = Encoder()
D = Decoder()
V = VAE(DATA_SHAPE, LATENT_NOISE_SHAPE, DATA_NOISE_SHAPE, E, latent_sampler, data_sampler, kl_to_target, D, neg_log_likelihood)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
imgs, lbls = mnist.train.images, mnist.train.labels
imgs = np.reshape(imgs, (-1, DATA_DIM))
eights = (lbls==DIGIT)
imgs = imgs[eights]
lbls = lbls[eights]

timgs, tlbls = mnist.test.images, mnist.test.labels
timgs = np.reshape(timgs, (-1, DATA_DIM))
teights = (tlbls==DIGIT)
tothers = (tlbls!=DIGIT) 
timgs_ = timgs[tothers] 
timgs = timgs[teights]
tlbls = tlbls[teights]


def get_batch(dataset=imgs):
    indices = np.random.choice(len(dataset), size=BATCH_SIZE)
    return dataset[indices]
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
        b += np.random.exponential(scale=1.0/256, size=b.shape) * (1.0 - 2.0 * np.random.binomial(1, 0.5, size=b.shape))
        n = get_latent_noise()
        il, _ = sess.run([
            V.loss, V.update], feed_dict={
            V.data:b,
            V.latent_noise:n,
        })
        if i%CONCISENESS==0:
            R = 5

            bo = get_batch(timgs)
            bo += np.random.exponential(scale=1.0/256, size=b.shape) * (1.0 - 2.0 * np.random.binomial(1, 0.5, size=b.shape))
            ols, ogs = np.zeros(BATCH_SIZE), np.zeros(BATCH_SIZE)
            for r in range(R):
                n = get_latent_noise()
                ol, og = sess.run([
                    V.losses, tf.reduce_mean(V.regularization_loss)], feed_dict={
                    V.data:bo,
                    V.latent_noise:n,
                })
                ols += ol
                ogs += og
            ols /= R
            ogs /= R

            ba = get_batch(timgs_)
            ba += np.random.exponential(scale=1.0/256, size=b.shape) * (1.0 - 2.0 * np.random.binomial(1, 0.5, size=b.shape))
            als, ags = np.zeros(BATCH_SIZE), np.zeros(BATCH_SIZE)
            for r in range(R):
                n = get_latent_noise()
                al, ag = sess.run([
                    V.losses, tf.reduce_mean(V.regularization_loss)], feed_dict={
                    V.data:ba,
                    V.latent_noise:n,
                })
                als += al
                ags += ag
            als /= R
            ags /= R

            print('step %4d \t train loss %8.2f \t test loss %8.2f \t other loss %8.2f' % (i, np.median(il), np.median(ols), min(als)))

            bad = ba[np.argmin(als)]

            b = np.concatenate([bo[:2], [bad], bo[3:]], axis=0)
            b = [b[int(i/2)] for i in range(len(b))]
            n = get_latent_noise()
            d = get_data_noise()
            c, r, s, l = sess.run([
                V.colored[:,:,:],
                V.recon_distribution[:,:DATA_DIM],
                V.recon_distribution[:,DATA_DIM:],
                V.losses
            ], feed_dict={V.latent_noise:n, V.data:b, V.data_noise:d})

            b = np.maximum(0, np.minimum(1, b))
            d = np.maximum(0, np.minimum(1, 0.2*np.abs(r-b)/s))
            r = np.maximum(0, np.minimum(1, r))
            s = np.maximum(0, np.minimum(1, s))
            c = np.maximum(0, np.minimum(1, c))

            L = np.log(256)*H*W
            l = np.stack([0.2 + ll/L + np.zeros((H, W), np.float32) for ll in l], axis=0) 
            l = np.maximum(0, np.minimum(1, l))

            side_by_side = np.concatenate([
                np.concatenate([
                    np.reshape(np.stack([b[i]]*3, axis=1), (H, W, 3)), 
                    np.reshape(np.stack([r[i]]*3, axis=1), (H, W, 3)),
                    np.reshape(np.stack([s[i]]*3, axis=1), (H, W, 3)),
                    np.reshape(np.stack([d[i]]*3, axis=1), (H, W, 3)),
                    np.reshape(          c[i],             (H, W, 3)),
                    np.reshape(np.stack([l[i]]*3, axis=1), (H, W, 3)),
                ], axis=1)
            for i in range(6)], axis=0)
            imageio.imwrite('r%04d.png'%i, (255*side_by_side).astype(np.uint8))
