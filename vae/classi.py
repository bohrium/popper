''' author: samtenka
    changed: 2017-10-08
    created: 2017-10-07
    credits: www.tensorflow.org/get_started/mnist/pros
    descr: convolutional classifier on MNIST, demonstrating saving 
           IDENTICAL to `convolutional.py` except in section 3 
    usage: Run `python convolutional.py`.
'''

import tensorflow as tf
import numpy as np
import glob
import imageio

################################################################################
#   0. SET HYPERPARAMETERS                                                     #
################################################################################

BATCH_SIZE = 64
NB_UPDATES = 5001
CONCISENESS = 10
LEARN_RATE = 0.001

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

################################################################################
#   1. DEFINE MODEL                                                            #
################################################################################

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
        means = tf.exp(tf.reduce_max(logth, axis=1))
        means = tf.reshape(means, [-1, DATA_DIM])

        stdvs = 0.04 + 0.1 * tf.exp(0.1 * tf.matmul(latent, self.weightc)) + 0.0*means

        return tf.concat([means, stdvs], axis=1) 

def latent_sampler(noise, latent_distr):
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
    def __init__(self, data_shape, latent_noise_shape, data_noise_shape, encoder, latent_sampler, kl_to_target, decoder, neg_log_likelihood):
        self.data = tf.placeholder(shape=data_shape, dtype=tf.float32) 
        self.latent_noise = tf.placeholder(shape=latent_noise_shape, dtype=tf.float32)   
        self.data_noise = tf.placeholder(shape=data_noise_shape, dtype=tf.float32)   

        latent_distribution = encoder(self.data)  
        sample = latent_sampler(self.latent_noise, latent_distribution) 
        self.recon_distribution = decoder(sample)

        self.reconstruction_loss = neg_log_likelihood(self.data, self.recon_distribution) 
        self.regularization_loss = kl_to_target(latent_distribution)
        self.losses = self.reconstruction_loss + self.regularization_loss
        self.loss = tf.reduce_mean(self.losses)

        self.update = tf.train.AdamOptimizer(LEARN_RATE).minimize(self.loss)

class Classifier(object):
    def __init__(self):
        self.V = {}

        self.means_ = np.zeros((10,), dtype=np.float32) 
        self.stdvs_ = np.zeros((10,), dtype=np.float32) 
        self.means = tf.placeholder(shape=[10], dtype=tf.float32) 
        self.stdvs = tf.placeholder(shape=[10], dtype=tf.float32) 

        for c in range(10): 
            E = Encoder(name='enc%d'%c)
            D = Decoder(name='dec%d'%c)
            self.V[c] = VAE(DATA_SHAPE, LATENT_NOISE_SHAPE, DATA_NOISE_SHAPE, E, latent_sampler, kl_to_target, D, neg_log_likelihood)

        self.labels = tf.placeholder(shape=[int(BATCH_SIZE/8)], dtype=tf.int64) 
        #z_scores = (tf.convert_to_tensor([self.V[c].losses for c in range(10)]) - tf.expand_dims(self.means, axis=1))/tf.expand_dims(self.stdvs, axis=1)
        z_scores = tf.convert_to_tensor([self.V[c].losses for c in range(10)])
        z_scores = tf.reduce_mean(tf.reshape(z_scores, [10, 8, 8]), axis=2)
        self.predictions = tf.argmin(z_scores, axis=0)
        self.marks = tf.cast(tf.equal(self.predictions, self.labels), tf.float32)
        self.accuracy = tf.reduce_mean(self.marks)

################################################################################
#   2. GET DATA                                                                #
################################################################################

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

DIGITS = range(10)

imgs, lbls = mnist.train.images, mnist.train.labels
imgs = np.reshape(imgs, (-1, DATA_DIM))
imgs_by_lbl = {d:imgs[lbls==d] for d in DIGITS} 

timgs, tlbls = mnist.test.images, mnist.test.labels
timgs = np.reshape(timgs, (-1, DATA_DIM))
timgs_by_lbl = {d:timgs[tlbls==d] for d in DIGITS}



################################################################################
#   3. RUN GRAPH                                                               #
################################################################################

    #--------------------------------------------------------------------------+
    #       3. RUN GRAPH                                                       |
    #--------------------------------------------------------------------------+

C = Classifier()


# 3.0. CREATE SAVER
saver = tf.train.Saver()
SAVE_PATH = 'checkpoints/conv.ckpt'

def get_batch(dataset):
    indices = np.random.choice(len(dataset), size=BATCH_SIZE)
    return dataset[indices]
def get_supervised_tbatch():
    indices = np.random.choice(len(timgs), size=BATCH_SIZE)
    return timgs[indices], tlbls[indices]
def get_supervised_batch():
    indices = np.random.choice(len(imgs), size=BATCH_SIZE)
    return imgs[indices], lbls[indices]

def get_latent_noise():
    return np.random.randn(*LATENT_NOISE_SHAPE)
def get_data_noise():
    mags = np.random.exponential(*DATA_NOISE_SHAPE)
    signs = 2*np.random.randint(0,2, DATA_NOISE_SHAPE) - 1
    return signs*mags

with tf.Session() as sess:
    # 3.1. Load or initialize as appropriate 
    if glob.glob(SAVE_PATH+'*'):
        print('Loading Model...')
        saver.restore(sess, SAVE_PATH)
    else:
        print('Initializing Model from scratch...')
        sess.run(tf.global_variables_initializer())

    for i in range(100):
        bi, bl = get_supervised_tbatch()
        bi = np.array([bi[int((i-i%8)/8)] for i in range(64)]) 
        bl = bl[:8]
        n = get_latent_noise()

        fd = {C.labels:bl, C.means:C.means_, C.stdvs:C.stdvs_}
        for c in DIGITS:
            fd[C.V[c].data] = bi
            fd[C.V[c].latent_noise] = n

        marks, predictions = sess.run([C.marks, C.predictions], feed_dict=fd)
        print(i, sum(marks))

        for j, (m, p) in enumerate(zip(marks, predictions)):
            if m: continue

            b = np.array([bi[8*j]]*BATCH_SIZE) 
            
            rs = [] 
            for cc in DIGITS:
                n = get_latent_noise()
                V = C.V[cc]
                r, = sess.run([
                    V.recon_distribution[:,:DATA_DIM],
                ], feed_dict={V.latent_noise:n, V.data:b})
                rs.append(r[0])
            rs = np.array(rs)

            b = np.maximum(0, np.minimum(1, b))
            rs= np.maximum(0, np.minimum(1, rs))

            side_by_side = np.concatenate([
                np.concatenate([
                    np.reshape(np.stack([ b[0]]*3, axis=1), (H, W, 3)), 
                    np.reshape(np.stack([rs[i]]*3, axis=1), (H, W, 3)),
                ], axis=1)
            for i in range(10)], axis=0)
            imageio.imwrite('true-%d-pred-%d-%02d%d.png'%(bl[j], p, i, j), (255*side_by_side).astype(np.uint8))


    #trainacc, testacc = 0.0, 0.0

    #for i in range(NB_UPDATES):
    #    for c in DIGITS:
    #        b = get_batch(imgs_by_lbl[c]) #+ get_data_noise() / 256.0
    #        n = get_latent_noise()
    #        V = C.V[c]
    #        insample_losses, _ = sess.run([V.losses, V.update], feed_dict={
    #            V.data:b,
    #            V.latent_noise:n,
    #        })
    #        C.means_[c] = 0.1*np.mean(insample_losses) + 0.9*C.means_[c]
    #        C.stdvs_[c] = 0.1*np.std(insample_losses)  + 0.9*C.stdvs_[c]


    #        if i%(10*CONCISENESS): continue

    #        rs = [] 
    #        for cc in DIGITS:
    #            n = get_latent_noise()
    #            V = C.V[cc]
    #            r, = sess.run([
    #                V.recon_distribution[:,:DATA_DIM],
    #            ], feed_dict={V.latent_noise:n, V.data:b})
    #            rs.append(r[0])
    #        rs = np.array(rs)

    #        b = np.maximum(0, np.minimum(1, b))
    #        rs= np.maximum(0, np.minimum(1, rs))

    #        side_by_side = np.concatenate([
    #            np.concatenate([
    #                np.reshape(np.stack([ b[0]]*3, axis=1), (H, W, 3)), 
    #                np.reshape(np.stack([rs[i]]*3, axis=1), (H, W, 3)),
    #            ], axis=1)
    #        for i in range(10)], axis=0)
    #        imageio.imwrite('out/r-%d-%04d.png'%(c,i), (255*side_by_side).astype(np.uint8))

    #    if i%CONCISENESS: continue

    #    bi, bl = get_supervised_batch()
    #    bi = np.array([bi[int((i-i%8)/8)] for i in range(64)]) 
    #    bl = bl[:8]
    #    n = get_latent_noise()

    #    fd = {C.labels:bl, C.means:C.means_, C.stdvs:C.stdvs_}
    #    for c in DIGITS:
    #        fd[C.V[c].data] = bi
    #        fd[C.V[c].latent_noise] = n

    #    train_acc_, = sess.run([C.accuracy], feed_dict=fd)

    #    bi, bl = get_supervised_tbatch()
    #    bi = np.array([bi[int((i-i%8)/8)] for i in range(64)]) 
    #    bl = bl[:8]
    #    n = get_latent_noise()

    #    fd = {C.labels:bl, C.means:C.means_, C.stdvs:C.stdvs_}
    #    for c in DIGITS:
    #        fd[C.V[c].data] = bi
    #        fd[C.V[c].latent_noise] = n

    #    test_acc_, = sess.run([C.accuracy], feed_dict=fd)

    #    trainacc = 0.1*train_acc_ + 0.9*trainacc
    #    testacc  = 0.1*test_acc_  + 0.9*testacc

    #    print('%d trainacc %.2f testacc %.2f; trainvae %s' % (i, trainacc, testacc, ','.join('%5d'%m for m in C.means_) ))

    ## 3.4. Save
    #print('Saving Model...')
    #saver.save(sess, SAVE_PATH)
