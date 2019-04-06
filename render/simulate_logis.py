''' author: samtenka
    change: 2019-03-04
    create: 2017-10-07
    descrp: Append summary of SGD and GD losses (on logistic regression on MNIST 0-vs-1 classification) to a file.
            Here, `SGD` means `batch size 1 without replacement`.  
            To run, type:
                python simulate_logis.py 1000 100 0.00 0.005 12 32 experdata_logis.txt 
            The                  1000   gives   a number of trials to perform per experimental condition;
            the                   100   gives   a training set size and number of gradient updates;
            the                  0.00   gives   a starting learning rate to sweep from;
            the                  0.05   gives   a ending learning rate to sweep to;
            the                    12   gives   (one less than) the number of learning rates to sweep through;
            the                    32   gives   a desired floating point precision (32 or 64);
            the   experdata_logis.txt   gives   a filename of a log to which to append.
'''

import tensorflow as tf
import numpy as np
from tqdm import tqdm
import sys 

assert(len(sys.argv)==8)
NB_TRIALS = int(sys.argv[1]) 
INS_SIZE  = int(sys.argv[2]) 
MIN_LR    = float(sys.argv[3]) 
MAX_LR    = float(sys.argv[4]) 
LR_SWEEP  = int(sys.argv[5]) 
PRECISION = {32:tf.float32, 64:tf.float64}[int(sys.argv[6])]
FILE_NM   = sys.argv[7]



################################################################################
#           0. DATASET BATCHES                                                 #
################################################################################

from tensorflow.examples.tutorials.mnist import input_data

class Dataset(object): 
    ''' MNIST is a classic image-classification dataset containing 28x28 grayscale photographs of handwritten digits (0
        through 9).  This class provides access to MNIST via in-sample and out-of-sample batches.  It allows us to
        sample from a training set potentially smaller than the the overall MNIST training set, and to sample with or
        without replacement.  Note that we DO NOT load labels as one-hot vectors.
        Thus, `get_batch` and `get_all` each return two arrays of shape (???, 28*28) and (???), respectively.
    '''
    def __init__(self):
        ''' Read MNIST, with labels one-hot. '''
        mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

        self.ins_images = []
        self.ins_labels = []
        for img, lbl in zip(mnist.train.images, mnist.train.labels):  
            if 2<=lbl: continue
            self.ins_images.append(img)
            self.ins_labels.append(lbl)

        self.out_images = []
        self.out_labels = []
        for img, lbl in zip(mnist.test.images, mnist.test.labels):  
            if 2<=lbl: continue
            self.out_images.append(img)
            self.out_labels.append(lbl)                      

        total_length = len(self.ins_images)+len(self.out_images)
        self.ins_images, self.out_images = ((self.ins_images + self.out_images)[:total_length//2],
                                            (self.ins_images + self.out_images)[total_length//2:])
        self.ins_labels, self.out_labels = ((self.ins_labels + self.out_labels)[:total_length//2],
                                            (self.ins_labels + self.out_labels)[total_length//2:])   
        print(total_length//2, total_length-total_length//2)

        self.ins_images = np.array(self.ins_images)
        self.ins_labels = np.array(self.ins_labels)
        self.out_images = np.array(self.out_images)
        self.out_labels = np.array(self.out_labels)

        self.resample(len(self.ins_images))

    def resample(self, sample_size):
        ''' MNIST is a classic image-classification dataset.  Its images are 28x28 grayscale photographs of handwritten
            digits (0 through 9).  Note that we load labels as one-hot vectors, making it easier to define the loss.
        '''
        indices = np.random.choice(np.arange(len(self.ins_images)), size=sample_size, replace=False)
        self.sample_images = self.ins_images[indices]
        self.sample_labels = self.ins_labels [indices]
        self.sample_size = sample_size
        self.index = 0

    def get_batch(self, batch_size, split='ins', opt='sgd', with_replacement=False):
        ''' Returns batch, by default from training set.  In case the specified optimizer is 'sgd', then `batch_size`
            and `with_replacment` become salient parameters; if the optimizer is 'gd', then the returned batch is the
            same (and of size equal to the number of training points) each time. 
        '''
        if split == 'out': 
            indices = np.random.choice(np.arange(len(self.out_images)), size=batch_size, replace=False)
            sample_images = self.ins_images[indices]
            sample_labels = self.ins_labels [indices]
            return sample_images, sample_labels
        if opt == 'gd': 
            return self.sample_images, self.sample_labels
        if with_replacement:
            indices = np.random.choice(self.ins_size, size)
            return self.ins_inputs[indices], self.ins_outputs[indices]
        if self.index + batch_size > self.sample_size: 
            assert(self.index == self.sample_size)
            indices = np.random.shuffle(np.arange(self.sample_size))
            self.sample_images = self.sample_images[indices] 
            self.sample_labels = self.sample_outputs[indices] 
            self.index = 0
        rtrn = self.sample_images[self.index:self.index+batch_size], self.sample_labels[self.index:self.index+batch_size]
        self.index += batch_size
        return rtrn

    def get_all(self, split='ins'):
        ''' Returns whole in-sample or out-of-sample points.  Good for evaluating train and test scores. ''' 
        if split == 'out':
            return self.out_images, self.out_labels
        return self.sample_images, self.sample_labels


################################################################################
#           1. LEARNING MODEL                                                  #
################################################################################

class Learner(object):
    ''' Creates, (re)initializes, trains, and evaluates a differentiable learner. '''
    def __init__(self, precision=PRECISION):
        self.create_model(precision)
        self.create_trainer(precision)
        self.session = tf.Session()
        self.initialize_weights(*self.sample_init_weights())

    def create_model(self, precision=tf.float32):
        ''' Define the loss landscape as a function of weights and data. '''
        self.Images = tf.placeholder(precision, shape=[None, 28*28])
        self.Labels= tf.placeholder(precision, shape=[None])

        self.Weights = tf.get_variable('flattened', shape=[28*28*1], dtype=precision)
        self.WeightsA = tf.reshape(self.Weights, [28*28, 1]) 

        self.InitWeightsA = tf.placeholder(precision, shape=[28*28*1])
        self.Inits = tf.concat([tf.reshape(init, [-1]) for init in [self.InitWeightsA]], axis=0)
        self.Initializer = tf.assign(self.Weights, self.Inits)

        self.Logits = tf.reshape(tf.matmul(self.Images, self.WeightsA), [-1])
        self.Losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.Labels, logits=self.Logits) 

    def create_trainer(self, precision=tf.float32):
        ''' Define the loss and corresponding gradient-based update.  The difference between gd and sgd is not codified
            here; instead, the difference lies in the size and correlations between batches we use to train the
            classifier, i.e. in the values assigned to `Data` and `TrueOutputs` at each gradient update step.
        '''
        self.LearningRate = tf.placeholder(dtype=precision)

        self.Loss = tf.reduce_mean(self.Losses)
        self.GradientWeights = tf.convert_to_tensor(tf.gradients(self.Loss, self.Weights))[0]
        self.Update = tf.tuple([
            tf.assign(self.Weights, self.Weights - self.LearningRate * self.GradientWeights), 
        ])

    def sample_init_weights(self): 
        ''' Sample weights (as numpy arrays) distributed according to Glorot-Bengio recommended length scales.  These
            weights are intended to be initializers. 
        '''
        wa = 0.0 + 0.0 * np.random.randn(28*28) / (28*28)**0.5
        return (wa,)

    def initialize_weights(self, wa):
        ''' Initialize weights as a RUNTIME OPERATION, not by creating new graph nodes. '''
        self.session.run(self.Initializer, feed_dict={
            self.InitWeightsA:wa,
        })

    def run(self, dataset, ins_time, batch_size, learning_rate, opt='sgd'): 
        ''' Compute post-training metrics for given dataset and hyperparameters.  Return in-sample loss and
            out-of-sample loss --- in that order.  This learning task has no auxilliary metrics such as accuracy. 
        '''
        for t in range(ins_time):
            batch_inputs, batch_outputs = dataset.get_batch(batch_size=batch_size, split='ins', opt=opt) 
            self.session.run([self.Update], feed_dict={
                self.Images:batch_inputs,
                self.Labels:batch_outputs,
                self.LearningRate:learning_rate
            }) 
        
        ins_inputs, ins_outputs = dataset.get_all('ins') 
        #out_inputs, out_outputs = dataset.get_all('out') 
        out_inputs, out_outputs = dataset.get_batch(batch_size=batch_size, split='out')
        ins_los = self.session.run(self.Loss, feed_dict={self.Images:ins_inputs,self.Labels:ins_outputs})
        out_los = self.session.run(self.Loss, feed_dict={self.Images:out_inputs, self.Labels:out_outputs})
        return ins_los, out_los



################################################################################
#           2. STATISTICS LOGGER                                               #
################################################################################

class Logger(object):
    ''' Collects and aggregates the metric scores of each trial.  Maintains a dictionary (of score-lists) indexed by
        keys that are intended to represent the experimental conditions yielding those scores.
    '''
    def __init__(self): 
        self.scores = {}

    def append(self, key, value):
        ''' Log (key, value) pair.  Intended typical use: key is tuple describing experimental condition, for instance
            recording learning rate, optimizer type, and so forth; value is individual score TO BE APPENDED TO A LIST.  
        '''
        if key not in self.scores:
            self.scores[key] = [] 
        self.scores[key].append(value)

    def gen_diffs(self): 
        ''' From GD and SGD logs, create DIFF log. '''
        for key in list(self.scores.keys()): 
            if key[5] != 'gd': continue
            key_ = key[:5] + ('sgd',) + key[6:]
            key__= key[:5] + ('diff',) + key[6:]
            if key_ not in self.scores: continue
            self.scores[key__] = [gd_val-sgd_val for gd_val, sgd_val in zip(self.scores[key], self.scores[key_])] 

    def get_stats(self, key):
        ''' Compute mean, sample deviation, min, and max '''
        s = np.array(self.scores[key])
        mean = np.mean(s)
        var = np.mean(np.square(s - mean)) * len(s)/(len(s)-1)
        return mean, np.sqrt(var), np.amin(s), np.amax(s)

    def write_summary(self, file_nm, key_renderer):
        ''' Compute statistics and append to file. '''
        self.gen_diffs()
        with open(file_nm, 'a') as f: 
            for key in self.scores:  
                stats = self.get_stats(key)
                f.write('%s:\t%.10f\t%.10f\t%.10f\t%.10f' % ((key_renderer(*key),)+stats)) 
                f.write('\n')



################################################################################
#           3. EXPERIMENT LOOP                                                 #
################################################################################

def run_experiment(nb_trials, ins_size, ins_time, batch_size, learning_rates):
    ''' Compute the 'eta curve', that is, sgd vs gd test performance as a function of learning rate.  Record summary 
        statistics by appending to the given filename.
    '''
    dataset = Dataset()
    learner = Learner()
    logger = Logger() 

    try:
        for learning_rate in learning_rates:
            print('LR =', learning_rate)
            for h in tqdm(range(nb_trials)):
                dataset.resample(ins_size)
                init_weights = learner.sample_init_weights() 
                for opt in ('gd', 'sgd'): 
                    learner.initialize_weights(*init_weights)
                    il, ol = learner.run(dataset, ins_time, batch_size, learning_rate, opt) 
                    logger.append((nb_trials, ins_size, ins_time, batch_size, learning_rate, opt, 'IL'), il)
                    logger.append((nb_trials, ins_size, ins_time, batch_size, learning_rate, opt, 'OL'), ol)
    except KeyboardInterrupt:
        pass

    logger.write_summary(FILE_NM,
        key_renderer=lambda nb_trials, ins_size, ins_time, batch_size, learning_rate, opt, metric:
            'NB_TRIALS=%d\tINS_SIZE=%d\tINS_TIME=%d\tBATCH_SIZE=%d\tLEARNING_RATE=%.10f\tOPT=%s\tMETRIC=%s' %
            (nb_trials, ins_size, ins_time, batch_size, learning_rate, opt, metric) 
    )

if __name__=='__main__':
    arith_prog = [MIN_LR + (MAX_LR-MIN_LR)*(float(i)/LR_SWEEP) for i in range(0, LR_SWEEP+1)]
    run_experiment(nb_trials=NB_TRIALS, ins_size=INS_SIZE, ins_time=INS_SIZE, batch_size=1, learning_rates=arith_prog)
