''' author: samtenka
    change: 2019-02-12
    create: 2019-02-12
    descrp: Demonstrate robustness of generative classification
'''

import tensorflow as tf
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def sample_data(n, sigma, offset):
    gauss = np.random.randn(n, 2) * sigma 
    angles = np.random.uniform(size=n) * 6.2832
    phases = np.stack([np.cos(angles), np.sin(angles)], axis=1) 
    radii = np.stack([np.random.laplace(size=n)]*2, axis=1) 
    bias = np.stack([[offset, 0]]*n, axis=0)
    superlaplace = radii * phases + bias 
    return gauss, superlaplace

N = 100
SIGMA, OFFSET = 2.0, 2.0

gauss, slapl = sample_data(n=N, sigma=SIGMA, offset=OFFSET)
gmean = np.mean(gauss, axis=0)
gvari = np.mean([np.std(gauss[:,i])**2 for i in range(2)]) * N/(N-1.0)
smean = np.mean(slapl, axis=0)
svari = np.mean([np.std(slapl[:,i])**2 for i in range(2)]) * N/(N-1.0)
print(gmean, gvari)

def gauss_logprob(x, sigma=SIGMA):
    return - np.linalg.norm(x)**2 / (2*sigma**2) - np.log(6.2832*sigma**2)
def slapl_logprob(x, offset=OFFSET):
    x -= np.array([offset, 0])
    return - np.linalg.norm(x) - np.log(6.2832 * np.linalg.norm(x))

def gauss_logprob_est(x, sigma=SIGMA):
    return - np.linalg.norm(x-gmean)**2 / (2*gvari) - np.log(6.2832 * gvari)
def slapl_logprob_est(x, offset=OFFSET):
    x -= np.array([offset, 0])
    return - np.linalg.norm(x-smean)**2 / (2*svari) - np.log(6.2832 * svari)

def unknown(x, alpha):
    g = np.exp(gauss_logprob_est(x))
    s = np.exp(slapl_logprob_est(x))
    return g/(g+s+alpha), s/(g+s+alpha)

def gauss_wins(x):
    return gauss_logprob(x, sigma=SIGMA) - slapl_logprob(x, offset=OFFSET) 

plt.scatter(gauss[:,0], gauss[:,1], color='#ff0000', alpha=0.2)
plt.scatter(slapl[:,0], slapl[:,1], color='#008080', alpha=0.2)

plot_ellipses=False
if plot_ellipses:
    e = Ellipse(xy=gmean, width=2*gvari**0.5, height=2*gvari**0.5)
    e.set_facecolor('#ff0000')
    e.set_alpha(0.5)
    plt.axes().add_artist(e)
    
    e = Ellipse(xy=smean, width=2*svari**0.5, height=2*svari**0.5)
    e.set_facecolor('#00f0f0')
    e.set_alpha(0.5)
    plt.axes().add_artist(e)

plt.axes().set_aspect('equal', 'box')
plt.ylim((-8.0, +8.0))
plt.xlim((-8.0, +8.0))
plt.savefig('real.png')
plt.clf()


for alpha in [0.001, 0.0001, 0.0000]:
    tests = np.random.randn(10000, 2) * 8.0 
    gtests = np.array([x for x in (tests) if unknown(x, alpha)[0]/(1.0-unknown(x, alpha)[0]) > np.exp(1.0)])
    stests = np.array([x for x in (tests) if unknown(x, alpha)[1]/(1.0-unknown(x, alpha)[1]) > np.exp(1.0)])
    
    plt.scatter(gtests[:,0], gtests[:,1], color='#ff0000', alpha=0.1, marker='o')
    plt.scatter(stests[:,0], stests[:,1], color='#008080', alpha=0.1, marker='o')
    plt.axes().set_aspect('equal', 'box')
    plt.ylim((-8.0, +8.0))
    plt.xlim((-8.0, +8.0))
    plt.savefig('gen-%f.png' % alpha if alpha else 'discr.png')
    plt.clf()


tests = np.random.randn(10000, 2) * 8.0 
gtests = np.array([x for x in (tests) if gauss_wins(x)>1.0])
stests = np.array([x for x in (tests) if gauss_wins(x)<-1.0])

plt.scatter(gtests[:,0], gtests[:,1], color='#ff0000', alpha=0.1, marker='o')
plt.scatter(stests[:,0], stests[:,1], color='#008080', alpha=0.1, marker='o')
plt.axes().set_aspect('equal', 'box')
plt.ylim((-8.0, +8.0))
plt.xlim((-8.0, +8.0))
plt.savefig('bayes.png')
plt.clf()



