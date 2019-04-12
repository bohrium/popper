'''
'''

import numpy as np
import imageio 

H = 28 
W = 28

xs = np.expand_dims(np.arange(float(W)), axis=0)
ys = np.expand_dims(np.arange(float(H)), axis=1)
coor = [ys, xs]

d = lambda a,b: np.sqrt(np.square(a[0]-b[0]) + np.square(a[1]-b[1])) 

def render(line_segs): 
    canvas = np.zeros((H, W), np.float32)
    for (start, end, t) in line_segs:
        canvas = np.maximum(canvas, 
            np.exp((d(start, end)-d(start, coor)-d(coor, end)) / t)
        )
    return canvas 

def random():
    return [
        ((np.random.randint(H), np.random.randint(W)),
         (np.random.randint(H), np.random.randint(W)), 0.1 + 0.4*np.random.random())
        for i in range(6)
    ]

imageio.imwrite('r.png', (255*render(random())).astype(np.uint8))

line_segs = [
    (( 5,15), ( 5, 20), 0.3),
    ((10,10), ( 5, 15), 0.4), 

    ((10,10), (15, 20), 0.5),

    ((20,15), (15, 20), 0.5),
    ((20, 5), (20, 15), 0.4),

    ((20, 5), ( 5, 20), 0.3),
]

#imageio.imwrite('synthetic-8.png', (255*canvas).astype(np.uint8))

#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
#imgs, lbls = mnist.train.images, mnist.train.labels
#imgs = np.reshape(imgs, (-1, 28, 28))
#
#best_score, best_img, best_lbl = min((np.mean(np.abs(img-canvas)), img, lbl) for img, lbl in zip(imgs, lbls))
#imageio.imwrite('closest-mnist.png', (255*best_img).astype(np.uint8))
#print(best_score, best_lbl)
