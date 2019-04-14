'''
'''

import numpy as np
import imageio 

H = 280 
W = 280

xs = np.expand_dims(np.arange(float(W)), axis=0) / W
ys = np.expand_dims(np.arange(float(H)), axis=1) / H
coor = [ys, xs]

d = lambda a,b: np.sqrt(np.square(a[0]-b[0]) + np.square(a[1]-b[1])) 
def transform(start, end, coor, bend=0.0):
    mid = [(start[0]+end[0])/2, (start[1]+end[1])/2]
    scale = d(end, mid) 
    dotA = ((coor[0] - mid[0]) * (end[0] - mid[0]) + (coor[1] - mid[1]) * (end[1] - mid[1]))/(1e-4+scale)
    dotB = ((coor[0] - mid[0]) * (end[1] - mid[1]) - (coor[1] - mid[1]) * (end[0] - mid[0]))/(1e-4+scale)
    return [
        dotA,
        dotB - bend * np.square(dotA)
    ]

def render(line_segs): 
    canvas = np.zeros((H, W), np.float32)
    for (start, end, thick, bend) in line_segs:
        s = transform(start, end, start, bend)
        e = transform(start, end, end  , bend)
        c = transform(start, end, coor , bend)
        canvas = np.maximum(canvas, 
            np.exp(-np.square((d(s, e)-d(s, c)-d(c, e)) / thick**2)),
        )
    return np.maximum(0, np.minimum(1, canvas))

def random():
    return [
        ((0.1+0.8*np.random.random(), 0.1+0.8*np.random.random()),
         (0.1+0.8*np.random.random(), 0.1+0.8*np.random.random()),
          0.01 + 0.09*np.random.random(),
         -4.0 + 8.0*np.random.random(),
         )
        for i in range(12)
    ]

imageio.imwrite('r.png', (255*render(random())).astype(np.uint8))

line_segs = [
    ((.3, .4), (.1, .7), 0.08, 6.0),
    ((.7, .4), (.7, .8), 0.05,-6.0),

    ((.3, .4), (.4, .6), 0.05,-4.0),
    ((.3, .7), (.4, .6), 0.08, 4.0),

    ((.7, .4), (.4, .6), 0.05, 4.0),
    ((.7, .8), (.4, .6), 0.05,-4.0),

]

imageio.imwrite('synthetic-8.png', (255*render(line_segs)).astype(np.uint8))


#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
#imgs, lbls = mnist.train.images, mnist.train.labels
#imgs = np.reshape(imgs, (-1, 28, 28))
#
#best_score, best_img, best_lbl = min((np.mean(np.abs(img-canvas)), img, lbl) for img, lbl in zip(imgs, lbls))
#imageio.imwrite('closest-mnist.png', (255*best_img).astype(np.uint8))
#print(best_score, best_lbl)
