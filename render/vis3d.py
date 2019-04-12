'''
'''

import numpy as np
import imageio 

H = 128
W = 128

xs = np.expand_dims(np.arange(float(W)), axis=0) / float(W)
ys = np.expand_dims(np.arange(float(H)), axis=1) / float(H)
coor = [ys, xs, 0.0*xs]

d = lambda a,b: np.sqrt(np.square(a[0]-b[0]) + np.square(a[1]-b[1])) 
d3 = lambda a,b: np.sqrt(np.square(a[0]-b[0]) + np.square(a[1]-b[1]) + np.square(a[2]-b[2])) 


def render(line_segs, colors): 
    canvas = np.zeros((H, W, 3), np.float32)
        
    thereness = []
    for (start, end, t) in line_segs:
        thereness.append(
            np.exp(-np.square((d(start, end)-d(start, coor)-d(coor, end))/t) - start[2]),
        )
    thereness = np.array(thereness) 
    thereness = thereness / np.expand_dims(np.sum(thereness, axis=0), axis=0)

    for i, t in enumerate(thereness):
        canvas += np.expand_dims(t, 3) * np.array([[colors[i]]])

    return canvas 

colors = ([
            [0.0, 0.0, 0.0],
            [1.0, 0.6, 0.1],
            [1.0, 0.1, 0.6],
            [0.8, 0.9, 0.0],
            [0.0, 0.9, 0.8],
            [0.6, 0.1, 1.0],
            [0.1, 0.6, 1.0],
        ])

line_segs = [
    ((0.5, 0.5, 5.0), (0.5, 0.5, 5.0), 1.000),
    ((0.2, 0.2, 0.0), (0.2, 0.8, 0.0), 0.005),
    ((0.5, 0.2, 2.0), (0.5, 0.8, 2.0), 0.005),
    ((0.8, 0.2, 4.0), (0.8, 0.8, 4.0), 0.005),
]
imageio.imwrite('rendered-size-depth.png', (255*render(line_segs, colors)).astype(np.uint8))

line_segs = [
    ((0.5, 0.5, 5.0), (0.5, 0.5, 5.0), 1.000),
    ((0.2, 0.2, 0.0), (0.2, 0.8, 0.0), 0.005),
    ((0.5, 0.2, 0.0), (0.5, 0.8, 0.0), 0.010),
    ((0.8, 0.2, 0.0), (0.8, 0.8, 0.0), 0.015),
]
imageio.imwrite('rendered-size-linear.png', (255*render(line_segs, colors)).astype(np.uint8))

line_segs = [
    ((0.5, 0.5, 5.0), (0.5, 0.5, 5.0), 1.000),
    ((0.4, 0.2, 0.0), (0.4, 0.8, 0.0), 0.005),
    ((0.2, 0.8, 2.0), (0.8, 0.2, 2.0), 0.005),
    ((0.8, 0.8, 4.0), (0.2, 0.2, 4.0), 0.005),
]
imageio.imwrite('rendered-occlusion.png', (255*render(line_segs, colors)).astype(np.uint8))

line_segs = [
    ((0.5, 0.5, 5.0), (0.5, 0.5, 5.0), 1.000),
    ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), 0.200),
    ((0.5, 0.7, 1.0), (0.5, 0.7, 1.0), 0.100),
    ((0.7, 0.7, 1.0), (0.7, 0.7, 1.0), 0.100),
    ((0.3, 0.3, 1.5), (0.3, 0.3, 1.5), 0.400),
    ((0.3, 0.3, 0.5), (0.3, 0.9, 0.5), 0.100),
]
imageio.imwrite('rendered-mixing.png', (255*render(line_segs, colors)).astype(np.uint8))



colors = ([
            [0.0 , 0.2 , 0.0],
            [0.3 , 0.2 , 0.0],
            [0.3 , 0.2 , 0.0],
            [0.3 , 0.2 , 0.0],
            [0.3 , 0.2 , 0.0],
            [0.3 , 0.2 , 0.0],
        ])
line_segs = [
    ((0.5, 0.5, 5.0), (0.5, 0.5, 5.0), 1.000),
    ((0.5, 0.3, 0.0), (0.5, 0.7, 0.0), 0.050),
    ((0.5, 0.3, 0.0), (0.8, 0.3, 0.0), 0.010),
    ((0.5, 0.4, 0.0), (0.8, 0.4, 0.0), 0.010),
    ((0.5, 0.6, 0.0), (0.8, 0.6, 0.0), 0.010),
    ((0.5, 0.7, 0.0), (0.8, 0.7, 0.0), 0.010),
]
imageio.imwrite('rendered-deer.png', (255*render(line_segs, colors)).astype(np.uint8))



colors = ([
            [0.0 , 0.0 , 0.0],
            [0.2 , 0.2 , 0.0],
            [0.2 , 0.5 , 0.0],
            [0.2 , 0.8 , 0.0],
            [0.5 , 0.2 , 0.0],
            [0.5 , 0.5 , 0.0],
            [0.5 , 0.8 , 0.0],
            [0.8 , 0.2 , 0.0],
            [0.8 , 0.5 , 0.0],
            [0.8 , 0.8 , 0.0],
        ])

def random():
    return [((0.5, 0.5, 5.0), (0.5, 0.5, 5.0), 3.0)] + [
        ((np.random.random(), np.random.random(), 3.0 * np.random.random()),
         (np.random.random(), np.random.random(), 3.0 * np.random.random()), 0.001 + 0.009*np.random.random())
        for i in range(9)
    ]

imageio.imwrite('rendered-random.png', (255*render(random(), colors)).astype(np.uint8))
