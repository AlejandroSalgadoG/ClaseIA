import numpy as np

colors = np.array([
            [31, 119, 180, 255],
            [255, 127, 14, 255],
            [44, 160, 44, 255],
            [214, 39, 40, 255],
            [148, 103, 189, 255],
            [140, 86, 75, 255],
            [227, 119, 194, 255],
            [127, 127, 127, 255],
            [188, 189, 34, 255],
            [23, 190, 207, 255]
         ]) / 255

def get_color( i, alpha=None ):
    color = colors[ i % 10 ].copy()
    if alpha is not None: color[-1] = alpha
    return color
