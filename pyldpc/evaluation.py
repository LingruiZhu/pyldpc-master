import numpy as np


def fer_IS(x, y, weights):
    n, n_trails = x.shape
    fer = 0
    for i in range(n_trails):
        if abs(x[:, i] - y[:, i]).sum() != 0:
            fer = 1*weights[i] / n_trails

    return fer
