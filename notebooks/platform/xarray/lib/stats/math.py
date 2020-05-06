import math
import numpy as np


def r(gn0, gn1):
    """Rogers Huff R

    Lift from https://github.com/cggh/scikit-allel/blob/961254bd583572eed7f9bd01060e53a8648e620c/allel/opt/stats.pyx
    """
    # initialise variables
    m0 = m1 = v0 = v1 = cov = n = 0

    # iterate over input vectors
    for i in range(len(gn0)):
        x = gn0[i]
        y = gn1[i]
        # consider negative values as missing
        if x >= 0 and y >= 0:
            n += 1
            m0 += x
            m1 += y
            v0 += x ** 2
            v1 += y ** 2
            cov += x * y

    # early out
    if n == 0 or v0 == 0 or v1 == 0:
        return np.nan

    # compute mean, variance, covariance
    m0 /= n
    m1 /= n
    v0 /= n
    v1 /= n
    cov /= n
    cov -= m0 * m1
    v0 -= m0 * m0
    v1 -= m1 * m1

    # compute correlation coeficient
    r = cov / math.sqrt(v0 * v1)

    return r


def r2(gn0, gn1):
    return r(gn0, gn1) ** 2
