"""@package NMF-visualization

Functions for normalizing matrices
"""

import numpy as np

def maxNormalize(X, axis = None):

    if (axis is None):
        return X/np.max(X)

    return np.apply_along_axis(maxNormalize, axis, X)

    #norm = np.max(X, axis = axis - 1)
    #print (norm.shape, norm)
    #Y = np.divide(X, norm)

    #return Y

def RMSnormalize(X, axis = None):

    if (axis is None):
        N = np.multiply(*X.shape)
        rms = np.sqrt(np.sum(abs(np.square(X)))/N)

        return X/np.max(X)

    return np.apply_along_axis(RMSnormalize, axis, X)
