"""@package NMF-visualization

Functions for normalizing matrices
"""

import numpy as np

def maxNormalize(X, axis = None):

    if (axis is None):
        return X/np.max(X)

    return np.apply_along_axis(maxNormalize, axis, X)

def sumNormalize(X, axis = None):

    if (axis is None):
        return X/np.sum(X)

    return np.apply_along_axis(sumNormalize, axis, X)

def RMSnormalize(X, axis = None):

    if (axis is None):
        N = X.size
        rms = np.sqrt(np.sum(abs(np.square(X)))/N)

        return X/np.max(X)

    return np.apply_along_axis(RMSnormalize, axis, X)
