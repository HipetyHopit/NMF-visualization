"""@package NMF-visualization
NMF first principles library

Functions for approximating the factors of a non-negative matrix.
"""

import numpy as np

def betaDivergence(X, Y, beta = 0.5):
    """
    Return the beta-divergence for X and its
    approximation..

    Keyword arguments:
    X - a matrix.
    Y - X's approximation
    beta -- (default 0.5)
    
    Returns:
    c -- the beta divergence.
    """
    
    c = (np.power(X, beta) + (beta-1)*np.power(Y, beta) 
            - beta*np.multiply(X, np.power(Y, beta - 1)))
    c = np.sum(c)/(beta*(beta - 1))
    
    return c

def frobenius(X, Y):
    """
    Return the Frobenius norm for X and its
    approximation.

    Keyword arguments:
    X - a matrix.
    Y - X's approximation
    
    Returns:
    c -- the Frobenius norm.
    """
    
    A = abs(X - Y)

    return 0.5*np.sum(np.power(A, 2))

def KLD(X, Y):
    """
    Return the Kullback-Liebler divergence for X and its
    approximation.
    
    Keyword arguments:
    X - a matrix.
    Y - X's approximation
    
    Returns:
    c -- the Kullback-Liebler divergence.
    """
    
    # Avoid zero division
    if (abs(Y).min() == 0):
        Y += np.full(Y.shape, 1e-32)
        print ("Warning! Almost divided by 0!")
    
    kld = np.multiply(X, np.log10(X/Y)) - X + Y
    
    return np.sum(kld)

def NMF(V, H = None, W = None, k = 1, threshold = 0.0001, iterations = 200, 
        verbose = False, **kwargs):

    if (W is None):
        W = 1 - np.random.rand(V.shape[0], k)

    if (H is None):
        H = 1 - np.random.rand(k, V.shape[1])

    for i in range(iterations):
        H_ = H.copy()

        # Update H.
        WV = np.matmul(W.T, V)
        WW = np.matmul(W.T, W)
        WWH = np.matmul(WW, H)
        H = H*WV/WWH

        # Update W.
        VH = np.matmul(V, H_.T)
        WH = np.matmul(W, H_)
        WHH = np.matmul(WH, H_.T)
        W = W*VH/WHH

        # Get cost.
        Vapprox = np.matmul(W, H)
        c = 0.5*np.sum(np.power(V - Vapprox, 2))

        if (verbose):
            print (("Iteration %d, cost = %.3g" % (i, c)) + 10*' ', end = '\r')

        if (c < threshold):
            break

    return H, W      

def frameNMF(v, W, beta = 0.5, h = None, threshold = 0.0001, cost = frobenius,
             iterations = 200, **kwargs):
    """
    Return h, the approximation for the unkown factor of v.
    
    Approximate h such that v = W.h, using the multiplicative
    update rule with W known and v and h being vectors taken over
    a short time.
    
    Keyword arguments:
    v -- the time sample to factorize.
    W -- the known factor matrix.
    beta -- the beta divergence parameter. (default 2)
    h -- an initialization for h. (default = None)
    threshold -- the cost threshold. (default = 0.0001)
    cost -- the cost function. (default = frobenius)
    iterations -- the maximum number of iterations. (default = 200)
    """

    assert (v.shape[1] == 1)
    if (h is None):
        h = np.matrix(1 - np.random.rand(W.shape[1], 1))
    e = np.matrix(np.ones(W.shape[1]))
    f = np.multiply(W, v*e).T
    c = 0
    
    for i in range(iterations):
        
        # Update H
        a = np.power(W*h, beta - 2)
        b = np.power(W*h, beta - 1)
        h = np.multiply(h, f*a/(W.T*b))
        
        # Calculate cost
        c = cost(v, W*h)
         
        if (c <= threshold):
            break
    
    h = np.squeeze(np.asarray(h))
    
    return h, c

def transcribeInstrument(V, W, cost = "frobenius", **kwargs):
    """
    Return H, the approximation for the unkown factor of V.
    
    Approximate H such that V = W.H, using the multiplicative
    update rule with W known.
    
    Keyword arguments:
    V -- the matrix to factorize.
    W -- the known factor.
    cost -- the cost function. (default = "frobenius")
    """
    
    numBins, numFrames = V.shape
    numBins, numNotes = W.shape
    
    H = np.empty((numFrames, numNotes))
    
    cost = cost.lower()
    if (cost == "kld"):
        c = KLD
    elif (cost == "beta"):
        if ("beta" in kwargs):
            def c(X, Y):
                return betaDivergence(X, Y, kwargs["beta"])
        else :
            c = betaDivergence
    else:
        c = frobenius
    
    for i in range(numFrames):
        v = V[:, i].reshape((numBins, 1))
        H[i], _ = frameNMF(v, W, cost = c, **kwargs)
    
    return H.T
