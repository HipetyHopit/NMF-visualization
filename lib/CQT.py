"""@package CQT
Functions for generating constant-Q transform spectrograms.
"""

import librosa
import math
import numpy as np

def CQTspectrogram(x, Fs, hopLen = 10, fftSize = 2048,
                   fMin = 27.5, numOctaves = 8,
                   octaveBins = 60, **kwargs):
    """
    Calculate a magnitude spectrogram.
    
    NOTE the number of CQT bins = octaves x bins per octave
    
    Keyword arguments:
    x -- input signal.
    Fs -- sampling frequency.
    hopLen -- the hop length in ms. (default = H)
    fftSize -- the FFT size. (default = FFT_SIZE)
    fMin -- the lowest bin frequency. (default = CQT_F1)
    numOctaves -- the number of octaves. (default = CQT_OCTAVES)
    octaveBins -- the number of bins per octave. 
        (default = CQT_OCTAVE_BINS)
    
    Returns:
    S -- the CQT spectrogram representation of x.
    """
    
    numBins = numOctaves*octaveBins
    
    hopTarget = math.floor(hopLen*Fs/1000)
    hopSize = 2**(numBins//octaveBins)
    hopSize0 = hopSize
    while (hopTarget > hopSize):
        hopSize += hopSize0
        
    # Check minimum signal length.
    f1 = fMin*np.power(2, numOctaves-1)     # Minimum bin frequency 
                                            # of highest octave.
    q = 1
    N = q*Fs/(f1*(np.power(2, 1/octaveBins) - 1))   # Bin window size.
    fftPoints = 1
    while (fftPoints < N):
        fftPoints *= 2
    minLen = fftPoints*np.power(2, numOctaves-1)
    
    if (len(x) < minLen):
        x = np.concatenate((x, np.zeros(minLen - len(x))))  # Pad.
    
    # CQT
    S = librosa.cqt(x, sr = Fs, hop_length = hopSize, fmin = fMin, 
                    n_bins = numBins, bins_per_octave = octaveBins, 
                    window = WINDOW, pad_mode = PAD_MODE)
    
    return abs(S)
