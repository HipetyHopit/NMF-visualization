"""@package spectrogram
Generate music sample spectrograms.

Functions for generating spectrograms for input into transcription
algorithms.
"""

import librosa
import math
import numpy as np

def magnitudeSpectrogram(x, Fs = 44100, windowLen = 46, hopLen = 10,
                         fftSize = 2048, window = "hamming",
                         padMode = "constant", **kwargs):
    """
    Calculate a magnitude spectrogram.
    
    Keyword arguments:
    x -- input signal.
    Fs -- sampling frequency. (default = 2048)
    windowLen -- the window length in ms. (default = N)
    hopLen -- the hop length in ms. (default = H)
    fftSize -- the FFT size. (default = FFT_SIZE)
    window -- the window to use. (default = WINDOW)
    padMode -- the pad mode to use. (default = PAD_MODE)
    
    Returns:
    S -- the magnitude spectrogram representation of x.
    """
    
    windowSize = math.floor(windowLen*Fs/1000)
    hopSize = math.floor(hopLen*Fs/1000)
    
    S = librosa.stft(x, n_fft = fftSize, hop_length = hopSize, 
                     win_length = windowSize, window = window, 
                     pad_mode = padMode, center = False)
    S, phase = librosa.magphase(S)
    
    return S
