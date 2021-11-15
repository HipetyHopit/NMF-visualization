"""@package NMF-visualization

Compare different spectrogram normalization methods for NMF.
"""

import numpy as np

from lib.NMF import NMF
from lib.normalize import maxNormalize
from lib.normalize import RMSnormalize
from lib.spectrogram import magnitudeSpectrogram
from lib.utils import createDir

AUDIO_PATH = "data/audio/"
SPECTROGRAM_PATH = "data/spectrograms/"

if (__name__ == "__main__"):

    excerpt = "bassoon-solo"
    hopLen = 10   # ms
    Fs = 44100

    try:
        print ("Loading spectrogram.")
        S = np.load(SPECTROGRAM_PATH + excerpt + ".npy")
    except:
        print ("Spectrogram not found. Calcualting from audio.")
        path = AUDIO_PATH + excerpt + ".wav"
        x, Fs = librosa.load(path, sr = Fs, mono = True)
        S = magnitudeSpectrogram(x, Fs)
        path = SPECTROGRAM_PATH + excerpt + ".npy"
        createDir(path)
        np.save(path, S)

    Y = RMSnormalize(S)
    print (Y.shape)
