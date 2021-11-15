"""@package NMF-visualization

Compare different spectrogram normalization methods for NMF.
"""

import numpy as np

from tabulate import tabulate

from lib.NMF import betaDivergence
from lib.NMF import frobenius
from lib.NMF import KLD
from lib.NMF import NMF
from lib.normalize import maxNormalize
from lib.normalize import RMSnormalize
from lib.normalize import sumNormalize
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

    table = []
    headers = ["Normalization", "Frobenius norm", "Beta divergence", "KLD"]
    fmt = ".3g"

    # No normalization.
    print ("Calculating NMF with no normalization.")
    V = S
    H, W = NMF(V, verbose = True)
    Vestimate = W*H
    frobeniusError = frobenius(V, Vestimate)
    betaError = betaDivergence(V, Vestimate)
    kldError = KLD(V, Vestimate)
    table += [["None", frobeniusError, betaError, kldError]]

    # Sum normalization.
    print ("Calculating NMF with sum normalization.")
    V = sumNormalize(S, axis = 1)
    H, W = NMF(V, verbose = True)
    Vestimate = W*H
    frobeniusError = frobenius(V, Vestimate)
    betaError = betaDivergence(V, Vestimate)
    kldError = KLD(V, Vestimate)
    table += [["Sum", frobeniusError, betaError, kldError]]

    # Max normalization.
    print ("Calculating NMF with max normalization.")
    V = maxNormalize(S, axis = 1)
    H, W = NMF(V, verbose = True)
    Vestimate = W*H
    frobeniusError = frobenius(V, Vestimate)
    betaError = betaDivergence(V, Vestimate)
    kldError = KLD(V, Vestimate)
    table += [["Max", frobeniusError, betaError, kldError]]

    # RMS normalization.
    print ("Calculating NMF with RMS normalization.")
    V = RMSnormalize(S, axis = 1)
    H, W = NMF(V, verbose = True)
    Vestimate = W*H
    frobeniusError = frobenius(V, Vestimate)
    betaError = betaDivergence(V, Vestimate)
    kldError = KLD(V, Vestimate)
    table += [["RMS", frobeniusError, betaError, kldError]]

    print(tabulate(table, headers, tablefmt = "github", floatfmt = fmt))
