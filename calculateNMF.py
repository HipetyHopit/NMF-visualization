"""@package NMF-visualization

Caculate NMF transcriptions.
"""

import argparse
import librosa
import numpy as np
import os.path

from lib.spectrogram import magnitudeSpectrogram
from lib.utils import createDir

DICTIONARY_PATH = "data/dictionaries/"
NMF_PATH = "data/NMFs/"
SPECTROGRAM_PATH = "data/spectrograms/"

if (__name__ == "__main__"):

    parser = argparse.ArgumentParser("Calculate an NMF transcription.")
    parser.add_argument("excerpt", help = "The excerpt spectrogram file.",
                        type = str)
    parser.add_argument("dictionary", help = "The instrument dictionary file.",
                        type = str)
    parser.add_argument("--norm",
                        help = "The spectrogram normalization. " +
                        "(default = 'max')", type = str, default = "max",
                        dest = "norm")
    parser.add_argument("-d",
                        help = "The destination file. (default = None)",
                        type = str, default = None, dest = "saveAs")
    args = parser.parse_args()

    # Load spectrogram.
    path = args.excerpt
    while (True):
        try:
            S = np.load(path)
            break
        except:
            pass

        try:
            S = np.np.load(path + ".npy")
            break
        except:
            pass

        try:
            S = np.load(SPECTROGRAM_PATH + path)
            break
        except:
            pass

        try:
            S = np.load(SPECTROGRAM_PATH + path + ".npy")
            break
        except:
            pass

        print ("Could not load spectrogram file!")
        raise SystemExit()

    # Load dictionary.
    path = args.dictionary
    while (True):
        try:
            W = np.load(path)
            break
        except:
            pass

        try:
            W = np.np.load(path + ".npy")
            break
        except:
            pass

        try:
            W = np.load(SPECTROGRAM_PATH + path)
            break
        except:
            pass

        try:
            W = np.load(SPECTROGRAM_PATH + path + ".npy")
            break
        except:
            pass

        print ("Could not load spectrogram file!")
        raise SystemExit()

    H, W = NMF(V, H = None, W = W, threshold = 0.0001, iterations = 200,
        verbose = False, seed = 314, **kwargs):

    S = magnitudeSpectrogram(x, Fs = args.Fs, hopLen = args.hopLen)

    if (args.saveAs is None):
        path = SPECTROGRAM_PATH + path + ".npy"
    else:
        path = args.saveAs
    createDir(path)
    np.save(path, S)
