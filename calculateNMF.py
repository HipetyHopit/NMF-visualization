"""@package NMF-visualization

Caculate NMF transcriptions.
"""

import argparse
import librosa
import numpy as np
import os.path

from lib.NMF import NMF
from lib.NMF import transcribeInstrument
from lib.normalize import maxNormalize
from lib.normalize import RMSnormalize
from lib.normalize import sumNormalize
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
    parser.add_argument("--updateW", default = False, action = "store_true")
    parser.add_argument("-b",
                        help = "Beta. (default = 0.5)",
                        type = float, default = 0.5, dest = "beta")
    parser.add_argument("--noFrame", help = "Don't use frame NMF.",
                        default = False, action = "store_true")
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
            S = np.load(path + ".npy")
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
            W = np.load(path + ".npy")
            break
        except:
            pass

        try:
            W = np.load(DICTIONARY_PATH + path)
            break
        except:
            pass

        try:
            W = np.load(DICTIONARY_PATH + path + ".npy")
            break
        except:
            pass

        print ("Could not load dictionary file!")
        raise SystemExit()

    if (args.norm.lower() == "max"):
        V = maxNormalize(S, axis = 1)
    elif (args.norm.lower() == "rms"):
        V = RMSnormalize(S, axis = 1)
    elif (args.norm.lower() == "sum"):
        V = sumNormalize(S, axis = 1)
    else:
        V = S

    numBins, numNotes = W.shape

    if (args.noFrame):
        H, _ = NMF(V, H = None, W = W, k = numNotes, threshold = 0.001,
                iterations = 200, updateW = args.updateW, verbose = False)
    else:
        H = transcribeInstrument(V, W, beta = args.beta)

    path = args.excerpt
    if (args.saveAs is None):
        path = NMF_PATH + path + ".npy"
    else:
        path = args.saveAs
    createDir(path)
    np.save(path, H)
