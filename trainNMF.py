"""@package NMF-visualization

Visualize NMF transcriptions.
"""

import argparse
import librosa
import numpy as np

from lib.CQT import CQTspectrogram
from lib.NMF import NMF
from lib.normalize import maxNormalize
from lib.normalize import RMSnormalize
from lib.normalize import sumNormalize
from lib.spectrogram import magnitudeSpectrogram
from lib.utils import createDir

NMF_DICTIONARY_PATH = "data/dictionaries/"
INSTRUMENT_INFO_PATH = "data/paths/"

def trainDictionary(instrument, instrumentRange = None, infoFile = None,
                    cqt = False, dictionaryPath = None, normalization = None,
                    Fs = 44100, fftSize = 2048, numOctaves = 8, octaveBins = 60,
                    **kwargs):
    """
    Train an instrument model with NMF.

    Keyword arguments:
    instrument -- the instrument name.
    instrumentRange -- the range of MIDI notes to cover as a tuple.
    infoFile -- the path to a textfile containg instrument note paths.
        (default = None)
    cqt -- whether to use CQT instead of STFT. (default = False)
    dictionaryPath -- path to save . (default = None)
    normalization -- the spectrogram normalization function.
        (default = None)
    Fs -- the sample rate. (default = SAMPLE_RATE)
    fftSize -- the number of FFT bins. (default =2048)
    numOctaves -- the number of CQT octaves. (default = 8)
    octaveBins -- the number of bins per CQT octavee. (default = 60)
    """

    kwargs.setdefault("fftSize", fftSize)
    kwargs.setdefault("numOctaves", numOctaves)
    kwargs.setdefault("octaveBins", octaveBins)

    if (infoFile is None):
        infoFile = INSTRUMENT_INFO_PATH + instrument + ".txt"

    with open(infoFile, "r") as f:
        lines = f.readlines()

    if (instrumentRange is None):
        minNote = -1
        maxNote = 0
        for line in lines:
            midi = int(line.split()[0])
            if (minNote == -1 or midi < minNote):
                minNote = midi
            elif (midi > maxNote):
                maxNote = midi
        instrumentRange = (minNote, maxNote)

    numNotes = instrumentRange[1] - instrumentRange[0] + 1

    if (cqt):
        numBins = octaveBins*numOctaves
    else:
        numBins = 1 + fftSize//2

    W = np.zeros((numNotes, numBins))

    for line in lines:
        line = line.split()
        i = int(line[0]) - instrumentRange[0]

        if (i < 0 or i >= numNotes):
            continue

        notePath = line[1]

        x, Fs = librosa.load(notePath, sr = Fs, mono = True)
        if (cqt):
            S = CQTspectrogram(x, Fs, **kwargs)
        else:
            S = magnitudeSpectrogram(x, Fs, **kwargs)

        if (not normalization is None):
            S = normalization(S, axis = 1)

        h, w = NMF(S, k = 1, **kwargs)
        W[i] = w.flatten()

    if (dictionaryPath is None):
        dictionaryPath = NMF_DICTIONARY_PATH

    path = dictionaryPath + instrument + ".npy"

    createDir(path)
    np.save(path, W.T)

if (__name__ == "__main__"):

    parser = argparse.ArgumentParser("Compute an NMF dictionary.")
    parser.add_argument("instrument", help = "The instrument to train a "
                        + "dictionary for.", type = str)
    parser.add_argument("--info",
                        help = "A textfile containing paths to note samples. "
                        + "(default = None)", type = str, default = None,
                        dest = "info")
    parser.add_argument("--norm",
                        help = "The spectrogram normalization. " +
                        "(default = 'max')", type = str, default = "max",
                        dest = "norm")
    parser.add_argument("-d",
                        help = "The destination file. (default = None)",
                        type = str, default = None, dest = "saveAs")
    args = parser.parse_args()


    if (args.norm.lower() == "max"):
        norm = maxNormalize
    elif (args.norm.lower() == "rms"):
        norm = RMSnormalize
    elif (args.norm.lower() == "sum"):
        norm = sumNormalize
    else:
        norm = None

    trainDictionary(args.insrument, normalization = norm,
                    dictionaryPath = args.dest)
