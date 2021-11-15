"""@package NMF-visualization

Caculate spectrograms.
"""

import argparse
import librosa
import numpy as np
import os.path

from lib.spectrogram import magnitudeSpectrogram
from lib.utils import createDir

AUDIO_PATH = "data/audio/"
SPECTROGRAM_PATH = "data/spectrograms/"

if (__name__ == "__main__"):

    parser = argparse.ArgumentParser("Calculate a spectrogram.")
    parser.add_argument("excerpt", help = "The excerpt audio file.", type = str)
    parser.add_argument("--fs",
                        help = "The sampling frequency. (default = 44100)",
                        type = int, default = 44100, dest = "Fs")
    parser.add_argument("--hop",
                        help = "The hop length in ms. (default = 10)",
                        type = int, default = 10, dest = "hopLen")
    parser.add_argument("-d",
                        help = "The destination file. (default = None)",
                        type = str, default = None, dest = "saveAs")
    args = parser.parse_args()

    path = args.excerpt

    while (True):
        try:
            x, Fs = librosa.load(path, sr = args.Fs, mono = True)
            break
        except:
            pass

        try:
            x, Fs = librosa.load(path + ".wav", sr = args.Fs, mono = True)
            break
        except:
            pass

        try:
            x, Fs = librosa.load(AUDIO_PATH + path, sr = args.Fs, mono = True)
            break
        except:
            pass

        try:
            x, Fs = librosa.load(AUDIO_PATH + path + ".wav", sr = args.Fs,
                                 mono = True)
            break
        except:
            pass

        print ("Could not load audio file!")
        raise SystemExit()

    S = magnitudeSpectrogram(x, Fs = args.Fs, hopLen = args.hopLen)

    if (args.saveAs is None):
        path = SPECTROGRAM_PATH + path + ".npy"
    else:
        path = args.saveAs
    createDir(path)
    np.save(path, S)
