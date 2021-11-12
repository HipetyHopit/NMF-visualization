"""@package NMF-visualization

Visualize NMF transcriptions.
"""

import librosa.display
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from multipleInstrumentTranscription.constants import H as HOP_LENGTH
from multipleInstrumentTranscription.datasets import getTracks
from multipleInstrumentTranscription.NMF import loadInstrumentModel
from multipleInstrumentTranscription.NMF import NMF
from overwriteConstants import *

if (__name__ == "__main__"):

    instruments = ["bassoon"]
    instrumentRange = BASSOON_RANGE
    dataset = "development_instrumentation1"

    numNotes = instrumentRange[1] - instrumentRange[0] + 1
    f0 = librosa.midi_to_hz(instrumentRange[0])

    excerpts = getTracks(dataset, split = None, instruments = instruments)
    excerptPath = excerpts[0]

    S = np.load(SPECTROGRAM_PATH + excerptPath + ".npy")

    # # Animate spectrogram.
    # numBins, numFrames = S.shape

    # fig, ax = plt.subplots()
    # k = np.arange(0, numBins)
    # line, = ax.plot(k, S[:, 0])

    # def animate(i):
    #     line.set_ydata(S[:, i])  # update the data
    #     return line,

    # # Init only required for blitting to give a clean slate.
    # def init():
    #     line.set_ydata(np.ma.array(k, mask=True))
    #     return line,

    # ani = animation.FuncAnimation(fig, animate, np.arange(0, numFrames), 
    #                               init_func = init, interval = 10, blit = True,
    #                               repeat = False)

    # plt.show()

    # Calculate NMF.
    W, H = NMF(S, H = None, W = None, k = numNotes,  verbose = True)

    # Plot W
    fig = plt.figure()
    ax = plt.subplot(111)
    Wspec = librosa.amplitude_to_db(W, ref = np.max)
    img = librosa.display.specshow(Wspec, ax = ax, x_axis = "cqt_note", 
                                    y_axis = "linear", fmin = f0, 
                                    sr = SAMPLE_RATE)
    ax.set_ylabel("Frequency (Hz)")
    plt.tight_layout()
    plt.show()

    # Plot H
    fig = plt.figure()
    ax = plt.subplot(111)
    hopSize = int(HOP_LENGTH*SAMPLE_RATE/1000)
    img = librosa.display.specshow(H, ax = ax, sr = SAMPLE_RATE, 
                                   hop_length = hopSize, x_axis = "s", 
                                   y_axis = "cqt_note", fmin = f0)
    plt.tight_layout()
    plt.show()
