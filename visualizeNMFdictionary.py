"""@package NMF-visualization

Visualize mixed NMF transcriptions.
"""

import argparse
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

def correlation(x1, x2):
    """
    Return the corelation coefficient of two signals.

    Keyword arguments:
    x1 -- the first signal.
    x2 -- the second signal.

    Returns:
    r -- the correlation coefficient.
    """

    assert (len(x1) == len(x2))

    N = len(x1)
    r = np.sum(np.multiply(x1, x2))
    r /= N

    return r

if (__name__ == "__main__"):

    parser = argparse.ArgumentParser("Visualize and compare the dictionary "
                                     + "entries for an NMF transcription.")
    parser.add_argument("--interval",
                        help = "The display time for each frame in ms. "
                        + "(default = 10)",
                        type = int, default = 10, dest = "interval")
    args = parser.parse_args()

    instrumentRange = (34, 74)

    spectrogram = np.load("data/bassoon-solo_spectrogram.npy").T
    dictionary = np.load("data/bassoon_dictionary.npy").T
    nmf = np.load("data/bassoon-solo_NMF.npy")
    truth = np.load("data/bassoon-solo_truth.npy")

    labels = ["True W entry", "Estimated W entry", "Spectrogram"]

    transcriptionDictIndx = np.argmax(nmf, axis = 0)
    truthDictIndx = np.argmax(truth, axis = 0)
    spectrogram = spectrogram/np.max(abs(spectrogram))

    hopLen = 10
    Fs = 44100
    numComp = 3

    plots = []
    notes = []
    correlations = []
    numFrames, numBins = spectrogram.shape
    for i in range(numFrames):
        transcriptionDict = dictionary[transcriptionDictIndx[i]]
        truthDict = dictionary[truthDictIndx[i]]
        notes += [(transcriptionDictIndx[i], truthDictIndx[i])]

        correlations += [correlation(transcriptionDict, truthDict)]

        transcriptionDict = 0.9*transcriptionDict/np.max(abs(transcriptionDict))
        truthDict = 0.9*truthDict/np.max(abs(truthDict))

        plots += [(truthDict, transcriptionDict, 0.9*spectrogram[i])]

    # Animate dictionary entries.
    toAnimate = []

    fig, ax = plt.subplots()
    k = np.arange(0, numBins)*(Fs/numBins)
    for i in range(numComp):
        line, = ax.plot(k, plots[0][1] + i, label = labels[i])
        toAnimate += [line]
    time = ax.text(0, numComp*2 - 1, "t = %.3g s" % 0.0)
    midiNotes = ax.text(0, numComp*2 - 2,
                        "Transcription note: %d   True note: %d" % notes[0])
    r = ax.text(0, numComp*2 - 3, "Correlation: %f" % correlations[0])
    legend = ax.legend(loc = 1)
    toAnimate += [time, midiNotes, r, legend]
    ax.set_ylim(-1, numComp*2)
    ax.set_yticklabels([])
    ax.set_xlabel("Frequency (Hz)")

    def animate(i):
        for j in range(numComp):
            toAnimate[j].set_ydata(plots[i][j] + j)
        time.set_text("t = %.3g s" % (i*hopLen/1000))
        midiNotes.set_text("Transcription note: %d   True note: %d" % notes[i])
        r.set_text("Correlation: %f" % correlations[i])
        return toAnimate

    def init():
        for j in range(numComp):
            toAnimate[j].set_ydata(np.ma.array(k, mask=True))
        time.set_text("")
        midiNotes.set_text("")
        r.set_text("")
        return (line, time)

    ani = animation.FuncAnimation(fig, animate, np.arange(0, numFrames),
                                  init_func = init, interval = args.interval,
                                  blit = True, repeat = True)

    plt.show()
