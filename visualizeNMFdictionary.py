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
                        + "(default = 100)", type = int, default = 100,
                        dest = "interval")
    parser.add_argument("-m", "--mix", help = "Display a mixed isntrument "
                        + "transcription", default = False, dest = "mix",
                        action = "store_true")
    parser.add_argument("--instrument",
                        help = "The instrument part to display. "
                        + "(default = 'bassoon')", type = str,
                        default = "bassoon", dest = "instrument")
    args = parser.parse_args()

    if (args.mix):
        spectrogram = np.load("data/mix_spectrogram.npy").T
        dictionary = np.load("data/%s_dictionary.npy" % args.instrument).T
        nmf = np.load("data/mix-%s_NMF.npy" % args.instrument)
        truth = np.load("data/mix-%s_truth.npy" % args.instrument)
    else:
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
    FFTbins = 2048
    numComp = 3
    plotBins = 256

    plots = []
    notes = []
    correlations = []
    numFrames, numBins = spectrogram.shape
    for i in range(numFrames):
        transcriptionDict = dictionary[transcriptionDictIndx[i]]
        truthDict = dictionary[truthDictIndx[i]]
        spect = spectrogram[i]
        notes += [(transcriptionDictIndx[i], truthDictIndx[i])]

        correlations += [(correlation(transcriptionDict, spect),
                          correlation(truthDict, spect))]

        transcriptionDict = 0.9*transcriptionDict/np.max(abs(transcriptionDict))
        truthDict = 0.9*truthDict/np.max(abs(truthDict))
        spect = 0.9*spect

        plots += [(truthDict[:plotBins], transcriptionDict[:plotBins],
                   spect[:plotBins])]

    # Animate dictionary entries.
    toAnimate = []

    fig, ax = plt.subplots()
    k = np.arange(0, plotBins)*(Fs/FFTbins)
    for i in range(numComp):
        line, = ax.plot(k, plots[0][1] + i, label = labels[i])
        toAnimate += [line]
    time = ax.text(0, numComp*2 - 1, "t = %.3g s" % 0.0)
    midiNotes = ax.text(0, numComp*2 - 2,
                        "Transcription note: %d   True note: %d" % notes[0])
    r = ax.text(0, numComp*2 - 3, ("Transcription correlation: %f   "
                + "True correlation: %f") % correlations[0])
    legend = ax.legend(loc = 1)
    toAnimate += [time, midiNotes, r, legend]
    ax.set_ylim(-1, numComp*2)
    ax.set_yticklabels([])
    ax.set_xlabel("Frequency (Hz)")

    # Enable pause
    running = True

    def onClick(event):
        global running
        if running:
            ani.event_source.stop()
            running = False
        else:
            ani.event_source.start()
            running = True

    fig.canvas.mpl_connect('button_press_event', onClick)

    def animate(i):
        for j in range(numComp):
            toAnimate[j].set_ydata(plots[i][j] + j)
        time.set_text("t = %.3g s" % (i*hopLen/1000))
        midiNotes.set_text("Transcription note: %d   True note: %d" % notes[i])
        r.set_text(("Transcription correlation: %f   "
                    + "True correlation: %f") % correlations[i])
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
