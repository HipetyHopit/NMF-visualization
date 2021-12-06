"""@package NMF-visualization

Visualize mixed NMF transcriptions.
"""

import argparse
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from lib.NMF import betaDivergence

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

def rmse(x1, x2):
    """
    Return the RMSE of two signals.

    Keyword arguments:
    x1 -- the first signal.
    x2 -- the second signal.

    Returns:
    r -- the RMSE.
    """

    assert (len(x1) == len(x2))

    r = np.sum(np.square(abs(x1 - x2)))

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
    parser.add_argument("-s", "--step", help = "Step through transcription",
                        default = False, dest = "step",
                        action = "store_true")
    parser.add_argument("-b", "--beta", help = "Beta for the beta divergence "
                        + "measure.", type = float, default = 0.5,
                        dest = "beta",)
    parser.add_argument("--instrument",
                        help = "The instrument part to display. "
                        + "(default = 'bassoon')", type = str,
                        default = "bassoon", dest = "instrument")
    args = parser.parse_args()

    def beta(x1, x2):
        """ Beta divergence wrapper function. """

        return betaDivergence(x1, x2, beta = args.beta)

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

    divergenceFunctions = [correlation, rmse, beta]
    divergenceLabels = ["correlation", "RMSE", "beta-divergence"]

    plots = []
    notes = []
    divergences = []
    numFrames, numBins = spectrogram.shape
    for i in range(numFrames):
        transcriptionDict = dictionary[transcriptionDictIndx[i]]
        truthDict = dictionary[truthDictIndx[i]]
        spect = spectrogram[i]
        notes += [(transcriptionDictIndx[i], truthDictIndx[i])]

        frameDivergences = []
        for f in divergenceFunctions:
            frameDivergences += [(f(transcriptionDict, spect),
                                  f(truthDict, spect))]
        divergences += [frameDivergences]

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
    time = ax.text(0, numComp + len(divergenceFunctions) + 1,
                   "t = %.3g s" % 0.0)
    midiNotes = ax.text(0, numComp + len(divergenceFunctions),
                        "Transcription note: %d   True note: %d" % notes[0])
    divergenceText = []
    for i in range(len(divergenceFunctions)):
        divergenceText += [ax.text(0, numComp + i,
                                   "Transcription %s: %f   True %s: %f"
                                   % (divergenceLabels[i], divergences[0][i][0],
                                   divergenceLabels[i], divergences[0][i][1]))]
    legend = ax.legend(loc = 1)
    toAnimate += [time, midiNotes, legend]
    toAnimate += divergenceText
    ax.set_ylim(-1, numComp + len(divergenceFunctions) + 2)
    ax.set_yticklabels([])
    ax.set_xlabel("Frequency (Hz)")

    # Enable pause
    running = True

    # Enable step
    step = args.step
    frameIndx = 0

    def onClick(event):
        global running
        if running:
            ani.event_source.stop()
            running = False
        else:
            ani.event_source.start()
            running = True

    def onKeyPress(event):
        global frameIndx

        if (event.key == "enter"):
            frameIndx += 1

    if (step):
        fig.canvas.mpl_connect('key_press_event', onKeyPress)
    else:
        fig.canvas.mpl_connect('button_press_event', onClick)

    def animate(i):
        global step, frameIndx

        if (step):
            for j in range(numComp):
                toAnimate[j].set_ydata(plots[frameIndx][j] + j)
            time.set_text("t = %.3g s" % (frameIndx*hopLen/1000))
            midiNotes.set_text("Transcription note: %d   True note: %d"
                               % notes[frameIndx])
            for j in range(len(divergenceFunctions)):
                divergenceText[j].set_text("Transcription %s: %f   True %s: %f"
                    % (divergenceLabels[j], divergences[frameIndx][j][0],
                    divergenceLabels[j], divergences[frameIndx][j][1]))
        else:
            for j in range(numComp):
                toAnimate[j].set_ydata(plots[i][j] + j)
            time.set_text("t = %.3g s" % (i*hopLen/1000))
            midiNotes.set_text("Transcription note: %d   True note: %d"
                               % notes[i])
            for j in range(len(divergenceFunctions)):
                divergenceText[j].set_text("Transcription %s: %f   True %s: %f"
                    % (divergenceLabels[j], divergences[i][j][0],
                    divergenceLabels[j], divergences[i][j][1]))
        return toAnimate

    def init():
        for j in range(numComp):
            toAnimate[j].set_ydata(np.ma.array(k, mask=True))
        time.set_text("")
        midiNotes.set_text("")
        for i in range(len(divergenceFunctions)):
            divergenceText[i].set_text("")
        return (line, time)

    ani = animation.FuncAnimation(fig, animate, np.arange(0, numFrames),
                                  init_func = init, interval = args.interval,
                                  blit = True, repeat = True)

    plt.show()
