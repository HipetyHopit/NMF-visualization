"""@package NMF-visualization

Visualize NMF transcriptions.
"""

import argparse
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

NMF_PATH = "data/NMFs/"
SPECTROGRAM_PATH = "data/spectrograms/"
TRUTH_PATH = "data/truths/"

if (__name__ == "__main__"):

    parser = argparse.ArgumentParser("Visualize and compare spectrograms and "
                                     + "transcriptions.")
    parser.add_argument("spectrograms", help = "Spectrograms or transcriptions "
                        + "to visualize", nargs = '+')
    parser.add_argument("-l", "--labels", help = "Lables corresponding to each "
                        + "spectrogram/transcription", nargs = '*',
                        default = [], dest = "labels")
    parser.add_argument("-i", "--interval",
                        help = "The display time for each frame in ms. "
                        + "(default = 10)",
                        type = int, default = 10, dest = "interval")
    parser.add_argument("-m", "--minNote",
                        help = "MIDI number of the lowest note bin."
                        + "(default = 0)",
                        type = int, default = 0, dest = "minNote")
    args = parser.parse_args()

    hopLen = 10

    if (len(args.spectrograms) > len(args.labels)):
        labels += [""]*(len(args.spectrograms) - len(args.labels))

    plots = []
    for path in args.spectrograms:
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
                S = np.load(NMF_PATH + path)
                break
            except:
                pass

            try:
                S = np.load(NMF_PATH + path + ".npy")
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

            try:
                S = np.load(TRUTH_PATH + path)
                break
            except:
                pass

            try:
                S = np.load(TRUTH_PATH + path + ".npy")
                break
            except:
                pass

            print ("Could not load %s!" % path)
            raise SystemExit()

        plots += [S.T/np.max(abs(S))]
    numComp = len(plots)

    # Animate NMFs.
    numFrames, numNotes = plots[0].shape

    toAnimate = []

    fig, ax = plt.subplots()
    k = np.arange(0, numNotes) + args.minNote
    for i in range(numComp):
        line, = ax.plot(k, plots[i][0] + i, label = args.labels[i])
        toAnimate += [line]
    time = ax.text(args.minNote, numComp, "t = %.3g s" % 0.0)
    legend = ax.legend(loc = 1)
    toAnimate += [time, legend]
    ax.set_ylim(-1, numComp + 1)
    ax.set_yticklabels([])
    ax.set_xlabel("Note (MIDI)")

    def animate(i):
        for j in range(numComp):
            toAnimate[j].set_ydata(plots[j][i] + j)
        time.set_text("t = %.3g s" % (i*hopLen/1000))
        return toAnimate

    def init():
        for j in range(numComp):
            toAnimate[j].set_ydata(np.ma.array(k, mask=True))
        time.set_text("")
        return (line, time)

    ani = animation.FuncAnimation(fig, animate, np.arange(0, numFrames),
                                  init_func = init, interval = args.interval,
                                  blit = True, repeat = True)

    plt.show()
