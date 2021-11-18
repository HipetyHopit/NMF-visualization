"""@package NMF-visualization

Visualize NMF transcriptions.
"""

import argparse
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

if (__name__ == "__main__"):

    parser = argparse.ArgumentParser("Visualize and compare mixed instrument "
                                     + "NMF transcriptions.")
    parser.add_argument("--interval",
                        help = "The display time for each frame in ms. "
                        + "(default = 10)",
                        type = int, default = 10, dest = "interval")
    args = parser.parse_args()

    fullRange = (34, 100)

    NMFs = ["mix-bassoon_truth", "mix-bassoon_NMF", "mix-saxophone_truth",
            "mix-saxophone_NMF", "mix-clarinet_truth", "mix-clarinet_NMF",
            "mix-violin_truth", "mix-violin_NMF"]
    labels = ["Bassoon truth", "Bassoon NMF", "Saxophone truth",
              "Saxophone NMF", "Clarinet truth", "Clarinet NMF", "Violin truth",
              "Violin NMF"]
    ranges = [(34, 74), (34, 74), (49, 80), (49, 80), (50, 95), (50, 95),
              (55, 100), (55, 100)]

    hopLen = 10

    plots = []
    for NMF, r in zip(NMFs, ranges):
        H = np.load("data/" + NMF + ".npy")
        pad1 = np.zeros((r[0] - fullRange[0], H.shape[1]))
        pad2 = np.zeros((fullRange[1] - r[1], H.shape[1]))
        H = np.concatenate((pad1, H, pad2), axis = 0)
        plots += [0.9*H.T/np.max(abs(H))]
    numComp = len(plots)

    # Animate NMFs.
    numFrames, numNotes = plots[0].shape

    toAnimate = []

    fig, ax = plt.subplots()
    k = np.arange(0, numNotes) + fullRange[0]
    for i in range(numComp):
        line, = ax.plot(k, plots[i][0] + i, label = labels[i])
        toAnimate += [line]
    time = ax.text(fullRange[0], numComp*2 - 1, "t = %.3g s" % 0.0)
    legend = ax.legend(loc = 1)
    toAnimate += [time, legend]
    ax.set_ylim(-1, numComp*2)
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
