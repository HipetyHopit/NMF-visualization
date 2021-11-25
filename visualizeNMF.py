"""@package NMF-visualization

Visualize NMF transcriptions.
"""

import argparse
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

if (__name__ == "__main__"):

    parser = argparse.ArgumentParser("Visualize and compare NMF "
                                     + "transcriptions.")
    parser.add_argument("--interval",
                        help = "The display time for each frame in ms. "
                        + "(default = 100)",
                        type = int, default = 100, dest = "interval")
    args = parser.parse_args()

    NMFs = ["bassoon-solo_truth", "bassoon-solo_NMF",
            "bassoon-solo_no-updateNMF", "bassoon-solo_updateNMF"]
    labels = ["Truth", "Frame NMF", "NMF without W update", "NMF with W update"]

    hopLen = 10

    plots = []
    for NMF in NMFs:
        H = np.load("data/" + NMF + ".npy")
        plots += [H.T/np.max(abs(H))]
    numComp = len(plots)

    # Animate NMFs.
    numFrames, numNotes = plots[0].shape

    toAnimate = []

    fig, ax = plt.subplots()
    k = np.arange(0, numNotes) + 34 # Bassoon lowest note is 34.
    for i in range(numComp):
        line, = ax.plot(k, plots[i][0] + i, label = labels[i])
        toAnimate += [line]
    time = ax.text(34, numComp, "t = %.3g s" % 0.0)
    legend = ax.legend(loc = 1)
    toAnimate += [time, legend]
    ax.set_ylim(-1, numComp + 1)
    ax.set_yticklabels([])
    ax.set_xlabel("Note (MIDI)")

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
