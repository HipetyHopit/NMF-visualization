"""@package NMF-visualization

Visualize the effect of beta in frames of NMF transcriptions.
"""

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

    if (np.max(x1) > 0):
        x1 = x1/np.max(x1)

    if (np.max(x2) > 0):
        x2 = x2/np.max(x2)

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

    if (np.max(x1) > 0):
        x1 = x1/np.max(x1)

    if (np.max(x2) > 0):
        x2 = x2/np.max(x2)

    r = np.sum(np.square(abs(x1 - x2)))

    return r


instrumentMinNote = {"violin": 55,
                     "clarinet": 50,
                     "saxophone":49,
                     "bassoon": 34}

notesNames = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

def spectrumToNote(s, offset = 0):
    """
    Return the frame note.
    """

    if (np.max(s) == 0):
        return "None"
    else:
        midi = np.argmax(s) + offset
    note = notesNames[midi%12] + str(midi//12 - 1)

    return note

def midiToNote(midi):
    """
    Return the note name of a midi number.
    """

    if (midi < 0):
        return "None"

    note = notesNames[midi%12] + str(midi//12 - 1)

    return note

def getDictionarySpectrums(W, H):
    """
    Return a list of spectrums corresponding to transcribed
    dictionary entries.
    """

    numBins, numNotes = W.shape
    numNotes, numFrames = H.shape

    W = W.T
    dictIndx = np.argmax(H, axis = 0)
    silentDict = np.zeros(numBins)

    plots = []

    for i in range(numFrames):
        transcriptionDict = W[dictIndx[i]]
        if (np.max(H[:, i]) == 0):
            transcriptionDict = silentDict
        #print (transcriptionDict.shape)
        #break
        plots += [transcriptionDict]

    return plots

def getNMFSpectrums(W, H):
    """
    Return a list of spectrums from NMF.
    """

    numBins, numNotes = W.shape
    numNotes, numFrames = H.shape

    V = np.matmul(W, H)

    return V.T

if (__name__ == "__main__"):

    interval = 100
    step = True

    hopLen = 10
    Fs = 44100

    transcriptions = ["data/solo-bassoon_truth",
                       "data/NMFs/bassoon_beta_-1",
                       #"data/NMFs/bassoon_beta_-05",
                       "data/NMFs/bassoon_beta_0",
                       #"data/NMFs/bassoon_beta_05",
                       "data/NMFs/bassoon_beta_1",
                       #"data/NMFs/bassoon_beta_15",
                       "data/NMFs/bassoon_beta_2"]
    #labels = ["Truth", "Beta = -1", "Beta = -0.5", "Beta = 0", "Beta = 0.5",
              #"Beta = 1", "Beta = 1.5", "Beta = 2"]
    labels = ["Truth", "Beta = -1", "Beta = 0", "Beta = 1", "Beta = 2"]

    # Animate dictionary entries.
    toAnimate = []

    fig, ax = plt.subplots(len(transcriptions))

    dataSize = 0.5
    dataMargin = 0.75

    # Plot transcriptions.
    for i in range(len(transcriptions)):
        transcriptions[i] = np.load(transcriptions[i] + ".npy").T
        line, = ax[i].plot(transcriptions[i][0])
        ax[i].set_title(labels[i])
        ax[i].set_ylim(-dataMargin*np.max(transcriptions[i]),
                       np.max(transcriptions[i]))
        if (i == len(transcriptions) - 1):
            ax[i].set_xlabel("Note bin")
        else:
            ax[i].set_xticks([])

        toAnimate += [line]

    numFrames, numNotes = transcriptions[0].shape

    # Plot info.
    time = ax[0].text(1, -dataSize*np.max(transcriptions[0]),
                      "t = %.3g s" % 0.0)

    toAnimate += [time]

    plt.tight_layout()

    # Enable pause
    running = True

    # Enable step
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

        if (event.key == "enter" or event.key == "right"):
            frameIndx += 1
        if (event.key == "left"):
            frameIndx -= 1

        frameIndx %= numFrames

    if (step):
        fig.canvas.mpl_connect('key_press_event', onKeyPress)
    else:
        fig.canvas.mpl_connect('button_press_event', onClick)

    def animate(i):
        global step, frameIndx

        if (step):
            indx = frameIndx
        else:
            indx = i

        # Plot info.
        time.set_text("t = %.3g s" % (frameIndx*hopLen/1000))

        # Plot transcriptions.
        for j in range(len(transcriptions)):
            toAnimate[j].set_ydata(transcriptions[j][indx])

        return toAnimate

    def init():
        for j in range(len(transcriptions)):
            k = np.arange(len(transcriptions[j][0]))
            toAnimate[j].set_ydata(np.ma.array(k, mask=True))
        time.set_text("")

        return tuple(toAnimate)

    ani = animation.FuncAnimation(fig, animate, np.arange(0, numFrames),
                                  init_func = init, interval = interval,
                                  blit = True, repeat = True)

    plt.show()
