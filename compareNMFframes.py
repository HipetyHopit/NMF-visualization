"""@package NMF-visualization

Visualize frames of NMF transcriptions.
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

    #parser = argparse.ArgumentParser("Visualize and compare the dictionary "
                                     #+ "entries for an NMF transcription.")
    #parser.add_argument("--interval",
                        #help = "The display time for each frame in ms. "
                        #+ "(default = 100)", type = int, default = 100,
                        #dest = "interval")
    #parser.add_argument("-m", "--mix", help = "Display a mixed isntrument "
                        #+ "transcription", default = False, dest = "mix",
                        #action = "store_true")
    #parser.add_argument("-r", "--run", help = "Run through transcription",
                        #default = False, dest = "run",
                        #action = "store_true")
    #parser.add_argument("-b", "--beta", help = "Beta for the beta divergence "
                        #+ "measure.", type = float, default = 0.5,
                        #dest = "beta",)
    #parser.add_argument("--instrument",
                        #help = "The instrument part to display. "
                        #+ "(default = 'bassoon')", type = str,
                        #default = "bassoon", dest = "instrument")
    #parser.add_argument("--TRIOS", help = "Use TRIOS excerpts",
                        #default = False, dest = "trios",
                        #action = "store_true")
    #args = parser.parse_args()

    interval = 100
    step = True

    beta = 0.5
    hopLen = 10
    Fs = 44100
    FFTbins = 2048
    plotBins = 256

    #excerpt = "TRIOS_mix"
    #excerpt = "TRIOS_solo"
    #excerpt = "mix"
    excerpt = "solo"
    mix = False #

    #instrument = "violin"
    #instrument = "clarinet"
    #instrument = "saxophone"
    instrument = "bassoon"

    midiOffset = instrumentMinNote[instrument]

    def betaCost(x1, x2):
        """ Beta divergence wrapper function. """

        if (np.max(x1) > 0):
            x1 = x1/np.max(x1)

        if (np.max(x2) > 0):
            x2 = x2/np.max(x2)

        return betaDivergence(x1, x2, beta = beta)

    def frameNote(x1, x2):
        """
        Return the note of the first signal.

        Keyword arguments:
        x1 -- the first signal.
        x2 -- the redundant signal.

        Returns:
        note -- the note string.
        """

        return spectrumToNote(x1, offset = midiOffset)

    divergenceFunctions = [frameNote, correlation, rmse, betaCost]
    divergenceLabels = ["note", "correlation", "RMSE", "beta-divergence"]

    transcriptions = []

    # Transcriptions to compare:
    transcriptions += [{"label": "Truth",
                        "dictionary": np.load("data/%s_dictionary.npy" % instrument),
                        "transcription": np.load("data/%s-%s_truth.npy" % (excerpt, instrument)),
                        "spectrumFunc": getDictionarySpectrums,
                        "divergenceFuncs": divergenceFunctions,
                        "divergenceLabels": divergenceLabels,
                        }]
    transcriptions += [{"label": "NMF",
                        "dictionary": np.load("data/%s_dictionary.npy" % instrument),
                        "transcription": np.load("data/%s-%s_NMF.npy" % (excerpt, instrument)),
                        "spectrumFunc": getDictionarySpectrums,
                        "divergenceFuncs": divergenceFunctions,
                        "divergenceLabels": divergenceLabels,
                        }]
    transcriptions += [{"label": "NMF Spectrogram Estimate",
                        "dictionary": np.load("data/%s_dictionary.npy" % instrument),
                        "transcription": np.load("data/%s-%s_NMF.npy" % (excerpt, instrument)),
                        "spectrumFunc": getNMFSpectrums,
                        "divergenceFuncs": divergenceFunctions[1:],
                        "divergenceLabels": divergenceLabels[1:],
                        }]

    if (mix):
        spectrogram = np.load("data/%s_spectrogram.npy" % excerpt).T
    else:
        spectrogram = np.load("data/%s-%s_spectrogram.npy" %
                              (excerpt, instrument)).T
    numFrames, numBins = spectrogram.shape
    silentDict = np.zeros(numBins)

    for transcription in transcriptions:

        H = transcription["transcription"]
        W = transcription["dictionary"]
        dictIndx = np.argmax(H, axis = 0)

        transcription["plot"] = transcription["spectrumFunc"](W, H)
        transcription["divergences"] = []

        for i in range(numFrames):
            spect = spectrogram[i]

            frameDivergences = []
            for f in transcription["divergenceFuncs"]:
                d = f(transcription["plot"][i], spect)
                frameDivergences += [str(d)]

            transcription["divergences"] += [frameDivergences]

    # Animate dictionary entries.
    toAnimate = []

    offset = 1
    fig, ax = plt.subplots(len(transcriptions) + offset)
    k = np.arange(0, plotBins)*(Fs/FFTbins)

    dataSize = 0.5
    dataMargin = 0.75

    # Plot transcriptions.
    for i in range(len(transcriptions)):
        line, = ax[i + offset].plot(k, transcriptions[i]["plot"][0][:plotBins])
        ax[i + offset].set_xlabel("Frequency (Hz)")
        ax[i + offset].set_title(transcriptions[i]["label"])
        ax[i + offset].set_ylim(-dataMargin*np.max(transcriptions[i]["plot"]),
                                np.max(transcriptions[i]["plot"]))

        toAnimate += [line]

    # Plot spectrogram.
    line, = ax[0].plot(k, spectrogram[0][:plotBins])
    ax[0].set_xlabel("Frequency (Hz)")
    ax[0].set_title("Spectrogram")
    ax[0].set_ylim(-dataMargin*np.max(spectrogram), np.max(spectrogram))
    toAnimate += [line]

    # Plot info.
    time = ax[0].text(1, -dataSize*np.max(spectrogram), "t = %.3g s" % 0.0)

    toAnimate += [time]

    # Plot stats.
    statsTexts = []
    for i in range(len(transcriptions)):
        stats = ""

        # Plot divergences.
        first = True
        for d, label in zip(transcriptions[i]["divergences"][0],
                            transcriptions[i]["divergenceLabels"]):
            if (not first):
                stats += ", "
            else:
                first = False
            stats += "%s: %s" % (label, d)

        statText = ax[i + offset].text(1,
            -dataSize*np.max(transcriptions[i]["plot"]), stats)
        toAnimate += [statText]
        statsTexts += [statText]

    #plt.tight_layout()

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
            toAnimate[j].set_ydata(transcriptions[j]["plot"][indx][:plotBins])

        toAnimate[len(transcriptions)].set_ydata(spectrogram[indx][:plotBins])

        for j in range(len(transcriptions)):
            stats = ""

            # Plot divergences.
            first = True
            for d, label in zip(transcriptions[j]["divergences"][indx],
                                transcriptions[j]["divergenceLabels"]):
                if (not first):
                    stats += ", "
                else:
                    first = False
                stats += "%s: %s" % (label, d)

            statsTexts[j].set_text(stats)

        return toAnimate

    def init():
        for j in range(len(transcriptions)):
            toAnimate[j].set_ydata(np.ma.array(k, mask=True))
        toAnimate[len(transcriptions)].set_ydata(np.ma.array(k, mask=True))
        time.set_text("")

        for j in range(len(transcriptions)):
            statsTexts[j].set_text("")

        return tuple(toAnimate)

    ani = animation.FuncAnimation(fig, animate, np.arange(0, numFrames),
                                  init_func = init, interval = interval,
                                  blit = True, repeat = True)

    plt.show()
