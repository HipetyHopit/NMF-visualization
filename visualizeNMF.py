"""@package NMF-visualization

Visualize NMF transcriptions.
"""

import librosa
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from lib.spectrogram import magnitudeSpectrogram
from lib.utils import createDir

NMF_PATH = "data/NMFs/"
SPECTROGRAM_PATH = "data/spectrograms/"
TRUTHS_PATH = "data/truths/"

if (__name__ == "__main__"):

    excerpt = "bassoon-solo"
    hopLen = 10   # ms
    Fs = 44100

    try:
        print ("Loading spectrogram.")
        S = np.load(SPECTROGRAM_PATH + excerpt + ".npy")
    except:
        print ("Spectrogram not found. Calcualting from audio.")
        path = AUDIO_PATH + excerpt + ".wav"
        x, Fs = librosa.load(path, sr = Fs, mono = True)
        S = magnitudeSpectrogram(x, Fs)
        path = SPECTROGRAM_PATH + excerpt + ".npy"
        createDir(path)
        np.save(path, S)

    # Animate spectrogram.
    numBins, numFrames = S.shape

    fig, ax = plt.subplots()
    k = np.arange(0, numBins)
    line, = ax.plot(k, S[:, 0])
    time = ax.text(800, 10, "t = %.3g s" % 0.0)

    def animate(i):
        line.set_ydata(S[:, i])
        time.set_text("t = %.3g s" % (i*hopLen/1000))
        return (line, time)

    def init():
        line.set_ydata(np.ma.array(k, mask=True))
        time.set_text("")
        return (line, time)

    ani = animation.FuncAnimation(fig, animate, np.arange(0, numFrames),
                                  init_func = init, interval = hopLen,
                                  blit = True, repeat = True)

    plt.show()
