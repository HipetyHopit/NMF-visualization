"""@package NMF-visualization.

Compare dictionaries.
"""

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import math

if (__name__ == "__main__"):

    dictionaries = ["data/bassoon_dictionary.npy",
                    "data/dictionaries/bassoon-extended.npy"]

    labels = ["Bassoon", "Extended bassoon"]
    Fs = 44100

    W = []

    for d, label in zip(dictionaries, labels):
        W += [np.load(d).T]

        w = W[-1].T
        w = librosa.amplitude_to_db(w, ref=np.max)

        fig = plt.figure()
        ax = plt.subplot(111)
        img = librosa.display.specshow(w, ax = ax, sr = Fs, y_axis = "linear",
                                       x_axis = "cqt_note", fmin = librosa.midi_to_hz(34))
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(label)
        fig.colorbar(img, ax = ax, label = "Magnitude (dB)")

        plt.show()
        plt.close()

    for i in range(W[0].shape[0]):
        fig, axs = plt.subplots(len(W))
        x = np.arange(W[0].shape[1])*Fs/2048

        print (34 + i)

        for j in range(len(W)):
            axs[j].plot(x, W[j][i], label = labels[j])
            axs[j].set_xlabel("Frequency (Hz)")
            axs[j].set_title(labels[j])

        plt.tight_layout()

        plt.show()
