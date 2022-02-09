"""@package NMF-visualization

Examine spectrogram.
"""

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import math

from scipy.io import wavfile

N = 46  # 46 ms window (to comply with Bach10 dataset).
H = 10  # 10 ms hop size.
FFT_SIZE = 2048 # 2048 FFT points.

def hamming(t):
    """
    Return the Hamming window function at t.

    Keyword arguments:
    t -- the window input.

    Returns:
    w -- the window value at t.
    """

    w = 25/46 + 21/46*np.cos(2*np.pi*t)
    w[t < -0.5] = 0
    w[t > 0.5] = 0

    return w

def stft(s, w, N, H, fftLen = None):
    """
    Return the stft of a signal s.

    Keyword arguments:
    s -- the input signal.
    w -- the window function.
    N -- the window size.
    H -- the hop size.
    fftLen -- the FFT size. (default = N)

    Returns:
    STFT -- the frequency transform of s.
    """

    numWindows = int(np.ceil(len(s)/H))

    if (fftLen == None):
        fftLen = N

    STFT = np.empty((numWindows, fftLen), dtype = np.complex_)

    t = np.arange(N)
    w = w((t-0.5)/N)

    # Zero pad
    s = np.append(np.zeros(N//2), s)
    s = np.append(s, np.zeros(numWindows*N -len(s)))

    for i in range(numWindows):
        si = s[i*H:(i*H+N)]
        si = np.append(si*w, np.zeros(fftLen - N))
        STFT[i] = np.fft.fft(si)
        #STFT[i] = firstPrinciplesFFT(si)

    return STFT

if (__name__ == "__main__"):

    excerpt = "data/audio/mix-TRIOS.wav"

    x, Fs = librosa.load(excerpt, sr = 44100)

    print (np.mean(x), np.max(abs(x)))

    #Fs, x = wavfile.read(excerpt)

    #print (Fs, np.mean(x), np.std(x))

    #x = np.array(x).flatten()
    #print (type(x), x.shape)
    #x /= np.max(abs(x))

    windowSize = math.floor(N*Fs/1000)
    hopSize = math.floor(H*Fs/1000)

    S = librosa.stft(x, n_fft = FFT_SIZE, hop_length = hopSize,
                     win_length = windowSize, window = "hamming",
                     pad_mode = "constant", center = False)
    S, phase = librosa.magphase(S)

    print (np.mean(S[0]), np.max(S[0]))
    print (np.mean(S), np.max(S))

    #print ("")
    #Salt = stft(x, hamming, windowSize, hopSize, FFT_SIZE)
    #S = abs(Salt.T[:FFT_SIZE//2+1])

    #print (np.max(S), np.max(S[0]))
    #print (np.mean(S), np.mean(S[0]))

    #S = librosa.amplitude_to_db(S, ref=np.max)

    #fig = plt.figure()
    #ax = plt.subplot(111)
    #img = librosa.display.specshow(S, ax = ax, sr = Fs, hop_length = hopSize,
                                   #x_axis = "s", y_axis = "linear")
    #ax.set_ylabel("Frequency (Hz)")

    #plt.tight_layout()

    #plt.show()


    fig = plt.figure()
    ax = plt.subplot(111)
    S0 = S[0]
    t = np.arange(len(S0))*H/1000
    ax.plot(t, S0)
    ax.set_xlabel("Time (s)")

    plt.tight_layout()

    plt.show()
