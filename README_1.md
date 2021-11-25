# NMF-visualization
Compute NMF transcriptions for music and display frame by frame visualizations.

Example NMF transcriptions are included for a solo bassoon excerpt from the
Bach10 dataset and the same excerpt with a mix of violin, clarinet, saxophone
and bassoon parts.

NMF transcriptions were obtained from spectrograms with 10 ms hop sizes, 46 ms
Hamming windows and a 2048 bin STFT. Dictionaries are pre-computed.

## Visualize solo NMF transcription

NMF transcriptions of the solo bassoon excerpt are shown along with the ground
truth transcription. Three transcriptions are compared: frame-wise NMF with the
Beta-divergence (the method that was used up until now),  as well as NMF for the
entire spectrogram simultaneously with the dictionary being kept constant or
being updated along with the H matrix.

Run
```
 $ python visualizeNMF.py
```

## Visualize mixed NMF transcription

NMF transcriptions of each instrument in the mix instrument are shown along with
the ground  truth transcriptions. Each instrument's transcription was calculated
seperately with a solo instrument dictionary. (The same method has been used as
input to the neural network up until now.)

Run
```
 $ python visualizeMixNMF.py
```

## Visualize NMF dictionary entries

NMF transcriptions of the solo bassoon excerpt are shown. The dictionary entries
in W corresponding to the correct note and the NMF transcribed note are shown.
Correlation coefficents between the spectrogram and the estimated dictionary
entries as well as the correct dictionary entries are shown.

Run
```
 $ python visualizeNMFdictionary.py [-m] [--instrument INSTRUMENT]
```
Add the -m flag to see transcription dictionaries for the mix excerpt.
Optionally a different instrument in the mix can be seen using the --instrument
parameter.

## Additional functionality

The time between animation frames for any of the programs can be changes by
using the --interval parameter to specify the interval between frames in ms
(defualt is 100 ms). Additional parameters can be seen using the -h flag.

Any animation can be paused by clicking on it and unpaused again by clicking.

## Requirements

- NumPy 1.20.3
- Matplotlib 3.4.3
