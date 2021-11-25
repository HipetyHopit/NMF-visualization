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

## Requirements

- NumPy 1.20.3
- Matplotlib 3.4.3
