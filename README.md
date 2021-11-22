# NMF-visualization
Compute NMF transcriptions for music and display frame by frame visualizations.

Example NMF transcriptions are included for a solo bassoon excerpt from the
Bach10 dataset and the same excerpt with a mix of violin, clarinet, saxophone
and bassoon parts.

NMF transcriptions are by default obtained from spectrograms with 10 ms hop
sizes, 46 ms Hamming windows and a 2048 bin STFT. Dictionaries are pre-computed.

## Visualize transcriptions and spectrograms
Show an animation of the frames of spectrograms or transcriptions and how they
change over time.

Run
```
 $ python visualize.py spectrograms [spectrograms ...] [-l [LABELS ...]]
```

_spectrograms_ is a list of spectrogram and transcription paths to be compared.
Optionally an array of labels can be added using the -l flag.

## Requirements
- Librosa 0.8.1
- NumPy 1.20.3
- Matplotlib 3.4.3
