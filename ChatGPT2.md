To automatically estimate the frequency range of the palm-muted guitar chugs in a song, you can use a technique called onset detection. Onset detection is the process of identifying the points in time where a musical event (such as a guitar chord or drum hit) occurs in an audio signal.

In the context of guitar chugs, you can use onset detection to identify the points in time where the guitar chugs occur in a song, and then estimate the frequency range of the chugs based on the frequency content of the audio signal around the detected onsets.

Here's some sample Python code that demonstrates how to use Librosa to perform onset detection and estimate the frequency range of the guitar chugs:

python

import librosa
import numpy as np

# Load an audio file and resample it to 44.1 kHz
y, sr = librosa.load('path/to/audio/file', sr=44100)

# Compute the onset strength envelope using a spectral flux method
onset_env = librosa.onset.onset_strength(y, sr=sr, hop_length=512, aggregate=np.median)

# Detect the onset times using a peak picking method
onset_times = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=512)

# Estimate the frequency range of the guitar chugs
chug_freqs = []
for onset_time in onset_times:
    # Extract a short window of audio around the onset time
    onset_window = y[max(0, onset_time-512):onset_time+512]
    
    # Compute the power spectral density (PSD) of the window
    f, psd = signal.welch(onset_window, sr=sr, nperseg=512, scaling='spectrum')
    
    # Find the frequency bin with the maximum PSD
    max_psd_bin = np.argmax(psd)
    
    # Add the frequency of the maximum PSD bin to the list of chug frequencies
    chug_freqs.append(f[max_psd_bin])

# Estimate the frequency range of the chugs as the minimum and maximum chug frequencies
chug_fmin = np.min(chug_freqs)
chug_fmax = np.max(chug_freqs)

# Apply a bandpass filter to the audio signal using the estimated frequency range
y_filt = librosa.effects.bandpass(y, chug_fmin, chug_fmax, sr=sr)

This code loads an audio file using Librosa, computes the onset strength envelope using a spectral flux method, and detects the onset times using a peak picking method. For each detected onset, a short window of audio around the onset time is extracted, and the power spectral density (PSD) of the window is computed using the Welch method. The frequency of the bin with the maximum PSD is then added to a list of chug frequencies. Finally, the frequency range of the chugs is estimated as the minimum and maximum chug frequencies, and a bandpass filter is applied to the audio signal using the estimated frequency range.

Note that the specific parameters used for onset detection, PSD computation, and bandpass filtering may need to be adjusted depending on the characteristics of the guitar chugs you're trying to transcribe.