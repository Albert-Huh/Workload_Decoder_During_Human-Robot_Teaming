import os
import mne
import numpy as np

def eeg_power_band(epochs):
    """EEG relative power band feature extraction.

    This function takes an ``mne.Epochs`` object and creates EEG features based
    on relative power in specific frequency bands that are compatible with
    scikit-learn.

    Parameters
    ----------
    epochs : Epochs
        The data.

    Returns
    -------
    X : numpy array of shape [n_samples, 5]
        Transformed data.
    """
    # specific frequency bands
    FREQ_BANDS = {"delta": [0.5, 5.0],
                  "theta": [5.0, 8.0],
                  "alpha": [8.0, 13.0],
                  "sigma": [13.0, 16.0],
                  "beta": [16.0, 30.0]}

    psds, freqs = mne.time_frequency.psd_welch(epochs, picks='eeg', fmin=0.5, fmax=30)
    
    # Normalize the PSDs
    psds /= np.sum(psds, axis=-1, keepdims=True)

    X = []
    for fmin, fmax in FREQ_BANDS.values():
        psds_band = psds[:,:, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
        X.append(psds_band.reshape(len(psds), -1))
    
    return np.concatenate(X, axis=1)