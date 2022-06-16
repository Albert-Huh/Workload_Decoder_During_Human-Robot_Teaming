import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import Read_Data
import pandas as pd
import singnal_processing_toolbox as stb

"""
This module is to perform signal processing of the raw EEG signals from the DAQ device
"""
plot = True
# Import data
data = Read_Data.GUI.get_row_data()
eeg = data.drop(['x_dir', 'y_dir', 'z_dir', 'Time', 'Time Stamp'], axis = 1)
sample = eeg.CP5.to_numpy()

# TODO Initialize from a measurement class


# TODO Signal detrending

fs = 1000.0 # Sampling freq (Hz)

# Design notch filter
f0 = 60.0 # Freq to be removed from signal (Hz)
Q = 30.0 # Quality factor
bn, an  = signal.iirnotch(f0, Q, fs)
taps = signal.firwin(9, 0.5/(fs/2), window=('kaiser', 0.5), pass_zero=False)
bl, al = signal.butter(10, 40/(fs/2), 'low')

# w, h = freqz(taps, worN=4000)
# Notch filter frequency response
freq, h = signal.freqz(bn, an, fs=fs)
if plot:
    fig1, ax = plt.subplots(2, 1, figsize=(8, 6))
    ax[0].plot(freq, 20*np.log10(abs(h)), color='blue')
    ax[0].set_title('Frequency Response')
    ax[0].set_ylabel('Amplitude (dB)', color='blue')
    ax[0].set_xlim([0, 100])
    ax[0].set_ylim([-25, 10])
    ax[0].grid()
    ax[1].plot(freq, np.unwrap(np.angle(h))*180/np.pi, color='green')
    ax[1].set_ylabel('Angle (degrees)', color='green')
    ax[1].set_xlabel('Frequency (Hz)')
    ax[1].set_xlim([0, 100])
    ax[1].set_yticks([-90, -60, -30, 0, 30, 60, 90])
    ax[1].set_ylim([-90, 90])
    ax[1].grid()
    plt.show()

# Apply filter
feeg = eeg.apply(stb.lfilt_function,args=(bn,an), axis=0) # notch filter at 60 hz
feeg = feeg.apply(stb.lfilt_function,args=(taps,1), axis=0) # high-pass filter at 0.5 hz
feeg = feeg.apply(stb.lfilt_function,args=(bl,al), axis=0) # high-pass filter at 70 hz
sample2 = feeg.CP5.to_numpy()
if plot:
    feeg.plot(subplots=True, figsize=(10, 6))
    fig2, ax = plt.subplots(2, 1, figsize=(8, 6))
    # ax[0].plot(eeg.Fp1.to_numpy(), color='blue')
    ax[0].plot(sample[3000:4000], color='blue')
    ax[0].set_title('CP5')
    ax[0].set_ylabel('Amplitude (V)', color='blue')
    ax[0].grid()
    ax[1].plot(sample2[3000:4000], color='green')
    # ax[1].plot(feeg.Fp1.to_numpy(), color='green')
    ax[1].set_ylabel('Amplitude (V)', color='green')
    ax[1].set_xlabel('Time (ms)')
    ax[1].grid()
    plt.show()
if plot:
    fig3, ax = plt.subplots(2, 1, figsize=(8, 6))
    ax[0].plot(eeg.CP5.to_numpy(), color='blue')
    ax[0].set_title('CP5')
    ax[0].set_ylabel('Amplitude (V)', color='blue')
    ax[0].grid()
    ax[1].plot(feeg.CP5.to_numpy(), color='green')
    ax[1].set_ylabel('Amplitude (V)', color='green')
    ax[1].set_xlabel('Time (ms)')
    ax[1].grid()
    plt.show()
# TODO EOG rejection


# TODO FIR bandspass filter