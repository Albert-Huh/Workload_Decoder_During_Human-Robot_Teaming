import os
from sqlite3 import Row
from xmlrpc.client import _Method
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import mne
matplotlib.use('TkAgg')

class Filtering:
    def __init__(self, raw):
        # filter parameters
        self.raw = raw
        self.sfreq = raw.info['sfreq']
        self.picks = ['eeg', 'eog']

    def highpass(self, cutoff_freq, filt_type='fir', iir_params=None, show_filt_params = False):
        filt_raw = self.raw.filter(
            l_freq=cutoff_freq, h_freq=None,
            picks=self.picks, method=filt_type,
            iir_params=iir_params)
        if show_filt_params == True:
            # check the filter parameter
            filter_params = mne.filter.create_filter(
                data=self.raw.get_data(), sfreq=self.sfreq,
                l_freq=cutoff_freq, h_freq=None,
                method=filt_type, iir_params=iir_params)
            mne.viz.plot_filter(h=filter_params, 
                sfreq=self.sfreq, flim=(0.01, self.sfreq/2))
        return filt_raw

    def lowpass(self, cutoff_freq, filt_type='fir', iir_params=None, show_filt_params = False):
        filt_raw = self.raw.filter(
            l_freq=None, h_freq=cutoff_freq,
            picks=self.picks, method=filt_type,
            iir_params=iir_params)
        if show_filt_params == True:
            # check the filter parameter
            filter_params = mne.filter.create_filter(
                data=self.raw.get_data(), sfreq=self.sfreq,
                l_freq=cutoff_freq, h_freq=None,
                method=filt_type, iir_params=iir_params)
            mne.viz.plot_filter(h=filter_params, 
                sfreq=self.sfreq, flim=(0.01, self.sfreq/2))
        return filt_raw

    def notch(self, target_freqs=60, notch_wdith=None, filt_type='fir', iir_params=None):
        filt_raw = self.raw.notch_filter(
            freqs=target_freqs, picks=self.picks,
            method=filt_type, iir_params=iir_params)
        return filt_raw

    def resample(self):
        pass

    
    pass

