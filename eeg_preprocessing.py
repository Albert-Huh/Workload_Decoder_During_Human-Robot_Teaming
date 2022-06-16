import os
from sqlite3 import Row
import numpy as np
import matplotlib
import mne
matplotlib.use('TkAgg')

raw_path = os.path.join(os.getcwd(), 'data/raw_data', '053122_AR_BV10-20_OCTEST.vhdr')
print(raw_path)
raw = mne.io.read_raw_brainvision(raw_path)
raw = raw.pick_types(eeg=True)
print(raw.info)
raw.plot(block=True)
# eeg = mne.pick_types(raw.info, eeg=True)
# raw.plot(order=eeg, block=True)

raw.load_data()
filt_raw = raw.copy().filter(l_freq=0.3, h_freq=None)
filt_raw = filt_raw.notch_filter(freqs=(60, 120, 180),method='spectrum_fit',filter_length='auto',phase='zero')
filt_raw = filt_raw.filter(None, 50, fir_design='firwin')
filt_raw.plot(block=True)
filt_raw.plot_psd(fmax=250,average=True)
