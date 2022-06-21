import os
from sqlite3 import Row
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import mne
matplotlib.use('TkAgg')

raw_path = os.path.join(os.getcwd(), 'data/raw_data', '053122_AR_BV10-20_OCTEST.vhdr')
montage_path = os.path.join(os.getcwd(), 'data/Workspaces_Montages/active electrodes/actiCAP for LiveAmp 32 Channel','CLA-32.bvef')
montage = mne.channels.read_custom_montage(montage_path)

raw = mne.io.read_raw_brainvision(raw_path)
raw.set_channel_types({'EOG':'eog'})
eeg = raw.copy().pick_types(eeg=True)
raw.set_montage(montage)
fig = raw.plot_sensors(show_names=True)


print(raw.info)
# raw.plot(block=True)
# raw.plot_sensors(block=True)
raw.load_data()
# tmin, tmax = 0, 100  # use the first 100s of data for training
# raw.crop(tmin, tmax).load_data()


filt_raw = raw.copy().filter(l_freq=0.1, h_freq=None)
filt_raw = filt_raw.notch_filter(freqs=(60, 120, 180),method='spectrum_fit',filter_length='auto',phase='zero')
filt_raw = filt_raw.filter(None, 50, fir_design='firwin')

filt_raw.plot(block=True)
filt_raw.plot_psd(fmax=250,average=True)
filt_raw.copy().crop(0, 100).plot_psd(fmax=50,average=True)
filt_raw.copy().crop(140, 240).plot_psd(fmax=50,average=True)



############### Physiologicraw.plot_sensors(block=True)al signal detection and correction via ICA ###############  
###############  Artifact detection
# create EOG epoch for correction
eog_epochs = mne.preprocessing.create_eog_epochs(filt_raw, ch_name='EOG', baseline=(-0.5, -0.3))
# eog_epochs = mne.preprocessing.create_eog_epochs(filt_raw, ch_name='EOG', tmin=-0.3,tmax=0.3, l_freq=1, h_freq=5, baseline=(-0.3, 0))
eog_epochs.plot_image(combine='mean')
# applying EOG baseline correction (mode: mean)
eog_epochs.average().plot_image()
print(filt_raw.info)
###############  Perform ICA and artifact correction
# create evoked EOG from created epoch for correction
eog_evoked = mne.preprocessing.create_eog_epochs(filt_raw, ch_name='EOG',baseline=(-0.5, -0.2)).average()
# create evoked ECG from created epoch for correction
# setup ICA
ica = mne.preprocessing.ICA(n_components=10, random_state=50,max_iter= 'auto') #50n 70
# fit ICA to preconditioned raw data
ica.fit(filt_raw)
# load raw data into memory for artifact detection
filt_raw.load_data()
# manual inspection of IC


ica.plot_components()
ica.plot_sources(filt_raw)
plt.show()
ica.exclude = [0, 5]
ica.plot_properties(filt_raw, picks=ica.exclude)
# Using EOG ch to select IC components for rejection
ica.exclude = []
# find which ICs match the EOG pattern
eog_indices, eog_scores = ica.find_bads_eog(filt_raw)
ica.exclude = eog_indices

# barplot of ICA component "EOG match" scores
ica.plot_scores(eog_scores)
# plot diagnostics
ica.plot_properties(filt_raw, picks=eog_indices)
# plot ICs applied to raw data, with EOG matches highlighted
ica.plot_sources(filt_raw)
plt.show()
# plot ICs applied to the averaged EOG epochs, with EOG matches highlighted
ica.plot_sources(eog_evoked)
plt.show()
'''
# find which ICs match the ECG pattern
ica.exclude = []
ecg_indices, ecg_scores = ica.find_bads_ecg(raw, method='correlation',
                                                threshold='auto')
ica.exclude = ecg_indices
if plot_fig == True:
    # barplot of ICA component "ECG match" scores
    ica.plot_scores(ecg_scores)
    # plot diagnostics
    ica.plot_properties(raw, picks=ecg_indices)
    # plot ICs applied to raw data, with ECG matches highlighted
    ica.plot_sources(raw)
    # plot ICs applied to the averaged ECG epochs, with ECG matches highlighted
    ica.plot_sources(ecg_evoked)
# redefind ica.exculde for EOG + ECG
ica.exclude = []
ica.exclude = eog_indices + ecg_indices
# blinks + heart beat artifact detection validation
if plot_fig == True:
    ica.plot_overlay(raw, exclude=ica.exclude, picks=eeg_channels)
# create a copy of raw
orig_raw = raw.copy()
# apply ICA to filt_raw
ica.apply(raw)
raw.filter(l_freq=0.1, h_freq=None)
raw.filter(None, 50., fir_design='firwin')
'''