import os
from sqlite3 import Row
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import mne
matplotlib.use('TkAgg')

# read files
raw_path = os.path.join(os.getcwd(), 'data/raw_data', '053122_AR_BV10-20_OCTEST.vhdr')
montage_path = os.path.join(os.getcwd(), 'data/Workspaces_Montages/active electrodes/actiCAP for LiveAmp 32 Channel','CLA-32.bvef')
montage = mne.channels.read_custom_montage(montage_path)
raw = mne.io.read_raw_brainvision(raw_path)
raw.set_channel_types({'EOG':'eog'})
raw.set_montage(montage)
fig = raw.plot_sensors(show_names=True)

raw.plot(block=True)

# annot. to event
print(raw.info)
print(raw.annotations.onset)
print(raw.annotations.duration)
print(raw.annotations.description)

events_from_annot, event_dict = mne.events_from_annotations(raw)
print(event_dict)
print(events_from_annot)

custom_mapping = {'New Segment/': 5, 'Comment/eyeclosed': 1, 'Comment/eyeopen': 2}
(events_from_annot,
 event_dict) = mne.events_from_annotations(raw, event_id=custom_mapping)
print(event_dict)
print(events_from_annot)

# event visiualization
fig = mne.viz.plot_events(events_from_annot, event_id=event_dict, sfreq=raw.info['sfreq'], first_samp=raw.first_samp)


raw.load_data()

# data preprocessing
filt_raw = raw.copy().filter(l_freq=1, h_freq=None)
filt_raw = filt_raw.notch_filter(freqs=(60, 120, 180),method='spectrum_fit',filter_length='auto',phase='zero')
filt_raw = filt_raw.filter(None, 50, fir_design='firwin')

# preprocessed raw
filt_raw.plot(block=True)
filt_raw.plot_psd(fmax=250,average=True)

# eye open psd
filt_raw.copy().crop(0, 100).plot_psd(fmax=50,average=True)
# eye closed psd
filt_raw.copy().crop(140, 240).plot_psd(fmax=50,average=True)
eeg = filt_raw.copy().pick_types(eeg=True)

# TODO: plot psd with topomap
# TODO: plot spectrogram at the eye closing and opneing moments
# TODO: save plots

# create EOG epoch for correction
eog_epochs = mne.preprocessing.create_eog_epochs(filt_raw, ch_name='EOG', baseline=(-0.5, -0.3))
# eog_epochs = mne.preprocessing.create_eog_epochs(filt_raw, ch_name='EOG', tmin=-0.3,tmax=0.3, l_freq=1, h_freq=5, baseline=(-0.3, 0))
eog_epochs.plot_image(combine='mean')
# applying EOG baseline correction (mode: mean)
eog_epochs.average().plot_image()
# print(filt_raw.info)

# create evoked EOG from created epoch for correction
eog_evoked = mne.preprocessing.create_eog_epochs(filt_raw, ch_name='EOG',baseline=(-0.5, -0.2)).average()
# create evoked ECG from created epoch for correction
# setup ICA
ica = mne.preprocessing.ICA(n_components=8, random_state=60,max_iter= 'auto') #50n 70
# fit ICA to preconditioned raw data
ica.fit(filt_raw)
# load raw data into memory for artifact detection
filt_raw.load_data()
# manual inspection of IC

ica.plot_components()
ica.plot_sources(filt_raw)
plt.show()
ica.exclude = [0,1,2,3,4,5,6,7]
ica.plot_properties(filt_raw, picks=ica.exclude)
# Using EOG ch to select IC components for rejection
ica.exclude = [0],2
# find which ICs match the EOG pattern
eog_indices, eog_scores = ica.find_bads_eog(filt_raw, threshold=0.8)
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
'''

ica.apply(filt_raw)
# ica.plot_overlay(filt_raw, exclude=ica.exclude, picks=eeg)
filt_raw.plot(block=True)

closing = mne.Epochs(filt_raw, baseline = (118, 222))
opening = mne.Epochs(filt_raw, baseline = (238, 242))
freqs = np.logspace(*np.log10([6, 15]), num=8)
n_cycles = freqs / 2.  # different number of cycle per frequency
# TODO: define ec eo epochs


power, itc = mne.time_frequency.tfr_morlet(closing, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, decim=3, n_jobs=1)
power.plot(['Fp1'], baseline=(-0.5, 0), mode='logratio', title=power.ch_names['Fp1'])
# power.plot_joint(mode='mean',timefreqs=[(.5, 10), (1.3, 8)])
# # create a copy of raw
# orig_raw = raw.copy()
# # apply ICA to filt_raw
# ica.apply(raw)
# raw.filter(l_freq=0.1, h_freq=None)
# raw.filter(None, 50., fir_design='firwin')
