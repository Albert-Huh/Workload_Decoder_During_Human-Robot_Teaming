import sys
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
print(raw.info['nchan'])
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
filt_raw = raw.copy().filter(l_freq=1, h_freq=None, picks=['eeg','eog'] )
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

# TODO: pub quality figures

# create EOG epoch for correction
eog_epochs = mne.preprocessing.create_eog_epochs(filt_raw, ch_name='EOG', baseline=(-0.5, -0.3))
# eog_epochs = mne.preprocessing.create_eog_epochs(filt_raw, ch_name='EOG', tmin=-0.3,tmax=0.3, l_freq=1, h_freq=5, baseline=(-0.3, 0))
eog_epochs.plot_image(combine='mean')
# applying EOG baseline correction (mode: mean)
eog_epochs.average().plot_image()
# print(filt_raw.info)

# create evoked EOG from created epoch for correction
eog_evoked = mne.preprocessing.create_eog_epochs(filt_raw, ch_name='EOG', picks='eeg',baseline=(-0.5, -0.2)).average()
# create evoked ECG from created epoch for correction
# setup ICA
ica = mne.preprocessing.ICA(n_components=8, random_state=60,max_iter= 'auto') #50n 70
# fit ICA to preconditioned raw data
ica.fit(filt_raw)
# load raw data into memory for artifact detection
filt_raw.load_data()

# create evoked EOG from created epoch for correction
eog_evoked = mne.preprocessing.create_eog_epochs(filt_raw, baseline=(-0.5, -0.2)).average()
# create evoked ECG from created epoch for correction
# ecg_evoked = mne.preprocessing.create_ecg_epochs(filt_raw, picks='eeg',baseline=(-0.5, -0.2)).average() # cannot creat ecg_evoked without ECG ch. Virtual ch only can be constructed using MEG
eog_epochs.average().plot_joint()
eog_evoked.plot_joint()
eog_evoked.plot_image()
# ecg_evoked.plot_joint()
sys.exit()
# manual inspection of IC
ica.plot_components()
ica.plot_sources(filt_raw)
plt.show()
ica.exclude = [0,1,2,3,4,5,6,7]
ica.plot_properties(filt_raw, picks=ica.exclude)
# Using EOG ch to select IC components for rejection
ica.exclude = [0]
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

# create Epochs
# define rejection dictionary to reject abnormal pick-to-pick epoch later
# (There is a auto reject package I can use to automize this process)
# reject_criteria = dict(eeg=150e-6,       # 150 µV
#                        eog=250e-6)       # 250 µV
# create EEG epochs
epochs = mne.Epochs(filt_raw, events_from_annot, event_id=event_dict, tmin=-10.0, tmax=10.0, preload=True, picks=['eeg','eog'])
epoch_eye_closed = epochs['Comment/eyeclosed']
epoch_eye_open = epochs['Comment/eyeopen']

freqs = np.logspace(*np.log10([5, 18]), num=50)
n_cycles = freqs / 2.  # different number of cycle per frequency
power, itc = mne.time_frequency.tfr_morlet(epoch_eye_closed, freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=True, decim=1, n_jobs=1, picks='eeg')
print(power)
power.plot([0], baseline=(-5.0, 0), mode='logratio', title=power.ch_names[0])
power.plot([10], baseline=(-5.0, 0), mode='logratio', title=power.ch_names[10])
fig, axis = plt.subplots(1, 2, figsize=(7, 4))
power.plot_topomap(ch_type='eeg', tmin=-10, tmax=-2, fmin=8, fmax=13, baseline=(-5.0, 0), mode='logratio', axes=axis[0], title='Eye-Open Alpha (8-13 Hz)', show=False)
power.plot_topomap(ch_type='eeg', tmin=2, tmax=10, fmin=8, fmax=13, baseline=(-5.0, 0), mode='logratio', axes=axis[1], title='Eye-Closed Alpha (8-13 Hz)', show=False)
mne.viz.tight_layout()
plt.show()

power, itc = mne.time_frequency.tfr_morlet(epoch_eye_open, freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=True, decim=1, n_jobs=1, picks='eeg')
print(power)
power.plot([0], baseline=(-5.0, 0), mode='logratio', title=power.ch_names[0])
power.plot([10], baseline=(-5.0, 0), mode='logratio', title=power.ch_names[10])
fig, axis = plt.subplots(1, 2, figsize=(7, 4))
power.plot_topomap(ch_type='eeg', tmin=-10, tmax=-2, fmin=8, fmax=13, baseline=(-5.0, 0), mode='logratio', axes=axis[0], title='Eye-Closed Alpha (8-13 Hz)', show=False)
power.plot_topomap(ch_type='eeg', tmin=2, tmax=10, fmin=8, fmax=13, baseline=(-5.0, 0), mode='logratio', axes=axis[1], title='Eye-Open Alpha (8-13 Hz)', show=False)
mne.viz.tight_layout()
plt.show()
# power.plot_joint(mode='mean',timefreqs=[(.5, 10), (1.3, 8)])
# # create a copy of raw
# orig_raw = raw.copy()
# # apply ICA to filt_raw
# ica.apply(raw)
# raw.filter(l_freq=0.1, h_freq=None)
# raw.filter(None, 50., fir_design='firwin')

