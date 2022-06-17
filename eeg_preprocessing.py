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

############### Physiological signal detection and correction via ICA ###############  
###############  Artifact detection
# create EOG epoch for correction
print("debug EOG")
eog_epochs = mne.preprocessing.create_eog_epochs(filt_raw, baseline=(-0.5, -0.2), picks=eeg_channels)
eog_epochs.plot_image(combine='mean')
# applying EOG baseline correction (mode: mean)
eog_epochs.average().plot_joint()

###############  Perform ICA and artifact correction
# create evoked EOG from created epoch for correction
eog_evoked = mne.preprocessing.create_eog_epochs(raw, ch_name='eog',baseline=(-0.5, -0.2)).average()
# create evoked ECG from created epoch for correction
# setup ICA
ica = mne.preprocessing.ICA(n_components=40, random_state=70,max_iter= 'auto') #50n 70
# fit ICA to preconditioned raw data
ica.fit(filt_raw, picks=eeg_channels)
# load raw data into memory for artifact detection
raw.load_data()
# manual inspection of IC
manual_inspection = False
if plot_fig == True:
    if manual_inspection == True:
        ica.plot_components()
        ica.plot_sources(raw)
        plt.show()
        ica.exclude = [0, 5]
        ica.plot_properties(raw, picks=ica.exclude)
# Using EOG ch to select IC components for rejection
ica.exclude = []
# find which ICs match the EOG pattern
eog_indices, eog_scores = ica.find_bads_eog(raw)
ica.exclude = eog_indices
if plot_fig == True:
    # barplot of ICA component "EOG match" scores
    ica.plot_scores(eog_scores)
    # plot diagnostics
    ica.plot_properties(raw, picks=eog_indices)
    # plot ICs applied to raw data, with EOG matches highlighted
    ica.plot_sources(raw)
    plt.show()
    # plot ICs applied to the averaged EOG epochs, with EOG matches highlighted
    ica.plot_sources(eog_evoked)
    plt.show()
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
# show some frontal channels to clearly illustrate the artifact removal
if plot_fig == True:
    artifact_picks = ['EEG 001', 'EEG 002', 'EEG 003', 'EEG 004', 'EEG 005', 'EEG 006',
        'EEG 007', 'EEG 008']
    chan_idxs = [raw.ch_names.index(ch) for ch in artifact_picks]
    orig_raw.plot(order=chan_idxs, n_channels=len(chan_idxs),duration=20)
    plt.show()
    raw.plot(order=chan_idxs, n_channels=len(chan_idxs),duration=20)
    plt.show()
    # raw.plot(duration=60, order=eeg_channels, n_channels=len(eeg_channels),
    #         remove_dc=False)
    # plt.show()
del orig_raw, raw_highpass, raw_notch_fit

############### EPOCHING RAW DATA ###############
print('debug epoching')
############### Detecting experimental events
events = mne.find_events(raw, stim_channel='STI 014')
# print(events[:5])  # show the first 5
event_dict = {'auditory/left': 1, 'auditory/right': 2, 'visual/left': 3,
              'visual/right': 4, 'smiley': 5, 'buttonpress': 32}
# event visiualization
if plot_fig == True:
    fig = mne.viz.plot_events(events, event_id=event_dict, sfreq=raw.info['sfreq'],
                first_samp=raw.first_samp)