import sys
import os
from sqlite3 import Row
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import mne
matplotlib.use('TkAgg')

# read files
raw_path = os.path.join(os.getcwd(), 'data/raw_data', '071322_Eye_OC_Test_MJ_AM_1.vhdr')
montage_path = os.path.join(os.getcwd(), 'data/Workspaces_Montages/active electrodes/actiCAP for LiveAmp 32 Channel','CLA-32.bvef')
raw = mne.io.read_raw_brainvision(raw_path)
raw.set_channel_types({'EOG':'eog'})
# raw.annotations.onset[:] = [0, 60.371, 120.59]
# raw.annotations.onset[:] = [0, 30.854, 59.998, 90, 119.9, 149.42, 180.9] # 063022_Eye_OC_Test_AR_AM_1_test.vhdr
print(raw.info)
# raw.plot(block=True)



# raw.set_eeg_reference(ref_channels=['A1', 'A2']) #channels are missing

bv_raw = raw.copy().pick_channels(['Fp1','Fp2','Fz','F3','F4','F7','F8','Cz','C3','C4','T7','T8','Pz','P3','P4','P7','P8','O1','O2','EOG'])
bv_raw.load_data()
bv_raw.set_eeg_reference('average')
# bv_raw.plot(block=True)

et_raw = raw.copy().pick_channels(['Fp1_ET','Fp2_ET','F7_ET','F8_ET','A1','A2','EOG'])
# use average of mastoid channels as reference
et_raw.load_data()
et_raw.set_eeg_reference(ref_channels=['A1', 'A2'])
# et_raw.plot(block=True)
new_names = dict(
    (ch_name,
     ch_name.replace('_ET', ''))
    for ch_name in et_raw.ch_names)
et_raw.rename_channels(new_names)

montage = mne.channels.read_custom_montage(montage_path)
bv_raw.set_montage(montage)
fig = bv_raw.plot_sensors(show_names=True)

# print(montage.get_positions())
et_montage = montage.copy()
fig = et_raw.plot_sensors(show_names=True)


# data preprocessing
bv_filt_raw = bv_raw.copy().filter(l_freq=1, h_freq=None, picks=['eeg','eog'] )
bv_filt_raw = bv_filt_raw.notch_filter(freqs=(60, 120, 180),method='spectrum_fit',filter_length='auto',phase='zero',picks=['eeg','eog'])
bv_filt_raw = bv_filt_raw.filter(None, 50, fir_design='firwin',picks=['eeg','eog'])

et_filt_raw = et_raw.copy().filter(l_freq=1, h_freq=None, picks=['eeg','eog'] )
et_filt_raw = et_filt_raw.notch_filter(freqs=(60, 120, 180),method='spectrum_fit',filter_length='auto',phase='zero',picks=['eeg','eog'])
et_filt_raw = et_filt_raw.filter(None, 50, fir_design='firwin',picks=['eeg','eog'])

# preprocessed raw
# filt_raw.set_eeg_reference(ref_channels=['A1', 'A2'])
bv_filt_raw.plot(block=True)
et_filt_raw.plot(block=True)

# # # eye open psd
# bv_filt_raw.copy().crop(79, 89).plot_psd(fmax=50,average=True)
# # # eye closed psd
# bv_filt_raw.copy().crop(91, 101).plot_psd(fmax=50,average=True)
# eeg = bv_filt_raw.copy().pick_types(eeg=True)

# TODO: pub quality figures
'''
# create EOG epoch for correction
eog_epochs = mne.preprocessing.create_eog_epochs(filt_raw, ch_name='EOG', baseline=(-0.5, -0.3))
# eog_epochs = mne.preprocessing.create_eog_epochs(filt_raw, ch_name='EOG', tmin=-0.3,tmax=0.3, l_freq=1, h_freq=5, baseline=(-0.3, 0))
eog_epochs.plot_image(combine='mean')
# applying EOG baseline correction (mode: mean)
eog_epochs.average().plot_image()
# print(filt_raw.info)
'''
# create evoked EOG from created epoch for correction
bv_eog_evoked = mne.preprocessing.create_eog_epochs(bv_filt_raw, ch_name='EOG', picks='eeg',baseline=(-0.5, -0.2)).average()
et_eog_evoked = mne.preprocessing.create_eog_epochs(et_filt_raw, ch_name='EOG', picks='eeg',baseline=(-0.5, -0.2)).average()

# create evoked ECG from created epoch for correction
# setup ICA
bv_ica = mne.preprocessing.ICA(n_components=8, random_state=50,max_iter= 'auto') #8n 80
et_ica = mne.preprocessing.ICA(n_components=4, random_state=30,max_iter= 'auto') #4n 60

# fit ICA to preconditioned raw data
bv_ica.fit(bv_filt_raw)
et_ica.fit(et_filt_raw)

# load raw data into memory for artifact detection
bv_filt_raw.load_data()
et_filt_raw.load_data()

# eog_epochs.average().plot_joint()
bv_eog_evoked.plot_joint()
bv_eog_evoked.plot_image()
et_eog_evoked.plot_joint()
et_eog_evoked.plot_image()

# ecg_evoked.plot_joint()
# sys.exit()
# manual inspection of IC
bv_ica.plot_components()
bv_ica.plot_sources(bv_filt_raw)
plt.show()
bv_ica.exclude = [0,1,2,3,4,5,6,7]
bv_ica.plot_properties(bv_filt_raw, picks=bv_ica.exclude)

et_ica.plot_components()
et_ica.plot_sources(et_filt_raw)
plt.show()
et_ica.exclude = [0,1,2,3]
et_ica.plot_properties(et_filt_raw, picks=et_ica.exclude)
# find which ICs match the EOG pattern
bv_eog_indices, bv_eog_scores = bv_ica.find_bads_eog(bv_filt_raw, threshold=0.9)
bv_ica.exclude = bv_eog_indices
et_eog_indices, et_eog_scores = et_ica.find_bads_eog(et_filt_raw, threshold=0.9)
et_ica.exclude = et_eog_indices

# barplot of ICA component "EOG match" scores
bv_ica.plot_scores(bv_eog_scores)
# plot diagnostics
bv_ica.plot_properties(bv_filt_raw, picks=bv_eog_indices)
# plot ICs applied to raw data, with EOG matches highlighted
bv_ica.plot_sources(bv_filt_raw)
plt.show()
# plot ICs applied to the averaged EOG epochs, with EOG matches highlighted
bv_ica.plot_sources(bv_eog_evoked)
plt.show()

# barplot of ICA component "EOG match" scores
et_ica.plot_scores(et_eog_scores)
# plot diagnostics
et_ica.plot_properties(et_filt_raw, picks=et_eog_indices)
# plot ICs applied to raw data, with EOG matches highlighted
et_ica.plot_sources(et_filt_raw)
plt.show()
# plot ICs applied to the averaged EOG epochs, with EOG matches highlighted
et_ica.plot_sources(et_eog_evoked)
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

bv_ica.apply(bv_filt_raw)
et_ica.apply(et_filt_raw)
# ica.plot_overlay(filt_raw, exclude=ica.exclude, picks=eeg)
bv_filt_raw.plot(block=True)
et_filt_raw.plot(block=True)

# create Epochs
# define rejection dictionary to reject abnormal pick-to-pick epoch later
# (There is a auto reject package I can use to automize this process)
# reject_criteria = dict(eeg=150e-6,       # 150 µV
#                        eog=250e-6)       # 250 µV


events_from_annot, event_dict = mne.events_from_annotations(raw)
print(event_dict)
print(events_from_annot)

custom_mapping = {'New Segment/': 5, 'Comment/Eye_Closed': 1, 'Comment/Eye_Open': 2}
(events_from_annot,
 event_dict) = mne.events_from_annotations(raw, event_id=custom_mapping)
print(event_dict)
print(events_from_annot)

# event visiualization
fig = mne.viz.plot_events(events_from_annot, event_id=event_dict, sfreq=raw.info['sfreq'], first_samp=raw.first_samp)


# create EEG epochs
bv_epochs = mne.Epochs(bv_filt_raw, events_from_annot, event_id=event_dict, tmin=-10.0, tmax=10.0, preload=True, picks=['eeg','eog'])
bv_epoch_eye_closed = bv_epochs['Comment/Eye_Closed']
bv_epoch_eye_open = bv_epochs['Comment/Eye_Open']

et_epochs = mne.Epochs(et_filt_raw, events_from_annot, event_id=event_dict, tmin=-10.0, tmax=10.0, preload=True, picks=['eeg','eog'])
et_epoch_eye_closed = et_epochs['Comment/Eye_Closed']
et_epoch_eye_open = et_epochs['Comment/Eye_Open']

freqs = np.logspace(*np.log10([1, 50]), num=160)
n_cycles = freqs / 2.  # different number of cycle per frequency
# cnorm = matplotlib.colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=3)
# cmap=cnorm
# baseline=(-5.0, -1.0)
power, itc = mne.time_frequency.tfr_morlet(bv_epoch_eye_closed, freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=True, decim=1, n_jobs=1, picks='eeg')
power.plot(baseline=(-10.0, -1.0), combine='mean', mode='logratio', title='BV Closing Epoch Average Power')
print(power.data)
print(np.nanmax(power.data))
fig, axis = plt.subplots(1, 2, figsize=(7, 4))
cnorm = matplotlib.colors.TwoSlopeNorm(vmin=-0.275, vcenter=0, vmax=0.275)
power.plot_topomap(ch_type='eeg', tmin=-10, tmax=-1, fmin=8, fmax=13, baseline=(-10.0, -1.0), mode='logratio', axes=axis[0], title='Eye-Open Alpha (8-13 Hz)', show=False)
power.plot_topomap(ch_type='eeg', tmin=1, tmax=10, fmin=8, fmax=13, baseline=(-10.0, -1.0), mode='logratio', axes=axis[1], title='Eye-Closed Alpha (8-13 Hz)', show=False)
mne.viz.tight_layout()
plt.show()

power, itc = mne.time_frequency.tfr_morlet(bv_epoch_eye_open, freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=True, decim=1, n_jobs=1, picks='eeg')
power.plot(baseline=(-10.0, -1.0), combine='mean', mode='logratio', title='BV Opening Epoch Average Power')
# power.plot([1], baseline=(-5.0, 0), mode='logratio', title=power.ch_names[1])
fig, axis = plt.subplots(1, 2, figsize=(7, 4))
cnorm = matplotlib.colors.TwoSlopeNorm(vmin=-0.225, vcenter=-0.1, vmax=0.025)
power.plot_topomap(ch_type='eeg', tmin=-10, tmax=-1, fmin=8, fmax=13, baseline=(-10.0, -1.0), mode='logratio', axes=axis[0], title='Eye-Closed Alpha (8-13 Hz)', show=False)
power.plot_topomap(ch_type='eeg', tmin=1, tmax=10, fmin=8, fmax=13, baseline=(-10.0, -1.0), mode='logratio', axes=axis[1], title='Eye-Open Alpha (8-13 Hz)', show=False)
mne.viz.tight_layout()
plt.show()

# closing_epoch
power, itc = mne.time_frequency.tfr_morlet(et_epoch_eye_closed, freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=True, decim=1, n_jobs=1, picks='eeg')
power.plot([2,3,4,5],baseline=(-10.0, -1.0), combine='mean', mode='logratio', title='ET Closing Epoch Average Power')

fig, axis = plt.subplots(1, 2, figsize=(7, 4))
cnorm = matplotlib.colors.TwoSlopeNorm(vmin=-0.2, vcenter=-0.05, vmax=0.1)
power.plot_topomap(ch_type='eeg', tmin=-10, tmax=-1, fmin=8, fmax=13, baseline=(-10.0, -1.0), mode='logratio', axes=axis[0], title='Eye-Open Alpha (8-13 Hz)', show=False)
power.plot_topomap(ch_type='eeg', tmin=1, tmax=10, fmin=8, fmax=13, baseline=(-10.0, -1.0), mode='logratio', axes=axis[1], title='Eye-Closed Alpha (8-13 Hz)', show=False)
mne.viz.tight_layout()
plt.show()

power, itc = mne.time_frequency.tfr_morlet(et_epoch_eye_open, freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=True, decim=1, n_jobs=1, picks='eeg')
power.plot([2,3,4,5],baseline=(-10.0, -1.0), combine='mean', mode='logratio', title='ET Opening Epoch Average Power')
# power.plot([3], baseline=(-5.0, 0), mode='logratio', title=power.ch_names[3])
fig, axis = plt.subplots(1, 2, figsize=(7, 4))
cnorm = matplotlib.colors.TwoSlopeNorm(vmin=-0.275, vcenter=-0.1875, vmax=-0.1)
power.plot_topomap(ch_type='eeg', tmin=-10, tmax=-1, fmin=8, fmax=13, baseline=(-10.0, -1.0), mode='logratio', axes=axis[0], title='Eye-Closed Alpha (8-13 Hz)', show=False)
power.plot_topomap(ch_type='eeg', tmin=1, tmax=10, fmin=8, fmax=13, baseline=(-10.0, -1.0), mode='logratio', axes=axis[1], title='Eye-Open Alpha (8-13 Hz)', show=False)
mne.viz.tight_layout()
plt.show()

'''
power, itc = mne.time_frequency.tfr_morlet(bv_epoch_eye_closed, freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=True, decim=1, n_jobs=1, picks='eeg')
power.plot(baseline=(-5.0, 0), combine='mean', mode='logratio', title='Average Power')
# power.plot([1], baseline=(-5.0, 0), mode='logratio', title=power.ch_names[1])
fig, axis = plt.subplots(1, 2, figsize=(7, 4))
power_open = power.copy().crop(tmin=-10, tmax=-2)
power_loc_open = [ i['loc'] for i in power_open.info['chs']]
power_closed = power.copy().crop(tmin=2, tmax=10)
power_loc_closed = [ i['loc'] for i in power_open.info['chs']]
power_loc = power_loc_open + power_loc_closed
print(power_loc)
vmax = np.nanmax(np.abs(power_loc))
vmin = -np.nanmin(np.abs(power_loc))
print(vmax, vmin)
power.plot_topomap(ch_type='eeg', tmin=-10, tmax=-2, fmin=8, fmax=13, baseline=(-5.0, 0), vmin=vmin, vmax=vmax, mode='logratio', axes=axis[0], title='Eye-Open Alpha (8-13 Hz)', show=False)
power.plot_topomap(ch_type='eeg', tmin=2, tmax=10, fmin=8, fmax=13, baseline=(-5.0, 0), vmin=vmin, vmax=vmax, mode='logratio', axes=axis[1], title='Eye-Closed Alpha (8-13 Hz)', show=False)
mne.viz.tight_layout()
plt.show()

power, itc = mne.time_frequency.tfr_morlet(bv_epoch_eye_open, freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=True, decim=1, n_jobs=1, picks='eeg')
power.plot(baseline=(-5.0, 0), combine='mean', mode='logratio', title='Average Power')
# power.plot([1], baseline=(-5.0, 0), mode='logratio', title=power.ch_names[1])
fig, axis = plt.subplots(1, 2, figsize=(7, 4))
power_open = power.copy().crop(tmin=2, tmax=10)
power_loc_open = [ i['loc'] for i in power_open.info['chs']]
power_closed = power.copy().crop(tmin=-10, tmax=-2)
power_loc_closed = [ i['loc'] for i in power_open.info['chs']]
power_loc = power_loc_open + power_loc_closed
print(power_loc)
vmax = np.nanmax(np.abs(power_loc))
vmin = -np.nanmin(np.abs(power_loc))
print(vmax, vmin)
power.plot_topomap(ch_type='eeg', tmin=-10, tmax=-2, fmin=8, fmax=13, baseline=(-5.0, 0), vmin=vmin, vmax=vmax, mode='logratio', axes=axis[0], title='Eye-Closed Alpha (8-13 Hz)', show=False)
power.plot_topomap(ch_type='eeg', tmin=2, tmax=10, fmin=8, fmax=13, baseline=(-5.0, 0), vmin=vmin, vmax=vmax, mode='logratio', axes=axis[1], title='Eye-Open Alpha (8-13 Hz)', show=False)
mne.viz.tight_layout()
plt.show()

# closing_epoch
power, itc = mne.time_frequency.tfr_morlet(et_epoch_eye_closed, freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=True, decim=1, n_jobs=1, picks='eeg')
power.plot([2,3,4,5],baseline=(-5.0, 0), combine='mean', mode='logratio', title='Average Power')

fig, axis = plt.subplots(1, 2, figsize=(7, 4))
power_open = power.copy().crop(tmin=-10, tmax=-2)
power_loc_open = [ i['loc'] for i in power_open.info['chs']]
power_closed = power.copy().crop(tmin=2, tmax=10)
power_loc_closed = [ i['loc'] for i in power_open.info['chs']]
power_loc = power_loc_open + power_loc_closed
print(power_loc)
vmax = np.nanmax(np.abs(power_loc))
vmin = -np.nanmin(np.abs(power_loc))
print(vmax, vmin)

power.plot_topomap(ch_type='eeg', tmin=-10, tmax=-2, fmin=8, fmax=13, baseline=(-5.0, 0), vmin=vmin, vmax=vmax, mode='logratio', axes=axis[0], title='Eye-Open Alpha (8-13 Hz)', show=False)
power.plot_topomap(ch_type='eeg', tmin=2, tmax=10, fmin=8, fmax=13, baseline=(-5.0, 0), vmin=vmin, vmax=vmax, mode='logratio', axes=axis[1], title='Eye-Closed Alpha (8-13 Hz)', show=False)
mne.viz.tight_layout()
plt.show()

power, itc = mne.time_frequency.tfr_morlet(et_epoch_eye_open, freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=True, decim=1, n_jobs=1, picks='eeg')
power.plot([2,3,4,5],baseline=(-5.0, 0), combine='mean', mode='logratio', title='Average Power')
# power.plot([3], baseline=(-5.0, 0), mode='logratio', title=power.ch_names[3])
fig, axis = plt.subplots(1, 2, figsize=(7, 4))
power_open = power.copy().crop(tmin=2, tmax=10)
power_loc_open = [ i['loc'] for i in power_open.info['chs']]
power_closed = power.copy().crop(tmin=-10, tmax=-2)
power_loc_closed = [ i['loc'] for i in power_open.info['chs']]
power_loc = power_loc_open + power_loc_closed
print(power_loc)
vmax = np.nanmax(np.abs(power_loc))
vmin = -np.nanmin(np.abs(power_loc))
print(vmax, vmin)

power.plot_topomap(ch_type='eeg', tmin=-10, tmax=-2, fmin=8, fmax=13, baseline=(-5.0, 0), vmin=vmin, vmax=vmax, mode='logratio', axes=axis[0], title='Eye-Closed Alpha (8-13 Hz)', show=False)
power.plot_topomap(ch_type='eeg', tmin=2, tmax=10, fmin=8, fmax=13, baseline=(-5.0, 0), vmin=vmin, vmax=vmax, mode='logratio', axes=axis[1], title='Eye-Open Alpha (8-13 Hz)', show=False)
mne.viz.tight_layout()
plt.show()
# power.plot_joint(mode='mean',timefreqs=[(.5, 10), (1.3, 8)])
# # create a copy of raw
# orig_raw = raw.copy()
# # apply ICA to filt_raw
# ica.apply(raw)
# raw.filter(l_freq=0.1, h_freq=None)
# raw.filter(None, 50., fir_design='firwin')
'''