from enum import auto
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.covariance import OAS
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

import mne
from mne.time_frequency import psd_welch

plot_fig = False

############### IMPORT DATA ############### 
# sample_data_folder = mne.datasets.sample.data_path('eeg_signal_processing\mne_tutorial\MNE-sample-data')

sample_data_raw_file = os.path.join('eeg_signal_processing\mne_tutorial\MNE-sample-data', 'MEG', 'sample',
                                    'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file)
# raw = mne.io.read_raw_brainvision(sample_data_raw_file,
#                                   eog='VEOGb',
#                                   misc='auto',
#                                   scale=1.0,
#                                   preload=False,
#                                   verbose=None)
print(raw.info)
tmin, tmax = 0, 250  # use the first 60s of data for training
raw.crop(tmin, tmax).load_data()
raw.info['bads'] = ['MEG 2443', 'EEG 053']  # bads + 2 more
eeg_channels = mne.pick_types(raw.info, eeg=True)
# print(eeg_channels)

############### SIGNAL PRECONDITIONING ############### 
############### Low frequency drift detection and filtering
print("debug HPF")
if plot_fig == True:
    # visually inspect low frequency drifts
    raw.plot(duration=60, order=eeg_channels, n_channels=len(eeg_channels),
            remove_dc=False)
    # design HPF design based on your inspection
    # design HPF with fc = 0.1, 0.2 and 0.3 Hz
    for cutoff in (0.1, 0.2, 0.3):
        raw_highpass = raw.copy().filter(l_freq=cutoff, h_freq=None)
        # plot filtered signal with two different designs
        fig = raw_highpass.plot(duration=60, order=eeg_channels, proj=False,
                                n_channels=len(eeg_channels), remove_dc=False)
        fig.subplots_adjust(top=0.9)
        fig.suptitle('High-pass filtered at {} Hz'.format(cutoff), size='xx-large',
                    weight='bold')
    # check the filter parameter
    filter_params = mne.filter.create_filter(raw.get_data(), raw.info['sfreq'],
                                            l_freq=0.3, h_freq=None)
    mne.viz.plot_filter(filter_params, raw.info['sfreq'], flim=(0.01, 5))
# appyl HPF  
raw_highpass = raw.copy().filter(l_freq=0.3, h_freq=None)

############### Powerline noise inspection and filtering
print("debug NF")
freqs = (60, 120, 180, 240) # harmonics of 60 Hz
if plot_fig == True:
    fig = raw.plot_psd(tmax=np.inf, fmax=250, average=True, picks = 'eeg')
    # add some arrows at 60 Hz and its harmonics:
    fig = raw_highpass.plot_psd(fmax=250, average=True, picks=eeg_channels)
    # compare NF effetiveness auto and spectrum fit method with given filter length
    raw_notch = raw.copy().notch_filter(freqs=freqs, picks=eeg_channels)
    for title, data in zip(['Un', 'Notch '], [raw, raw_notch]):
        fig = data.plot_psd(fmax=250, average=True, picks=eeg_channels)
        # fig.subplots_adjust(top=0.85)
        # fig.suptitle('{}filtered'.format(title), size='xx-large', weight='bold')
        # add_arrows(fig.axes[:2])
# apply NF
raw_notch_fit = raw_highpass.notch_filter(freqs=freqs, method='spectrum_fit', filter_length='auto', phase='zero')

############### High frequency noise rejection (resampling)
# low pass filtering below 50 Hz
print("debug LPF")
filt_raw = raw_notch_fit.filter(None, 50., fir_design='firwin')
if plot_fig == True:    
    for title, data in zip(['Un', 'spectrum_fit '], [raw, filt_raw]):
        fig = data.plot_psd(fmax=250, average=True, picks=eeg_channels)
        fig.subplots_adjust(top=0.85)
        fig.suptitle('{}filtered'.format(title), size='xx-large', weight='bold')

############### Physiological signal detection and correction via ICA ###############  
###############  Artifact detection
manual_inspection = False
if plot_fig == True:
    if manual_inspection == True:
        # create ECG epoch for correction (no ECG channel for our case)
        print("debug ECG")
        ecg_epochs = mne.preprocessing.create_ecg_epochs(raw, picks=eeg_channels)
        ecg_epochs.plot_image(combine='mean')
        # applying ECG baseline correction (mode: mean)
        avg_ecg_epochs = ecg_epochs.average().apply_baseline((-0.5, -0.2))
        avg_ecg_epochs.plot_topomap(times=np.linspace(-0.05, 0.05, 11))
        avg_ecg_epochs.plot_joint(times=[-0.25, -0.025, 0, 0.025, 0.25])
        # create EOG epoch for correction
        print("debug EOG")
        eog_epochs = mne.preprocessing.create_eog_epochs(raw, baseline=(-0.5, -0.2), picks=eeg_channels)
        eog_epochs.plot_image(combine='mean')
        # applying EOG baseline correction (mode: mean)
        eog_epochs.average().plot_joint()

###############  Perform ICA and artifact correction
print('debug ICA')
# create evoked EOG from created epoch for correction
eog_evoked = mne.preprocessing.create_eog_epochs(raw, picks=eeg_channels,baseline=(-0.5, -0.2)).average()
# create evoked ECG from created epoch for correction
ecg_evoked = mne.preprocessing.create_ecg_epochs(raw, picks=eeg_channels,baseline=(-0.5, -0.2)).average()
if plot_fig == True:
    eog_evoked.plot_joint()
    ecg_evoked.plot_joint()
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

############### Create Epochs
# define rejection dictionary to reject abnormal pick-to-pick epoch later
# (There is a auto reject package I can use to automize this process)
reject_criteria = dict(eeg=150e-6,       # 150 µV
                       eog=250e-6)       # 250 µV
# create training and test data
tmin, tmax = 0, 200  # use the first 200s of data for training
raw_train = raw.copy().crop(tmin, tmax).load_data()
tmin, tmax = 200, 250  # use the last 50s of data for testing
raw_test = raw.copy().crop(tmin, tmax).load_data()
# create EEG epochs
epochs_train = mne.Epochs(raw_train, events, event_id=event_dict, tmin=-0.2, tmax=0.5,
                    reject=reject_criteria, preload=True, picks=['eeg','eog'])
epochs_test = mne.Epochs(raw_test, events, event_id=event_dict, tmin=-0.2, tmax=0.5,
                    reject=reject_criteria, preload=True, picks=['eeg','eog'])
# equalize event counts to avoid biasing
conds_we_care_about = ['auditory/left', 'auditory/right',
                       'visual/left', 'visual/right']
epochs_train.equalize_event_counts(conds_we_care_about)  # this operates in-place
epochs_train.pick_types(eeg=True)
epochs_train = epochs_train['auditory','visual']
aud_epochs_train = epochs_train['auditory']
vis_epochs_train = epochs_train['visual']
epochs_test.equalize_event_counts(conds_we_care_about)  # this operates in-place
epochs_test.pick_types(eeg=True)
epochs_test = epochs_test['auditory','visual']
aud_epochs_test = epochs_test['auditory']
vis_epochs_test = epochs_test['visual']
del raw, raw_train, raw_test # free up memory
# check some epochs
if plot_fig == True:
    fig = aud_epochs_train.plot(events=events)
    plt.show()
    fig = aud_epochs_test.plot(events=events)
    plt.show()

############### Epochs analysis
print('debug epochs analysis')
aud = aud_epochs_train['auditory'].average()
vis = vis_epochs_train['visual'].average()
if plot_fig == True:
    for evk in (aud, vis):
        # global field power and spatial plot
        evk.plot(gfp=True, spatial_colors=True, ylim=dict(eeg=[-12, 12]))
        # spatial plot + topomap
        evk.plot_joint()
    evokeds = dict(auditory=list(aud_epochs_train.iter_evoked()),
                visual=list(vis_epochs_train.iter_evoked()))
    mne.viz.plot_compare_evokeds(evokeds, combine='mean', picks='eeg')
# Specify event codes of interest with descriptive labels
epochs_train.events = mne.merge_events(epochs_train.events, [1, 2], 12, replace_events=True)
epochs_train.events = mne.merge_events(epochs_train.events, [3, 4], 34, replace_events=True)
epochs_test.events = mne.merge_events(epochs_test.events, [1, 2], 12, replace_events=True)
epochs_test.events = mne.merge_events(epochs_test.events, [3, 4], 34, replace_events=True)

############### FEATURE ENGINEERING ###############
###############  Featrue extraction
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
    FREQ_BANDS = {"delta": [0.5, 4.5],
                  "theta": [4.5, 8.5],
                  "alpha": [8.5, 11.5],
                  "beta": [11.5, 30],
                  "gamma": [30, 45]}

    psds, freqs = psd_welch(epochs, picks='eeg', fmin=0.5, fmax=45., )
    # Normalize the PSDs
    psds /= np.sum(psds, axis=-1, keepdims=True)

    X = []
    for fmin, fmax in FREQ_BANDS.values():
        psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
        X.append(psds_band.reshape(len(psds), -1))

    return np.concatenate(X, axis=1)
X_train = eeg_power_band(epochs_train)
# Train
y_train = epochs_train.events[:,2]
# print(y_train)
# print(epochs_train.events)

############### CLASSIFICATION ###############
###############  Random forest classification
pipe_RF = make_pipeline(FunctionTransformer(eeg_power_band, validate=False),
                     RandomForestClassifier(n_estimators=100, random_state=42))
pipe_RF.fit(epochs_train, y_train)
# Test
y_pred = pipe_RF.predict(epochs_test)
# Assess the results
y_test = epochs_test.events[:,2]
acc = accuracy_score(y_test, y_pred)
print('Random Forest Classifier')
print('Accuracy score: {}'.format(acc))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
event_id = {'Auditory': 12,
            'Visual': 34}
print('Classification Report:')
print(classification_report(y_test, y_pred, target_names=event_id.keys()))

###############  K-nearest neighbors classification
pipe_KNN = make_pipeline(FunctionTransformer(eeg_power_band, validate=False),
                     KNeighborsClassifier(3))
pipe_KNN.fit(epochs_train, y_train)
# Test
y_pred = pipe_KNN.predict(epochs_test)
# Assess the results
y_test = epochs_test.events[:,2]
acc = accuracy_score(y_test, y_pred)
print('KNN Classifier')
print('Accuracy score: {}'.format(acc))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
event_id = {'Auditory': 12,
            'Visual': 34}
print('Classification Report:')
print(classification_report(y_test, y_pred, target_names=event_id.keys()))

###############  Linear discriminant analysis
oa = OAS(store_precision=False, assume_centered=False)
pipe_LDA = make_pipeline(FunctionTransformer(eeg_power_band, validate=False),
                     LinearDiscriminantAnalysis(solver="lsqr", covariance_estimator=oa))
pipe_LDA.fit(epochs_train, y_train)
# Test
y_pred = pipe_LDA.predict(epochs_test)
# Assess the results
y_test = epochs_test.events[:,2]
acc = accuracy_score(y_test, y_pred)
print('Linear Discriminant Analysis')
print('Accuracy score: {}'.format(acc))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
event_id = {'Auditory': 12,
            'Visual': 34}
print('Classification Report:')
print(classification_report(y_test, y_pred, target_names=event_id.keys()))

###############  Gaussian process classifier
pipe_GP = make_pipeline(FunctionTransformer(eeg_power_band, validate=False),
                     GaussianProcessClassifier(1.0 * RBF(1.0)))
pipe_GP.fit(epochs_train, y_train)
# Test
y_pred = pipe_GP.predict(epochs_test)
# Assess the results
y_test = epochs_test.events[:,2]
acc = accuracy_score(y_test, y_pred)
print('Gaussian Process Classifier')
print('Accuracy score: {}'.format(acc))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
event_id = {'Auditory': 12,
            'Visual': 34}
print('Classification Report:')
print(classification_report(y_test, y_pred, target_names=event_id.keys()))

###############  Multi-layer perceptron classifier
pipe_MLP = make_pipeline(FunctionTransformer(eeg_power_band, validate=False),
                     MLPClassifier(alpha=0.0001, max_iter=10000))
pipe_MLP.fit(epochs_train, y_train)
# Test
y_pred = pipe_MLP.predict(epochs_test)
# Assess the results
y_test = epochs_test.events[:,2]
acc = accuracy_score(y_test, y_pred)
print('MLP Classifier')
print('Accuracy score: {}'.format(acc))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
event_id = {'Auditory': 12,
            'Visual': 34}
print('Classification Report:')
print(classification_report(y_test, y_pred, target_names=event_id.keys()))