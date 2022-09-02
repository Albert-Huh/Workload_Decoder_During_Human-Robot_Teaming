import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from setup import Setup as setup
import preprocessing

raw_data_list = os.listdir(os.path.join(os.getcwd(), 'data/raw_data'))
for file_name in raw_data_list:
    if file_name.endswith('.fif') and file_name.startswith('070622_Dual'): #startswith('date_Dual') can isolate experiment session  ('Dual', 7, 11)
        raw_path_annot = os.path.join(os.path.join(os.getcwd(), 'data/raw_data'), file_name)
        montage_path = os.path.join(os.getcwd(), 'data/Workspaces_Montages/active electrodes/actiCAP for LiveAmp 32 Channel','CLA-32.bvef')
        raw_annot = setup(raw_path_annot, montage_path, mode='Binary')
        onset, duration, description = raw_annot.get_annotation_info()
        print(description)
        raw_path = os.path.join(os.path.join(os.getcwd(), 'data/raw_data'), file_name.replace('.fif','.vhdr'))
        raw = setup(raw_path, montage_path, mode='Dual')
        raw.set_annotation(raw.raw, onset=onset, duration=duration, description=description)
        # raw.get_brainvision_raw()
        # raw.get_e_tattoo_raw()
        # fig = raw.bv_raw.plot()
        # fig = raw.et_raw.plot()
        # plt.show()
        raw.get_brainvision_raw()
        raw.bv_raw.load_data()
        raw.bv_raw.set_eeg_reference('average')
        raw.get_e_tattoo_raw()
        raw.et_raw.load_data()
        raw.et_raw.set_eeg_reference(ref_channels=['A1', 'A2'])

        bv_filters = preprocessing.Filtering(raw.bv_raw, l_freq=1, h_freq=50)
        raw.bv_raw = bv_filters.external_artifact_rejection()
        et_filters = preprocessing.Filtering(raw.et_raw, l_freq=1, h_freq=50)
        raw.et_raw = et_filters.external_artifact_rejection()

        # raw.get_brainvision_raw()
        # raw.get_e_tattoo_raw()
        # fig = raw.bv_raw.plot()
        # fig = raw.et_raw.plot()
        # plt.show()
        # print(raw.raw.info['meas_date'])
        meas_date = str(raw.raw.info['meas_date'])
        recorder_meas_time = meas_date[0:4]+meas_date[5:7]+meas_date[8:10]+meas_date[11:19].replace(':','')
        report_list = os.listdir(os.path.join(os.getcwd(), 'data/reports'))
        for report_name in report_list:
            report_path = os.path.join(os.path.join(os.getcwd(), 'data/reports'), report_name)
            report_log_time = report_path.split('_',1)[1][0:15].replace('_', '')
            if abs(int(recorder_meas_time)-int(report_log_time)) < 60:
                resampled_freq = 200
                # print(report_path)
                nback_events = raw.get_events_from_nback_report(report_path=report_path, fs=resampled_freq)
        event_dict = {'0-back': 0, '1-back': 1, '2-back': 2}
        fig = mne.viz.plot_events(nback_events, event_id=event_dict, sfreq=resampled_freq, first_samp=raw.bv_raw.first_samp)
        
        bv_ica = preprocessing.Indepndent_Component_Analysis(raw.bv_raw, n_components=8)
        et_ica = preprocessing.Indepndent_Component_Analysis(raw.et_raw, n_components=4)

        bv_eog_evoked = bv_ica.create_physiological_evoked()
        et_eog_evoked = et_ica.create_physiological_evoked()

        bv_ica.perfrom_ICA()
        et_ica.perfrom_ICA()
        # fig = raw.bv_raw.plot()
        # fig = raw.et_raw.plot()
        # plt.show()
        # print(nback_event)
        # del raw.bv_raw, raw.et_raw
        # raw.bv_raw.load_data()
        # raw.et_raw.load_data()

        bv_theta = preprocessing.Filtering(raw.bv_raw, 4, 7)
        bv_alpha = preprocessing.Filtering(raw.bv_raw, 8, 13)
        bv_beta = preprocessing.Filtering(raw.bv_raw, 14, 30)
        bv_theta_raw = bv_theta.bandpass()
        bv_alpha_raw = bv_alpha.bandpass()
        bv_beta_raw = bv_beta.bandpass()
        fig = bv_theta_raw.plot()
        fig = bv_alpha_raw.plot()
        fig = bv_beta_raw.plot()
        plt.show()

        bv_theta_epochs = mne.Epochs(bv_theta_raw, events=nback_events, event_id=event_dict, tmin=-0.2, tmax=1.8, preload=True, picks='eeg')
        bv_alpha_epochs = mne.Epochs(bv_alpha_raw, events=nback_events, event_id=event_dict, tmin=-0.2, tmax=1.8, preload=True, picks='eeg')
        bv_beta_epochs = mne.Epochs(bv_beta_raw, events=nback_events, event_id=event_dict, tmin=-0.2, tmax=1.8, preload=True, picks='eeg')
        fig = bv_alpha_epochs['0-back'].plot_image(picks='eeg',combine='mean')
        fig = bv_alpha_epochs['1-back'].plot_image(picks='eeg',combine='mean')
        fig = bv_alpha_epochs['2-back'].plot_image(picks='eeg',combine='mean')
        fig = bv_theta_epochs['0-back'].plot_image(picks='eeg',combine='mean')
        fig = bv_theta_epochs['1-back'].plot_image(picks='eeg',combine='mean')
        fig = bv_theta_epochs['2-back'].plot_image(picks='eeg',combine='mean')


        # bv_epochs = mne.Epochs(raw=raw.bv_raw, events=nback_events, event_id=event_dict, tmin=-0.2, tmax=1.8, preload=True, picks='eeg')
        # et_epochs = mne.Epochs(raw=raw.et_raw, events=nback_events, event_id=event_dict, tmin=-0.2, tmax=1.8, preload=True, picks='eeg')
        # print(bv_epochs)
        # print(et_epochs)
        # zeroback_epochs = bv_epochs['0-back']
        # fig = bv_epochs['0-back'].plot_image(picks='eeg',combine='mean')
        # fig = bv_epochs['1-back'].plot_image(picks='eeg',combine='mean')
        # fig = bv_epochs['2-back'].plot_image(picks='eeg',combine='mean')
        # plt.show()
        break
        