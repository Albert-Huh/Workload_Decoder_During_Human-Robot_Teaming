import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from setup import Setup as setup
import preprocessing

raw_data_list = os.listdir(os.path.join(os.getcwd(), 'data/raw_data'))
for file_name in raw_data_list:
    if file_name.endswith('.fif') and file_name.startswith('Dual', 7, 11):
        print('flag')
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
        filters = preprocessing.Filtering(raw.raw, l_freq=1, h_freq=50)
        raw.raw = filters.external_artifact_rejection()
        raw.get_brainvision_raw()
        raw.get_e_tattoo_raw()
        # fig = raw.bv_raw.plot()
        # fig = raw.et_raw.plot()
        # plt.show()
        print(raw.raw.info['meas_date'])
        meas_date = str(raw.raw.info['meas_date'])
        recorder_meas_time = meas_date[0:4]+meas_date[5:7]+meas_date[8:10]+meas_date[11:19].replace(':','')
        report_list = os.listdir(os.path.join(os.getcwd(), 'data/reports'))
        for report_name in report_list:
            report_path = os.path.join(os.path.join(os.getcwd(), 'data/reports'), report_name)
            report_log_time = report_path.split('_',1)[1][0:15].replace('_', '')
            if abs(int(recorder_meas_time)-int(report_log_time)) < 60:
                # print(report_path)
                nback_event = raw.get_events_from_nback_report(report_path=report_path)
        event_dict = {'0-back': 0, '1-back': 1, '2-back': 2}
        # fig = mne.viz.plot_events(nback_event, event_id=event_dict, sfreq=raw.raw.info['sfreq'], first_samp=raw.raw.first_samp)
        
        '''
        bv_ica = preprocessing.Indepndent_Component_Analysis(raw.bv_raw, n_components=8)
        et_ica = preprocessing.Indepndent_Component_Analysis(raw.et_raw, n_components=4)

        bv_eog_evoked = bv_ica.create_physiological_evoked()
        et_eog_evoked = et_ica.create_physiological_evoked()

        raw.bv_raw = bv_ica.perfrom_ICA()
        raw.et_raw = et_ica.perfrom_ICA()
        fig = raw.bv_raw.plot()
        fig = raw.et_raw.plot()
        plt.show()
        '''
        
        bv_epochs = mne.Epochs(raw.bv_raw, nback_event, event_id=event_dict, tmin=0, tmax=2.0, preload=True, picks=['eeg','eog'])
        et_epochs = mne.Epochs(raw.et_raw, nback_event, event_id=event_dict, tmin=0, tmax=2.0, preload=True, picks=['eeg','eog'])
        break
        