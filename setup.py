import sys
import os
from sqlite3 import Row
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import mne
matplotlib.use('TkAgg')

class Setup:
    def __init__(self, raw_path=None, montage_path=None, mode=None):
        self.raw = mne.io.read_raw_brainvision(raw_path)
        self.montage = mne.channels.read_custom_montage(montage_path)
        self.mode = mode # 'BV', 'ET', 'Dual'
        self.raw.set_channel_types({'EOG':'eog'})
        self.raw.set_montage(self.montage)

    def get_annotation_info(self):
        onset = self.raw.annotations.onset
        duration = self.raw.annotations.duration
        description = self.raw.annotations.description
        return onset, duration, description

    def annotate_interactively(self):
        fig = self.raw.plot()
        fig.fake_keypress('a')
        interactive_annot = self.raw.annotations
        return interactive_annot

    def get_events_from_annot(self, custom_mapping='auto'):
        events, event_dict = mne.events_from_annotations(self.raw, event_id=custom_mapping)
        print(event_dict)
        print(events)
        return events, event_dict

    def get_events_from_raw(self, stim_channel=None):
        events = mne.find_events(raw, stim_channel=stim_channel)
        return events




# read files
raw_path = os.path.join(os.getcwd(), 'data/raw_data', '070622_Eye_OC_Test_CG_AM_1.vhdr')
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

custom_mapping = {'New Segment/': 5, 'Comment/Eye_Closed': 1, 'Comment/Eye_Open': 2}
(events_from_annot,
 event_dict) = mne.events_from_annotations(raw, event_id=custom_mapping)
print(event_dict)
print(events_from_annot)

# event visiualization
fig = mne.viz.plot_events(events_from_annot, event_id=event_dict, sfreq=raw.info['sfreq'], first_samp=raw.first_samp)