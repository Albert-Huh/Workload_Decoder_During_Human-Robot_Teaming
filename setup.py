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
        self.mode = mode # 'BV','BV10-20', 'ET', 'Dual'
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

    def 

# event visiualization
fig = mne.viz.plot_events(events_from_annot, event_id=event_dict, sfreq=raw.info['sfreq'], first_samp=raw.first_samp)