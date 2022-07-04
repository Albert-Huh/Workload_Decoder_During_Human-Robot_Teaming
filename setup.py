import sys
import os
from sqlite3 import Row
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import mne
matplotlib.use('TkAgg')

class Setup:
    def __init__(self, raw_path, montage_path):
        pass
        self.raw = mne.io.read_raw_brainvision(raw_path)
        self.montage = mne.channels.read_custom_montage(montage_path)

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
