import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mne
matplotlib.use('TkAgg')

plot_fig = False
# Load Raw Data
sample_data_path = os.path.join(
    os.getcwd(),
    'data/mne_sample_data', 
    'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(sample_data_path, verbose=False)
raw.crop(tmax=60).load_data()

# Method 1: Importing event records from stim channel STI 014
events = mne.find_events(raw, stim_channel='STI 014')

# Method 2: Importing event records from _eve.txt or _eve.fif files
sample_data_events_file = os.path.join(
    os.getcwd(),
    'data/mne_sample_data', 
    'sample_audvis_raw-eve.fif')
events_from_file = mne.read_events(sample_data_events_file)
assert np.array_equal(events, events_from_file[:len(events)])

event_dict = {'auditory/left': 1, 'auditory/right': 2, 'visual/left': 3, 'visual/right': 4, 'smiley': 5, 'buttonpress': 32}

if plot_fig == True:
    fig = mne.viz.plot_events(events, sfreq=raw.info['sfreq'], first_samp=raw.first_samp, event_id=event_dict)
    fig.subplots_adjust(right=0.7)
    raw.plot(events=events, start=5, duration=10, color='gray', event_color={1: 'r', 2: 'g', 3: 'b', 4: 'm', 5: 'y', 32: 'k'}, block=True)
