import os
from sqlite3 import Row
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import mne
matplotlib.use('TkAgg')

# read files
raw_path = os.path.join(os.getcwd(), 'data/raw_data', '060122_AR_GPO7_OCTEST1.vhdr')
montage_path = os.path.join(os.getcwd(), 'data/Workspaces_Montages/active electrodes/actiCAP for LiveAmp 32 Channel','CLA-32.bvef')
montage = mne.channels.read_custom_montage(montage_path)
raw = mne.io.read_raw_brainvision(raw_path)
print(raw.info)
raw.set_channel_types({'EOG':'eog'})
raw.plot(block=True)
raw.set_montage(montage)
fig = raw.plot_sensors(show_names=True)

raw.plot(block=True)
# raw.set_eeg_reference(ref_channels=['A1', 'A2']) #channels are missing