import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import timedelta
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

# Create annotations manually
my_annot = mne.Annotations(onset=[3, 5, 7],  # in seconds
                           duration=[1, 0.5, 0.25],  # in seconds, too
                           description=['AAA', 'BBB', 'CCC'])
print(my_annot)

# Setting annotations to raw
raw.set_annotations(my_annot)
print(raw.annotations)

# Convert meas_date (a turple of seconds, microseconds) into a float:
meas_date = raw.info['meas_date']
orig_time = raw.annotations.orig_time
# Notice that orig_time is None, because we haven’t specified it. In those cases, when you add the annotations to a Raw object, it is assumed that the orig_time matches the time of the first sample of the recording, so orig_time will be set to match the recording measurement date (raw.info['meas_date']).
print(meas_date == orig_time)

print(raw.first_samp)
time_of_first_sample = raw.first_samp / raw.info['sfreq']
print(my_annot.onset + time_of_first_sample)
print(raw.annotations.onset)

# If you know that your annotation onsets are relative to some other time, you can set orig_time before you call set_annotations(), and the onset times will get adjusted based on the time difference between your specified orig_time and raw.info['meas_date'], but without the additional adjustment for raw.first_samp.
time_format = '%Y-%m-%d %H:%M:%S.%f'
new_orig_time = (meas_date + timedelta(seconds=50)).strftime(time_format)

print(new_orig_time)

later_annot = mne.Annotations(onset=[3, 5, 7],
                              duration=[1, 0.5, 0.25],
                              description=['DDD', 'EEE', 'FFF'],
                              orig_time=new_orig_time)

raw2 = raw.copy().set_annotations(later_annot)
print(later_annot.onset)
print(raw2.annotations.onset)

fig = raw.plot(start=2, duration=6, block=True)
fig.fake_keypress('a')