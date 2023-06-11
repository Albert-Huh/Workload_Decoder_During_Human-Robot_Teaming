from setup import Setup
from datetime import timedelta
import mne
import matplotlib.pyplot
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

# constants 
fs = 250 

# Create an instance of the Setup class
raw = Setup(raw_path="test_etat_data/eeg/USAARL Feb 2023 Shipping/From USAARL/Thu Feb 09 181227 CST 2023/EEG-Test_E0.85.DB.5C.86.C0/Thu Feb 09 181227 CST 2023.csv", 
                 montage_path=None, mode="USAARL E-tattoo", clock_correction=timedelta(days=56, hours=15, minutes=34))

# plot raw (debug)
fig = raw.raw.plot()
plt.show(block=False)
print(raw.raw.info)

# Call the get_events_from_nback_report function on the instance
nback_events = raw.get_events_from_nback_report(report_path="test_etat_data/eeg/USAARL Feb 2023 Shipping/From USAARL/Thu Feb 09 181227 CST 2023/EEG-Test_E0.85.DB.5C.86.C0/20230406_094815_S1001_2__Report.txt", fs=fs)

# Show event array (debug)
# print(nback_events)
# print(nback_events.shape)

# custom event dictionary
event_dict = {'0-back': 0, '1-back': 1, '2-back': 2}
fig = mne.viz.plot_events(nback_events, event_id=event_dict, sfreq=fs, first_samp=raw.raw.first_samp)
plt.show(block=True)
