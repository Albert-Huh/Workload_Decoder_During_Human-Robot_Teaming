import os
import mne
import matplotlib
matplotlib.use('TkAgg')

sample_data_folder = os.path.join(
    os.getcwd(),
    'data/mne_sample_data')
sample_data_raw_file = os.path.join(
    sample_data_folder, 
    'sample_audvis_raw.fif')

raw = mne.io.read_raw_fif(sample_data_raw_file, verbose=False)
raw.crop(tmax=60).load_data()
raw.pick(['EEG 0{:02}'.format(n) for n in range(41, 60)])

# code lines below are commented out because the sample data doesn't have
# earlobe or mastoid channels, so this is just for demonstration purposes:

# use a single channel reference (left earlobe)
# raw.set_eeg_reference(ref_channels=['A1'])

# use average of mastoid channels as reference
# raw.set_eeg_reference(ref_channels=['M1', 'M2'])

# use a bipolar reference (contralateral)
# raw.set_bipolar_reference(anode='[F3'], cathode=['F4'])

raw.plot(block=True)

# add new reference channel (all zero)
raw_new_ref = mne.add_reference_channels(raw, ref_channels=['EEG 999'])
raw_new_ref.plot(block=True)

# set reference to `EEG 050`
raw_new_ref.set_eeg_reference(ref_channels=['EEG 050'])
raw_new_ref.plot(block=True)

# use the average of all channels as reference
raw_avg_ref = raw.copy().set_eeg_reference(ref_channels='average')
raw_avg_ref.plot(block=True)

raw.set_eeg_reference('average', projection=True)
print(raw.info['projs'])

for title, proj in zip(['Original', 'Average'], [False, True]):
    with mne.viz.use_browser_backend('matplotlib'):
        fig = raw.plot(proj=proj, n_channels=len(raw))
    # make room for title
    fig.subplots_adjust(top=0.9)
    fig.suptitle('{} reference'.format(title), size='xx-large', weight='bold')

