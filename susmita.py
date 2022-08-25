from distutils.core import setup
import os
from setup import Setup as setup
import preprocessing
import mne

montage_path = os.path.join(os.getcwd(), 'data/Workspaces_Montages/active electrodes/actiCAP for LiveAmp 32 Channel','CLA-32.bvef')
print(os.listdir(os.path.join(os.getcwd(), 'data/raw_data/081822_PEDOT_single_electrodes')))
raw_data_list = os.listdir(os.path.join(os.getcwd(), 'data/raw_data/081822_PEDOT_single_electrodes'))
for file_name in raw_data_list:
    if file_name.endswith('.vhdr'):
        if file_name.replace('.vhdr','.fif') in raw_data_list:
            print(file_name, 'already has a preprocessed .fif file.')
        else:
            print(file_name, 'is not preprocessed.')
            # Read BV EEG files
            raw_path = os.path.join(os.getcwd(), 'data/raw_data/081822_PEDOT_single_electrodes', file_name)
            
            raw = setup(raw_path, montage_path, mode='BV')
            onset, duration, description = raw.get_annotation_info()

            filters = preprocessing.Filtering(raw.raw, l_freq=4, h_freq=50)
            raw.raw = filters.external_artifact_rejection()
            interactive_annot = raw.annotate_interactively()
            print(interactive_annot)
            raw.raw.save(os.path.join(os.getcwd(), 'data/raw_data', file_name.replace('.vhdr','.fif')))