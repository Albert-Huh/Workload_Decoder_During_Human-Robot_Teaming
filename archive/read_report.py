import os
from datetime import datetime, date, time, timedelta
import mne
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

## Import Raw File From BV and interactively reject bad data
# Read raw from vhdr
raw_path = os.path.join(os.getcwd(), 'data/raw_data', '070622_Dual_Nback_Test_CG_AM_1.vhdr')
raw = mne.io.read_raw_brainvision(raw_path)

# Interactively annotate
fig = raw.plot(block=True)
fig.fake_keypress('a')  # Simulates user pressing 'a' on the keyboard.
fig = raw.plot(block=True)
# Save annotated raw
raw.save('70622_Dual_Nback_Test_CG_AM_1_raw.fif', overwrite=True)

# raw = mne.io.read_raw_fif('70622_Dual_Nback_Test_CG_AM_1_raw.fif')
# fig = raw.plot(block=True)

## Read Report text file from Nback and extract key infos
# Read Report.txt as read only
path = os.path.join(os.getcwd(), 'data/reports', 'GonzalezCarlos_20220706_115204_Report.txt')
file = open(path, 'r')

# Read text file line by line
content = list(file)
lines = []
for i in content:
    lines.append(i.replace('\n',''))
# Close the Report.txt
file.close()

# Get the list of indication stings to extract data
string_timestamp = 'Timestamp before stimuli: '
string_alpha = 'Alpha Sequence '
string_alpha_solution = 'Alpha Answer '
string_alpha_user = 'User Input Alpha: '
string_position = 'Position Sequence '
string_position_solution = 'Position Answer '
string_position_user = 'User Input Position: '
key_string_list = [string_timestamp, string_alpha, string_position, string_alpha_solution, string_position_solution, string_alpha_user, string_position_user]

# Get the list of Nback sequence
nback_sequence = lines[0]
sequence = [int(s) for s in list(nback_sequence) if s.isdigit()]
# print(sequence)

# Create report dict data object
report = {}
report['nback'] = sequence
report['stim_timestamp'] = []
report['stim_alphabet'] = []
report['stim_position'] = []
report['sol_alphabet'] = []
report['sol_position'] = []
report['user_alphabet'] = []
report['user_position'] = []
key_ind = ['stim_timestamp','stim_alphabet','stim_position','sol_alphabet','sol_position','user_alphabet','user_position']

line_idx = 0
flag = 0
line_timestamps = []
timestamps = []
# Loop through the file line by line
for line in lines:
    # checking string is present in line or not
    str_idx = 0
    for string in key_string_list:
        if string in line:
            if str_idx == 0:
                stamp = []
                line_timestamps.append(line_idx)
                stamp = line.split(': ',1)[1]
                # timestamps.append(stamp)
                report[key_ind[str_idx]].append(stamp)
                flag = 1
            elif str_idx == 1 or str_idx == 2:
                lst = line.split('[',1)[1].replace(']','').split(', ')
                lst = [int(s) for s in lst]
                report[key_ind[str_idx]].append(lst)
            else:
                lst = line.split('[',1)[1].replace(']','').split(', ')
                lst = [s == 'True' for s in lst]
                report[key_ind[str_idx]].append(lst)
        str_idx += 1
    line_idx += 1
if flag == 0:
    print('String', string_timestamp , 'Not Found')
else:
    print('String', string_timestamp, 'Found In Line', line_timestamps)

# Get time delta between recording meas_timestamp and stim_timestamps
meas_isodate = datetime.fromisoformat(str(raw.info['meas_date']))
fs = raw.info['sfreq']
meas_time = meas_isodate.time()
meas_date = meas_isodate.date()
timestamp_tdel = []
for i in report['stim_timestamp']:
    tdel = datetime.combine(meas_date, time.fromisoformat(i)) - datetime.combine(meas_date, meas_time)
    t = tdel.total_seconds()//(1/fs)
    timestamp_tdel.append(t)

# Create event from stim time_delta
nback_event = np.zeros((12*20,3))
for i in range(len(report['nback'])):
    for j in range(20):
        event_onset = timestamp_tdel[i] + j*2*fs
        event_duration = 0
        event_id = int(report['nback'][i])
        nback_event[i*20+j,:] = [event_onset, event_duration, event_id]
# print(report)
print(nback_event)