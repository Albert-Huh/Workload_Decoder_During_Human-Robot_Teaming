import os
from datetime import datetime, date, time, timedelta
from re import S
import mne
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
raw_path = os.path.join(os.getcwd(), 'data/raw_data', '070622_Dual_Nback_Test_CG_AM_1.vhdr')
raw = mne.io.read_raw_brainvision(raw_path)
'''
fig = raw.plot(block=True)
fig.fake_keypress('a')  # Simulates user pressing 'a' on the keyboard.
raw.save('70622_Dual_Nback_Test_CG_AM_1_raw.fif', overwrite=True)

raw = mne.io.read_raw_fif('70622_Dual_Nback_Test_CG_AM_1_raw.fif')
fig = raw.plot(block=True)
'''

path = os.path.join(os.getcwd(), 'data/reports', 'GonzalezCarlos_20220706_115204_Report.txt')
file = open(path, 'r')
content = list(file)
lines = []
for i in content:
    lines.append(i.replace('\n',''))
string_timestamp = 'Timestamp before stimuli: '
string_alpha = 'Alpha Sequence '
string_alpha_solution = 'Alpha Answer '
string_alpha_user = 'User Input Alpha: '
string_position = 'Position Sequence '
string_position_solution = 'Position Answer '
string_position_user = 'User Input Position: '
string_list = [string_timestamp, string_alpha, string_position, string_alpha_solution, string_position_solution, string_alpha_user, string_position_user]

index = 0
flag = 0
line_index = []
timestamps = []

nback_sequence = lines[0]
sequence = [int(s) for s in list(nback_sequence) if s.isdigit()]
print(sequence)
report = {}
report['nback'] = sequence
report['stim_alphabet'] = []
report['stim_position'] = []
report['sol_alphabet'] = []
report['sol_position'] = []
report['user_alphabet'] = []
report['user_position'] = []
key_ind = ['timestamps','stim_alphabet','stim_position','sol_alphabet','sol_position','user_alphabet','user_position']
# Loop through the file line by line
for line in lines:
    # checking string is present in line or not
    str_ind = 0
    for string in string_list:
        if string in line:
            if str_ind == 0:
                stamp = []
                line_index.append(index)
                stamp = line.split(': ',1)[1]
                timestamps.append(stamp)
                flag = 1
                # 	break
            elif str_ind == 1 or str_ind == 2:
                lst = line.split('[',1)[1].replace(']','').split(', ')
                lst = [int(s) for s in lst]
                report[key_ind[str_ind]].append(lst)
            else:
                lst = line.split('[',1)[1].replace(']','').split(', ')
                lst = [s == 'True' for s in lst]
                report[key_ind[str_ind]].append(lst)
        str_ind += 1
    index += 1
if flag == 0:
    print('String', string_timestamp , 'Not Found')
else:
    print('String', string_timestamp, 'Found In Line', line_index)

file.close()

print(timestamps)

meas_isodate = datetime.fromisoformat(str(raw.info['meas_date']))
fs = raw.info['sfreq']
print(fs)
meas_time = meas_isodate.time()
meas_date = meas_isodate.date()
# time_offset = timedelta(hours=meas_time.hour,minutes=meas_time.minute,seconds=meas_time.second,microseconds=meas_time.microsecond)
print(meas_time)
timestamps_list = []
for i in timestamps:
    tdel = datetime.combine(meas_date, time.fromisoformat(i)) - datetime.combine(meas_date, meas_time)
    t = tdel.total_seconds()//(1/fs)
    timestamps_list.append(t)
print(timestamps_list)
report['stim_timestamps'] = timestamps_list
# print(nback_sequence)
# for i in timestamps_list:
nback_event = np.zeros((12*20,3))
for i in range(len(sequence)):
    for j in range(20):
        event_onset = timestamps_list[i] + j*2*fs
        event_duration = 0
        event_id = int(sequence[i])
        nback_event[i*20+j,:] = [event_onset, event_duration, event_id]

# print(nback_event)
print(report)