import os
from datetime import datetime, date, time, timedelta
import mne
import numpy as np
raw_path = os.path.join(os.getcwd(), 'data/raw_data', '070622_Dual_Nback_Test_CG_AM_1.vhdr')
raw = mne.io.read_raw_brainvision(raw_path)



path = os.path.join(os.getcwd(), 'data/reports', 'GonzalezCarlos_20220706_115204_Report.txt')
file = open(path, 'r')
content = list(file)
lines = []
for i in content:
    lines.append(i.replace('\n',''))
string1 = 'Timestamp before stimuli:'
index = 0
flage = 0
line_index = []
timestamps = []
stamp = []
nback_sequence = lines[0]
sequence = [int(s) for s in list(nback_sequence) if s.isdigit()]
print(sequence)
# Loop through the file line by line
for line in lines:
    # checking string is present in line or not
    if string1 in line:
        line_index.append(index)
        stamp = line.split(': ',1)[1]
        timestamps.append(stamp)
        flag = 1
        # 	break
    index += 1
if flag == 0:
    print('String', string1 , 'Not Found')
else:
    print('String', string1, 'Found In Line', line_index)

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

# print(nback_sequence)
# for i in timestamps_list:
nback_event = np.zeros((12*20,3))
for i in range(len(sequence)):
    for j in range(20):
        event_onset = timestamps_list[i] + j*2*fs
        event_duration = 0
        event_id = int(sequence[i])
        nback_event[i*20+j,:] = [event_onset, event_duration, event_id]

print(nback_event)