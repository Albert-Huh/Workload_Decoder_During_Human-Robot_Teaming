import os
from datetime import datetime, date, time, timedelta
import mne
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

# meas_date = raw.info['meas_date']
meas_isodate = datetime.fromisoformat(str(raw.info['meas_date']))
meas_time = meas_isodate.time()
meas_date = meas_isodate.date()
# time_offset = timedelta(hours=meas_time.hour,minutes=meas_time.minute,seconds=meas_time.second,microseconds=meas_time.microsecond)
print(meas_time)
timestamps_list = []
for i in timestamps:
    tdel = datetime.combine(meas_date, time.fromisoformat(i)) - datetime.combine(meas_date, meas_time)
    timestamps_list.append(tdel)
print(timestamps_list)
print(raw.annotations.onset)
print(raw.info)