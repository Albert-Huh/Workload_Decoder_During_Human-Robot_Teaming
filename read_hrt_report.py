from datetime import datetime, date, time, timedelta
import numpy as np

def read_report_txt(path):
    file = open(path, 'r')
    # Read text file line by line
    content = list(file)
    lines = []
    for i in content:
        lines.append(i.replace('\n',''))
    # Close the Report.txt
    file.close()
    return lines

def get_key_string():
    # Get the list of indication stings to extract data
    string_start = 'Simulation_Started,'
    string_end = 'Simulation_Ended_,'
    string_decision = 'Decision_Presented_@_'
    string_decision_user = 'Decision_Made_@_'
    string_messege = 'Chat_Presented_,'
    string_messege_user = 'Chat_Responded_,'
    key_string_list = [string_start, string_end,string_decision, string_messege, string_decision_user, string_messege_user]
    return key_string_list

def get_report_data(lines, key_string_list):
    # Get the list of Nback sequence
    hrt_sequence = lines[0]
    sequence = [int(s) for s in list(hrt_sequence) if s.isdigit()]
    # Create report dict data object
    report = {}
    report['hrt'] = sequence
    report['scenario_start'] = []
    report['scenario_end'] = []
    report['stim_decision'] = []
    report['stim_messege'] = []
    report['user_decision'] = []
    report['user_messege'] = []
    key_ind = ['scenario_start', 'scenario_end','stim_decision','stim_messege','user_decision','user_messege']

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
                if str_idx == 0 or str_idx == 1:
                    stamp = []
                    line_timestamps.append(line_idx)
                    stamp = line.split(',',1)[1]
                    timestamps.append(stamp)
                    report[key_ind[str_idx]].append(stamp)
                    flag = 1
            str_idx += 1
        line_idx += 1
    if flag == 0:
        print('The key string for stim timestamps were not found')
    else:
        print('The key string for stim timestamps were found in line', line_timestamps)
    return report

def get_stim_time_delta(report, meas_isodate, fs):
    # Get time delta between recording meas_timestamp and stim_timestamps
    meas_time = meas_isodate.time()
    meas_date = meas_isodate.date()
    start_samplestamp_tdel = []
    end_samplestamp_tdel = []
    for i in report['scenario_start']:
        tdel = datetime.combine(meas_date, time.fromisoformat(i)) - datetime.combine(meas_date, meas_time)
        start_t_sample = tdel.total_seconds()//(1/fs) + 10*fs
        start_samplestamp_tdel.append(start_t_sample)
    for i in report['scenario_end']:
        tdel = datetime.combine(meas_date, time.fromisoformat(i)) - datetime.combine(meas_date, meas_time)
        end_t_sample = tdel.total_seconds()//(1/fs)
        end_samplestamp_tdel.append(end_t_sample)
    return start_samplestamp_tdel, end_samplestamp_tdel

def get_hrt_event(report, start_samplestamp_tdel, end_samplestamp_tdel, fs, t_epoch_length):
    # Create event from stim time_delta
    hrt_events = np.empty((0,3), int)
    for i in range(len(report['hrt'])):
        ndel = end_samplestamp_tdel[i] - start_samplestamp_tdel[i]
        nepoch = int(ndel // (0.5*t_epoch_length*fs))
        # print(nepoch)
        scenario_events = np.zeros((nepoch,3), int)
        for j in range(nepoch):
            event_onset = start_samplestamp_tdel[i] + 0.5*(j+1)*t_epoch_length*fs
            event_duration = 0
            event_id = int(report['hrt'][i])
            scenario_events[j,:] = [event_onset, event_duration, event_id]
        hrt_events= np.concatenate((hrt_events, scenario_events), axis=0)
    return hrt_events