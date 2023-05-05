from setup import Setup
from datetime import timedelta

# constants 
fs = 250 

# Create an instance of the Setup class
my_setup = Setup(raw_path=r"test_etat_data\eeg\USAARL Feb 2023 Shipping\From USAARL\Thu Feb 09 181227 CST 2023\EEG-Test_E0.85.DB.5C.86.C0\Thu Feb 09 181227 CST 2023.csv", 
                 montage_path=None, mode="USAARL E-tattoo", clock_correction=timedelta(days=56, hours=15, minutes=34))

# Call the get_events_from_nback_report function on the instance
nback_events = my_setup.get_events_from_nback_report(report_path=r"test_etat_data\eeg\USAARL Feb 2023 Shipping\From USAARL\Thu Feb 09 181227 CST 2023\EEG-Test_E0.85.DB.5C.86.C0\20230406_094815_S1001_2__Report.txt", fs=fs)
print(nback_events)

