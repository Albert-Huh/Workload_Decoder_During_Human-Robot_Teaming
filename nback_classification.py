import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from setup import Setup as setup
import preprocessing
import feature_extraction

# Classifier learning packages
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.covariance import OAS
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

raw_data_list = os.listdir(os.path.join(os.getcwd(), 'data/raw_data'))
bv_epochs_list = []
et_epochs_list = []
events_list = []

for file_name in raw_data_list:
    if file_name.endswith('.fif') and file_name.startswith('070622_Dual'): #startswith('date_Dual') can isolate experiment session  ('Dual', 7, 11) '070622_Dual' 062922_Dual
        raw_path_annot = os.path.join(os.path.join(os.getcwd(), 'data/raw_data'), file_name)
        montage_path = os.path.join(os.getcwd(), 'data/Workspaces_Montages/active electrodes/actiCAP for LiveAmp 32 Channel','CLA-32.bvef')
        raw_annot = setup(raw_path_annot, montage_path, mode='Binary')
        onset, duration, description = raw_annot.get_annotation_info()
        print(description)
        raw_path = os.path.join(os.path.join(os.getcwd(), 'data/raw_data'), file_name.replace('.fif','.vhdr'))
        raw = setup(raw_path, montage_path, mode='Dual')
        raw.set_annotation(raw.raw, onset=onset, duration=duration, description=description)
        # raw.get_brainvision_raw()
        # raw.get_e_tattoo_raw()
        # fig = raw.bv_raw.plot()
        # fig = raw.et_raw.plot()
        # plt.show()
        raw.get_brainvision_raw()
        raw.bv_raw.load_data()
        raw.bv_raw.set_eeg_reference('average')
        raw.get_e_tattoo_raw()
        raw.et_raw.load_data()
        raw.et_raw.set_eeg_reference(ref_channels=['A1', 'A2'])

        bv_filters = preprocessing.Filtering(raw.bv_raw, l_freq=1, h_freq=50)
        raw.bv_raw = bv_filters.external_artifact_rejection()
        et_filters = preprocessing.Filtering(raw.et_raw, l_freq=1, h_freq=50)
        raw.et_raw = et_filters.external_artifact_rejection()

        # raw.get_brainvision_raw()
        # raw.get_e_tattoo_raw()
        # fig = raw.bv_raw.plot()
        # fig = raw.et_raw.plot()
        # plt.show()
        # print(raw.raw.info['meas_date'])
        meas_date = str(raw.raw.info['meas_date'])
        recorder_meas_time = meas_date[0:4]+meas_date[5:7]+meas_date[8:10]+meas_date[11:19].replace(':','')
        report_list = os.listdir(os.path.join(os.getcwd(), 'data/reports'))
        for report_name in report_list:
            report_path = os.path.join(os.path.join(os.getcwd(), 'data/reports'), report_name)
            report_log_time = report_path.split('_',1)[1][0:15].replace('_', '')
            if abs(int(recorder_meas_time)-int(report_log_time)) < 60:
                resampled_freq = 200
                # print(report_path)
                nback_events = raw.get_events_from_nback_report(report_path=report_path, fs=resampled_freq)
        event_dict = {'0-back': 0, '1-back': 1, '2-back': 2}
        # fig = mne.viz.plot_events(nback_events, event_id=event_dict, sfreq=resampled_freq, first_samp=raw.bv_raw.first_samp)
        
        bv_ica = preprocessing.Indepndent_Component_Analysis(raw.bv_raw, n_components=8)
        et_ica = preprocessing.Indepndent_Component_Analysis(raw.et_raw, n_components=4)

        bv_eog_evoked = bv_ica.create_physiological_evoked()
        et_eog_evoked = et_ica.create_physiological_evoked()

        bv_ica.perfrom_ICA()
        et_ica.perfrom_ICA()
        # fig = raw.bv_raw.plot()
        # fig = raw.et_raw.plot()
        # plt.show()
        # print(nback_event)
        # del raw.bv_raw, raw.et_raw
        # raw.bv_raw.load_data()
        # raw.et_raw.load_data()
        '''
        bv_theta = preprocessing.Filtering(raw.bv_raw, 4, 7)
        bv_alpha = preprocessing.Filtering(raw.bv_raw, 8, 13)
        bv_beta = preprocessing.Filtering(raw.bv_raw, 14, 30)
        bv_theta_raw = bv_theta.bandpass()
        bv_alpha_raw = bv_alpha.bandpass()
        bv_beta_raw = bv_beta.bandpass()
        fig = bv_theta_raw.plot()
        fig = bv_alpha_raw.plot()
        fig = bv_beta_raw.plot()
        plt.show()

        bv_theta_epochs = mne.Epochs(bv_theta_raw, events=nback_events, event_id=event_dict, tmin=-0.2, tmax=1.8, preload=True, picks='eeg')
        bv_alpha_epochs = mne.Epochs(bv_alpha_raw, events=nback_events, event_id=event_dict, tmin=-0.2, tmax=1.8, preload=True, picks='eeg')
        bv_beta_epochs = mne.Epochs(bv_beta_raw, events=nback_events, event_id=event_dict, tmin=-0.2, tmax=1.8, preload=True, picks='eeg')
        fig = bv_alpha_epochs['0-back'].plot_image(picks='eeg',combine='mean')
        fig = bv_alpha_epochs['1-back'].plot_image(picks='eeg',combine='mean')
        fig = bv_alpha_epochs['2-back'].plot_image(picks='eeg',combine='mean')
        fig = bv_beta_epochs['0-back'].plot_image(picks='eeg',combine='mean')
        fig = bv_beta_epochs['1-back'].plot_image(picks='eeg',combine='mean')
        fig = bv_beta_epochs['2-back'].plot_image(picks='eeg',combine='mean')
        fig = bv_theta_epochs['0-back'].plot_image(picks='eeg',combine='mean')
        fig = bv_theta_epochs['1-back'].plot_image(picks='eeg',combine='mean')
        fig = bv_theta_epochs['2-back'].plot_image(picks='eeg',combine='mean')
        '''

        bv_epochs = mne.Epochs(raw=raw.bv_raw, events=nback_events, event_id=event_dict, tmin=-0.2, tmax=1.8, preload=True, picks='eeg')
        bv_epochs.equalize_event_counts()
        bv_epochs_list.append(bv_epochs)
        et_epochs = mne.Epochs(raw=raw.et_raw, events=nback_events, event_id=event_dict, tmin=-0.2, tmax=1.8, preload=True, picks=['Fp1','Fp2','F7','F8'])
        et_epochs.equalize_event_counts()
        et_epochs_list.append(et_epochs)
        events_list.append(nback_events)

# all_events = mne.concatenate_events(events_list)

all_bv_epochs = mne.concatenate_epochs(bv_epochs_list)
all_et_epochs = mne.concatenate_epochs(et_epochs_list)
# print(len(all_bv_epochs))
# print(len(all_et_epochs))
bv_x = feature_extraction.eeg_power_band(all_bv_epochs, mean=False)
et_x = feature_extraction.eeg_power_band(all_et_epochs,mean=False)
print(bv_x.shape)
print(et_x.shape)
y = all_bv_epochs.events[:,2]
print(y.shape)
X_train, Y_train, X_test, Y_test = feature_extraction.create_train_test_sets(bv_x, y, 0.3)
print(len(X_train), len(Y_train), len(X_test), len(Y_test))

############### CLASSIFICATION ###############
############### Random forest classification
# pipe_RF = make_pipeline(FunctionTransformer(feature_extraction.eeg_power_band, validate=False), RandomForestClassifier(n_estimators=100, random_state=42))
pipe_RF = make_pipeline(RandomForestClassifier(n_estimators=50, random_state=42))
pipe_RF.fit(X_train, Y_train)

# Test
Y_pred = pipe_RF.predict(X_test)
# Assess the results
acc = accuracy_score(Y_test, Y_pred)
print('Random Forest Classifier')
print('Accuracy score: {}'.format(acc))
print('Confusion Matrix:')
print(confusion_matrix(Y_test, Y_pred))
print('Classification Report:')
print(classification_report(Y_test, Y_pred, target_names=event_dict.keys()))

############### Linear discriminant analysis
oa = OAS(store_precision=False, assume_centered=False)
# pipe_LDA = make_pipeline(FunctionTransformer(feature_extraction.eeg_power_band, validate=False), LinearDiscriminantAnalysis(solver="lsqr", covariance_estimator=oa))
pipe_LDA = make_pipeline(LinearDiscriminantAnalysis(solver="lsqr", covariance_estimator=oa))
pipe_LDA.fit(X_train, Y_train)
# Test
Y_pred = pipe_RF.predict(X_test)
# Assess the results
acc = accuracy_score(Y_test, Y_pred)
print('Linear Discriminant Analysis')
print('Accuracy score: {}'.format(acc))
print('Confusion Matrix:')
print(confusion_matrix(Y_test, Y_pred))
print('Classification Report:')
print(classification_report(Y_test, Y_pred, target_names=event_dict.keys()))

###############  K-nearest neighbors classification
k_range = np.arange(1,20)
weight_fuc = ['uniform', 'distance']
opt_KNN_param = []
acc_max = 0
opt_Y_pred = []
for KNN_param in [(x, y) for x in k_range for y in weight_fuc]:
    pipe_KNN = make_pipeline(KNeighborsClassifier(n_neighbors=KNN_param[0],weights=KNN_param[1]))
    pipe_KNN.fit(X_train, Y_train)
    # Test
    Y_pred = pipe_KNN.predict(X_test)
    # Assess the results
    acc = accuracy_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred, average='weighted', labels=np.unique(Y_pred))
    if acc_max < acc and f1 > 0.0:
        acc_max = acc
        opt_KNN_param = KNN_param
        opt_Y_pred = Y_pred

print('KNN Classifier')
print(opt_KNN_param)
print('Accuracy score: {}'.format(acc))
print('Confusion Matrix:')
print(confusion_matrix(Y_test, opt_Y_pred))
print('Classification Report:')
print(classification_report(Y_test, opt_Y_pred, target_names=event_dict.keys()))

############### Support vector machine
C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
opt_C_gamma_param = []
acc_max = 0
opt_Y_pred = []
for C_gamma_param in [(x, y) for x in C_range for y in gamma_range]:
    pipe_SVM = make_pipeline(StandardScaler(), SVC(kernel='rbf',C=C_gamma_param[0],gamma=C_gamma_param[1]))
    pipe_SVM.fit(X_train, Y_train)
    # Test
    Y_pred = pipe_SVM.predict(X_test)
    acc = accuracy_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred, average='weighted', labels=np.unique(Y_pred))
    if acc_max < acc and f1 > 0.0:
        acc_max = acc
        opt_C_gamma_param = C_gamma_param
        opt_Y_pred = Y_pred

# pipe_SVM = make_pipeline(StandardScaler(), SVC(kernel='rbf',gamma=10**-1, C=10**0))
# pipe_SVM.fit(X_train, Y_train)
# # Test
# Y_pred = pipe_SVM.predict(X_test)
# acc = accuracy_score(Y_test, Y_pred)
print('SVM Classifier')
print(opt_C_gamma_param)
print('Accuracy score: {}'.format(acc_max))
print('Confusion Matrix:')
print(confusion_matrix(Y_test, opt_Y_pred))
print('Classification Report:')
print(classification_report(Y_test, opt_Y_pred, target_names=event_dict.keys()))
# Assess the results
###### Archive
# print(y)
# all_events = mne.concatenate_events(events_list)[:,2]
# print(all_events.shape)
# print(y == all_events)
# print(x)
# print(x.size)
# print(len(x))
# print(x.shape)
# print(type(y_bv))

        # et_epochs = mne.Epochs(raw=raw.et_raw, events=nback_events, event_id=event_dict, tmin=-0.2, tmax=1.8, preload=True, picks='eeg')
        # print(bv_epochs)
        # print(et_epochs)
        # zeroback_epochs = bv_epochs['0-back']
        # fig = bv_epochs['0-back'].plot_image(picks='eeg',combine='mean')
        # fig = bv_epochs['1-back'].plot_image(picks='eeg',combine='mean')00
        # fig = bv_epochs['2-back'].plot_image(picks='eeg',combine='mean')
        # plt.show()
        # break
        