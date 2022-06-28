import os
import random
import matplotlib
import matplotlib.pyplot as plt
import mne
matplotlib.use('TkAgg')

class Filtering:
    def __init__(self, raw, l_freq, h_freq, nf_freqs=(60,120,180)):
        # filter parameters
        self.raw = raw.copy()
        self.sfreq = raw.info['sfreq']
        self.picks = ['eeg', 'eog']
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.nf_freqs = nf_freqs

    def highpass(self, cutoff_freq=None, filt_type='fir', iir_params=None, show_filt_params = False, verbose='warning'):
        if cutoff_freq == None:
            cutoff_freq = self.l_freq
        filt_raw = self.raw.filter(
            l_freq=cutoff_freq, h_freq=None,
            picks=self.picks, method=filt_type,
            iir_params=iir_params, verbose=verbose)
        if show_filt_params == True:
            # check the filter parameter
            filter_params = mne.filter.create_filter(
                data=self.raw.get_data(), sfreq=self.sfreq,
                l_freq=cutoff_freq, h_freq=None,
                method=filt_type, iir_params=iir_params)
            mne.viz.plot_filter(h=filter_params, 
                sfreq=self.sfreq, flim=(0.01, self.sfreq/2))
        return filt_raw

    def lowpass(self, cutoff_freq=None, filt_type='fir', iir_params=None, show_filt_params = False, verbose='warning'):
        if cutoff_freq == None:
            cutoff_freq = self.h_freq
        filt_raw = self.raw.filter(
            l_freq=None, h_freq=cutoff_freq,
            picks=self.picks, method=filt_type,
            iir_params=iir_params, verbose=verbose)
        if show_filt_params == True:
            # check the filter parameter
            filter_params = mne.filter.create_filter(
                data=self.raw.get_data(), sfreq=self.sfreq,
                l_freq=cutoff_freq, h_freq=None,
                method=filt_type, iir_params=iir_params)
            mne.viz.plot_filter(h=filter_params, 
                sfreq=self.sfreq, flim=(0.01, self.sfreq/2))
        return filt_raw

    def notch(self, target_freqs=None, notch_wdith=None, filt_type='fir', iir_params=None, verbose='warning'):
        if target_freqs == None:
            target_freqs = self.nf_freqs
        filt_raw = self.raw.notch_filter(
            freqs=target_freqs, picks=self.picks,
            method=filt_type, iir_params=iir_params, verbose=verbose)
        return filt_raw

    def resample(self,new_sfreq=None, window='boxcar', events=None, verbose='warning'):
        if new_sfreq == None:
            new_sfreq == 4*self.h_freq
        if new_sfreq > self.sfreq:
            print('Error: New sampling frequency is greather than previous sampling freqeuncy. (Recommended new_f_s = 4 x high_f_c)')
            return
        filt_raw = self.raw.resample(sfreq=new_sfreq,
            napd='auto', window=window, events=events, verbose=verbose)
        return filt_raw

    def external_artifact_rejection(self):
        filt_raw = self.highpass()
        filt_raw = self.notch()
        filt_raw = self.lowpass()
        filt_raw = self.resample()
        return filt_raw
    
class Indepndent_Component_Analysis:
    def __init__(self,raw, n_components=None,seed=random.randint(1,100), max_iter= 'auto'):
        if n_components == None:
            n_components = raw.info['nchan']
        self.ica = mne.preprocessing.ICA(n_components=n_components, random_state=seed,max_iter= max_iter)
        self.ica.fit(raw) #whitening and n-comps ICA performed
    
    def visuallize_ICA_components(self):
        self.ica.plot_components
        plt.show()

    
    pass

