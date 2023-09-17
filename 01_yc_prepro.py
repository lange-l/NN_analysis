"""
author: Lena Lange
last update: 24.04.2023
Version: Python 3.9

- 2 finger (left ring and index finger) electrical stimulation near the perceptual threshold (50% detection rate)
- Y/N detection task and 2AFC localisation task
- anatomical and crossed hand position (left hand in right hemispace)

Pre-processing of EEG data:
1. Cropping and concatenating experimental blocks 1-5, setting channel types (EOG, ECG) and electrode montage
2. Manually marking bad channels, interpolating them, re-referencing to average
3. Applying a FIR bandpass (1 Hz, 45 Hz) & notch (50 Hz) filter
4. Downsampling to 250 Hz, Artefact correction using ICA: 55 PCA components, extended infomax algorithm, using ICLabel
   to classify ICs (+ visual confirmation)
5. Epoching (from -0.4s to 0.5s to stimulus), adding behavioural metadata, baseline correction (-0.1s to 0s to stimulus
   onset)
6. Epoch rejection based on behavioural data


TO DO:
-pre-process yc06 (buffer overflow), yc28 (weird boundaries between blocks), yc29 (lots of noisy channels)
"""

import os
import os.path as op
import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from csv import writer
import pickle

# Set relevant data paths:
# eeg data path
dir_eeg = r'/data/pt_02438/numbtouch_neglect/young_controls/yc_eeg/%s/'
# behavioural data path
dir_behav = r'/data/pt_02438/numbtouch_neglect/young_controls/yc_behav/'
# ICA path (ICA object, component plots etc)
dir_ica = r'/data/pt_02438/numbtouch_neglect/young_controls/yc_eeg/ica/newfilt/'
# save plots to
dir_plots = r'/data/pt_02438/numbtouch_neglect/young_controls/yc_eeg/plots/newfilt/'

# Set file names:
# raw eeg file name (first %s = participant ID, second %s = block name)
fname_raw = r'numbtouch_neglect_EEG_%s_%s.vhdr'
# concatenated .fif file name
fname_conc = r'%s_conc_raw.fif'
# filtered data
fname_filt = r'%s_newfilt_raw.fif'
# ICA object
fname_ica = r'%s_newfilt_ica.fif'
# ICs plot
plt_comp = r'%s_newfilt_components.pdf'
# properties plot
plt_prop = r'%s_newfilt_properties.pdf'
# overlay plot
plt_overlay = r'%s_newfilt_overlay.png'
# .csv file with calculated probability values from ICLabel
fname_csv = r'newfilt_ICLabels_%s.csv'
# artifact corrected EEG data
fname_filt_ica = r'%s_newfilt_clean_raw.fif'
# behavioural data frame
fname_behav = r'yc_trials.csv'
# epochs
fname_epo = r'%s_newfilt_epo.fif'
# behaviourally pre-processed epochs
fname_epo_behav = r'%s_newfilt_behav_epo.fif'
# behaviourally pre-processed epochs without ICA
fname_epo_noica = r'%s_newfilt_noica_epo.fif'

# list of participant IDs:
subj_names = ['yc01', 'yc02', 'yc03', 'yc04', 'yc05', 'yc06', 'yc07', 'yc08', 'yc09', 'yc10', 'yc11', 'yc12', 'yc13',
              'yc14', 'yc15', 'yc16', 'yc17', 'yc18', 'yc19', 'yc20', 'yc21', 'yc22', 'yc23', 'yc24', 'yc25', 'yc26',
              'yc27', 'yc28', 'yc29', 'yc30', 'yc31', 'yc32', 'yc33', 'yc34', 'yc35', 'yc36', 'yc37', 'yc38', 'yc39',
              'yc40']
subj_excl = ['yc06', 'yc28', 'yc29']
# yc06 BufferOverflow, yc28 weird boundary between blocks?, yc29 too many channels excluded for ICA

raw_dict = {}
for subj in subj_names:
    if subj in subj_excl:
        continue
    raw_dict[subj] = mne.io.read_raw_fif(dir_eeg % subj + fname_filt_ica % subj, preload=True)

################################################################################
# 1) Load raw data blocks, crop & concatenate them into 1 .fif file per participant:
exp_blocks = ['0001', '0002', '0003', '0004', '0005']
trial_numbers = []

for subj in subj_names:
    if subj in subj_excl:
        continue
    # Load all blocks for participant
    blocks = []
    for exp in exp_blocks:
        fname = fname_raw % (subj, exp)
        if not os.path.exists(os.path.join(dir_eeg % subj, fname)):
            continue

        raw = mne.io.read_raw_brainvision(os.path.join(dir_eeg % subj, fname))

        # Crop data to relevant time window
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        events = events[2:]
        tmin = events[0, 0] / 2500 - 1
        tmax = events[-1, 0] / 2500 + 2
        raw.crop(tmin=tmin, tmax=tmax)

        # Save trial count and append to list
        events, event_id = mne.events_from_annotations(raw)
        trial_numbers.append(len(events))

        blocks.append(raw)

    # Concatenate blocks
    conc = mne.concatenate_raws(blocks)

    # Set channel types for VEOG, ECG, TP7, CPz electrodes
    conc.set_channel_types({'VEOG': 'eog', 'ECG': 'ecg', 'TP7': 'eeg', 'CPz': 'eeg'}, verbose=False)

    # Set montage
    ten_ten_montage = mne.channels.make_standard_montage('easycap-M1')
    conc.set_montage(ten_ten_montage)

    # Save concatenated unfiltered data
    conc.save(os.path.join(dir_eeg % subj, fname_conc % subj), overwrite=True)

    # store in raw_dict
    raw_dict[subj] = conc

################################################################################
# 2) & 3) Interpolate bad channels, re-reference & filter

# visually identify bad channels:
for subj in subj_names:
    if subj in subj_excl:
        continue

    conc = raw_dict[subj]
    conc.plot(n_channels=64, duration=25)
    plt.show(block=True)

# Dictionary to store the bad channels for each subject ID
bads_dict = {
    'yc01': ['FT10'],
    'yc02': [],
    'yc03': ['AF7', 'Iz'],
    'yc04': ['AF7', 'FT8', 'FT10', 'Iz'],
    'yc05': ['F7', 'FC5', 'F5', 'Iz'],
    'yc06': [],
    'yc07': [],
    'yc08': ['T7', 'T8'],
    'yc09': ['T8', 'FT10', 'FC6', 'C6', 'FT8', 'F6', 'Iz'],
    'yc10': ['O1', 'Oz', 'Iz'],
    'yc11': ['T7', 'P7'],
    'yc12': ['FC5', 'F5', 'FC3'],
    'yc13': ['Iz', 'T7'],
    'yc14': ['AF7', 'T7', 'TP7'],
    'yc15': ['Iz'],
    'yc16': [],
    'yc17': ['AF7', 'Iz'],
    'yc18': ['T7', 'Iz'],
    'yc19': [],  # very noisy longer segments
    'yc20': ['FT10'],
    'yc21': [],
    'yc22': [],
    'yc23': [],
    'yc24': [],
    'yc25': [],
    'yc26': ['T8', 'FT7'],
    'yc27': ['T7', 'T8', 'C5', 'TP7'],
    'yc28': ['C6', 'FT8', 'AF8', 'T8', 'FT10', 'FC6'],
    'yc29': ['F3', 'T7', 'T8', 'FT10', 'FC6', 'F4', 'F8', 'FT7', 'TP8', 'FT8', 'F6', 'AF8'],  # very noisy in general
    'yc30': [],
    'yc31': ['T7', 'T8'],
    'yc32': [],
    'yc33': [],
    'yc34': ['Iz'],
    'yc35': ['Iz'],
    'yc36': [],
    'yc37': [],
    'yc38': ['AF4', 'AF8', 'F6'],
    'yc39': ['AF7'],
    'yc40': []
    # Add more subject IDs and bad channels if needed
}

for subj in subj_names:
    if subj in subj_excl:
        continue

    # get bad channels for current subject
    bads = bads_dict.get(subj, [])

    # load eeg data, exclude bad channels & interpolate them
    conc = raw_dict[subj]
    conc.info['bads'] = bads
    #conc.load_data()
    conc.interpolate_bads(reset_bads=False)

    # set reference
    conc.set_eeg_reference(ref_channels='average', ch_type='eeg')

    # highpass filter at 1 Hz to remove slow drifts, lowpass filter at 45 Hz
    conc_bandpass = conc.copy().filter(l_freq=1, h_freq=45, method='fir')  # _newfilt_: 45 Hz, _conc_: 60 Hz

    # notch filter at 50 Hz to remove power line artefacts
    picks = mne.pick_types(conc.info, eeg=True)
    freqs = (50, 100, 150, 200, 250)
    conc_notch = conc_bandpass.copy().notch_filter(freqs=freqs)

    # save concatenated filtered data
    conc_notch.save(dir_eeg % subj + fname_filt % subj, overwrite=True)


################################################################################
# 4) Downsampling & ICA

# define function for saving ICs & properties in 1 PDF file per participant
def save_components(filename):
    p = PdfPages(filename)  # PdfPages is a wrapper around pdf file so there is no clash and create files with no error.
    for fig in figs:  # iterating over the numbers in list and saving the files
        fig.savefig(p, format='pdf')
    p.close()  # close the object


for subj in subj_names:
    if subj in subj_excl:
        continue

    print('Processing participant %s...' % subj)

    # load & resample to 250 Hz
    cont = raw_dict[subj]
    cont.resample(250, npad='auto')

    # initialize ICA object with extended Infomax algorithm and 55 components
    ica = ICA(method='infomax',
              fit_params=dict(extended=True),
              n_components=55)

    # fit to continuous data & save ICA solution
    ica.fit(cont, picks=['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1',
                         'Oz', 'O2', 'P4', 'P8', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8',
                         'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5',
                         'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8',
                         'F6', 'AF8', 'AF4', 'F2', 'Iz'])
    ica.save(dir_eeg % subj + fname_ica % subj)

    # plot ICs
    figs = ica.plot_components(show=False)
    filename = plt_comp
    save_components(dir_ica + filename % subj)
    plt.close('all')

    # plot properties
    figs = ica.plot_properties(cont, picks=list(range(0, 55)), show=False)
    filename = plt_prop
    save_components(dir_ica + filename % subj)
    plt.close('all')

    # classify every IC & calculate probability values
    ic_labels = label_components(cont, ica, method="iclabel")

    # shorten label names
    ic_labels['labels'] = ['eye' if x == 'eye blink' else x for x in ic_labels['labels']]
    ic_labels['labels'] = ['muscle' if x == 'muscle artifact' else x for x in ic_labels['labels']]
    ic_labels['labels'] = ['channel' if x == 'channel noise' else x for x in ic_labels['labels']]
    ic_labels['labels'] = ['heart' if x == 'heart beat' else x for x in ic_labels['labels']]

    # write into data frame
    ic_labels = pd.DataFrame(ic_labels)

    # add column with participant ID
    ic_labels['subj_ID'] = subj

    # extract labels, extract probability values, add 2 columns to ic_labels data frame with 1s & 0s
    # (exclude: all labels that are not "brain" or "others" with probability > 60%;
    #  include: all labels that are "brain" with probability < 60%)
    labels = ic_labels["labels"]
    y_pred_proba = ic_labels["y_pred_proba"]

    exclude_idx = [idx for idx, (label, prob) in enumerate(zip(labels, y_pred_proba))
                   if ((label not in ["brain", "other"]) & (prob > 0.6))]
    ic_labels['excluded'] = 0
    ic_labels.loc[exclude_idx, "excluded"] = 1

    include_idx = [idx for idx, (label, prob) in enumerate(zip(labels, y_pred_proba))
                   if ((label in ["brain"]) & (prob > 0.6))]
    ic_labels['included'] = 0
    ic_labels.loc[include_idx, "included"] = 1

    print(f"Excluding these ICA components: {exclude_idx}")

    # export components dataframe into .csv file
    ic_labels.to_csv(dir_ica + fname_csv % subj, header=False)

# Dictionary to store the ICs to be excluded for each subject ID

"""
# ICA on epoched data
excl_dict = {
    'yc01': [0,1,2,10,15,18,20,21,22,23,25,26,27,28,29,30,31,32,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,
             52,53,54],
        #[0, 1, 2, 6, 14, 17, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
         #    42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],

    'yc02': [0,2,3,5,8,19,20,21,23,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,43,44,45,46,47,48,49,50,51,52,53,54],
        #[0, 2, 3, 7, 16, 17, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
         #    42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],

    'yc03': [0,2,3,4,6,7,11,15,16,17,18,19,21,24,25,26,27,28,29,30,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,
             49,50,51,52,53,54],
        #[0, 2, 3, 4, 6, 7, 9, 12, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 37, 39,
         #    40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],

    'yc04': [0,1,2,9,10,11,12,13,14,16,17,22,23,25,26,27,29,30,31,32,33,34,35,36,37,41,43,44,49,51],
        #[0, 1, 2, 3, 6, 9, 10, 12, 14, 15, 16, 17, 22, 23, 25, 26, 27, 28, 30, 31, 32, 34, 35, 37, 38, 39, 40, 41,
         #    42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],

    'yc05': [0,1,3,5,7,11,17,18,19,20,21,22,23,24,28,29,30,32,33,34,35,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,
             54],
        #[0, 1, 5, 7, 6, 14, 16, 17, 18, 21, 22, 25, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 41, 42, 43, 44, 45, 46,
         #    47, 48, 49, 50, 51, 52, 53, 54],

    'yc06': [],
        #[],

    'yc07': [0,1,2,27,28,30,34,35,36,37,39,40,42,43,44,45,46,47,48,49,50,51,52,53,54],
        #[0, 1, 12, 26, 27, 28, 30, 31, 32, 35, 36, 37, 38, 39, 40, 41, 42, 43, 46, 47, 48, 49, 50, 51, 52, 53, 54],

    'yc08': [0,4,6,9,11,16,17,18,19,21,22,24,26,27,29,30,32,33,34,35,36,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,
             54],
        #[0, 3, 6, 7, 11, 18, 19, 20, 21, 22, 23, 24, 25, 27, 29, 30, 32, 34, 35, 36, 37, 39, 40, 41, 42, 43, 44, 45,
         #    46, 47, 48, 49, 50, 51, 52, 53, 54],

    'yc09': [0,1,2,3,6,15,16,17,23,24,25,26,27,29,30,31,32,34,35,37,38,40,43,44,45,46,47,48,49,50,51,52,53,54],
        #[0, 1, 2, 4, 9, 15, 17, 18, 19, 21, 24, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 40, 41, 42, 43, 44, 45, 46,
         #    47, 48, 49, 50, 51, 52, 53, 54],

    'yc10': [0,1,4,10,18,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,
             52,53,54],
        #[0, 1, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
         #    42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],

    'yc11': [0,1,3,6,26,29,30,31,32,33,35,36,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54],
        #[0, 1, 6, 8, 21, 25, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
         #    53, 54],

    'yc12': [0,2,11,12,13,14,15,17,18,19,20,21,22,23,24,26,27,28,31,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,
             53,54],
        #[0, 2, 8, 10, 11, 13, 14, 16, 17, 18, 19, 20, 21, 22, 24, 25, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38,
         #    39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],

    'yc13': [0,14,16,18,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,
             53,54],
        #[0, 3, 14, 17, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
         #    43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],

    'yc14': [0,2,17,18,19,20,21,22,23,24,25,26,28,29,30,32,33,34,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,
             54],
        #[0, 2, 15, 16, 17, 20, 21, 22, 23, 24, 25, 26, 27, 32, 34, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47, 48,
         #    49, 50, 51, 52, 53, 54],

    'yc15': [0,1,2,3,4,5,7,10,11,14,15,16,17,18,19,23,24,25,26,27,28,30,31,33,34,35,36,37,38,41,42,43,44,45,46,47,48,49,
             50,51,52,53,54],
        #[0, 1, 2, 3, 4, 6, 7, 9, 13, 15, 16, 17, 18, 19, 21, 22, 23, 24, 26, 27, 29, 30, 31, 34, 35, 36, 37, 38, 40,
         #    41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],

    'yc16': [0,1,5,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,37,38,39,42,43,44,45,46,47,48,49,50,51,52,53,54],
        #[0, 1, 11, 17, 20, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 34, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
         #    48, 49, 50, 51, 52, 53, 54],
    'yc17': [0,1,2,12,15,16,18,19,20,23,25,26,27,28,29,32,33,35,36,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54],
        #[0, 1, 2, 11, 12, 16, 17, 18, 19, 20, 21, 23, 26, 27, 28, 29, 30, 33, 34, 36, 37, 38, 39, 40, 41, 42, 44,
         #    45, 46, 47, 48, 49, 50, 51, 52, 53, 54],

    'yc18': [0,3,12,25,27,28,29,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54],
        #[0, 3, 19, 22, 23, 25, 26, 27, 29, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
         #    49, 50, 51, 52, 53, 54],

    'yc19': [0,1,2,3,4,5,8,9,10,13,14,15,16,17,18,19,20,21,22,23,24,26,27,28,31,34,35,36,37,39,40,41,42,43,44,45,46,47,
             48,49,50,51,52,53,54],
        #[0, 1, 2, 3, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
         #    34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
         # very noisy longer segments, few good ICs

    'yc20': [0,1,4,8,13,15,19,20,21,25,28,29,30,31,32,33,34,35,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54],
        #[0, 1, 3, 7, 8, 12, 13, 14, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
         #    38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc21': [0,2,15,18,19,21,22,25,26,28,30,31,32,33,34,35,36,37,38,39,40,41,42,44,45,46,47,48,49,50,51,52,53,54],
        #[0, 2, 13, 14, 15, 16, 17, 18, 20, 21, 23, 24, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 40, 41, 42,
         #    43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],

    'yc22': [0,1,2,12,13,14,17,19,21,22,25,26,27,28,29,30,32,35,36,37,38,40,41,42,43,44,46,47,48,49,50,51,52,53,54],
        #[0, 1, 2, 7, 10, 13, 15, 17, 18, 20, 21, 22, 24, 26, 27, 28, 29, 30, 31, 32, 33, 34, 39, 40, 41, 42, 44, 46,
         #    47, 48, 49, 50, 51, 52, 53, 54],

    'yc23': [0,1,2,4,7,9,10,12,14,15,16,19,20,21,22,23,25,27,31,32,33,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,
             51,52,53,54],
        #[0, 1, 2, 3, 7, 8, 14, 16, 17, 18, 19, 21, 22, 23, 26, 28, 29, 30, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
         #    43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],

    'yc24': [0,3,4,9,10,11,14,16,18,22,26,28,30,31,32,34,35,38,39,41,42,43,44,45,47,48,49,50,51,52,53,54],
        #[0, 3, 4, 7, 10, 11, 12, 17, 20, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 37, 38, 39, 40, 41, 42, 43,
         #    44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],

    'yc25': [0,1,3,13,15,16,17,18,19,21,22,23,26,27,28,30,31,32,33,36,37,41,45,46,47,48,49,50,51,52,53,54],
        #[0, 1, 4, 6, 9, 11, 12, 16, 18, 21, 22, 24, 25, 26, 27, 28, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
         #    43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],

    'yc26': [0,1,3,4,5,6,7,10,14,15,18,19,21,22,23,24,25,26,27,28,30,31,33,34,35,37,39,40,41,42,43,44,45,46,47,48,49,50,
             51,52,53,54]
        #[0, 1, 3, 5, 6, 8, 9, 12, 13, 14, 16, 17, 20, 21, 22, 23, 24, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
         #    38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]
    'yc27': [0, 2, 9, 12, 16, 19, 20, 21, 22, 23, 24, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 41, 42, 44,
             45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc28': [],
    'yc29': [],
    'yc30': [0, 1, 7, 14, 15, 16, 17, 21, 23, 25, 26, 27, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 42, 43, 44, 45, 46,
             47, 48, 49, 50, 51, 52, 53, 54],
    'yc31': [0, 1, 2, 4, 7, 8, 9, 19, 20, 24, 25, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 40, 41, 42, 43, 47, 48, 49,
             50, 51, 52, 53, 54],
    'yc32': [0, 3, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
             36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc33': [0, 1, 4, 14, 15, 16, 18, 19, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 37, 40, 41, 42, 43,
             44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc34': [0, 1, 2, 10, 12, 16, 17, 19, 20, 21, 22, 23, 24, 26, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
             43, 45, 46, 47, 48, 52, 53, 54],
    'yc35': [0, 1, 12, 13, 14, 16, 23, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
             50, 51, 52, 53, 54],
    'yc36': [0, 4, 6, 21, 26, 29, 30, 31, 32, 35, 36, 37, 38, 40, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc37': [0, 1, 2, 3, 4, 9, 12, 13, 15, 16, 19, 20, 21, 22, 24, 25, 26, 29, 31, 32, 33, 36, 37, 38, 40, 41, 42, 43,
             44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc38': [0, 1, 9, 11, 13, 15, 16, 17, 19, 20, 21, 23, 26, 27, 28, 29, 30, 31, 34, 35, 36, 37, 38, 39, 40, 41, 42,
             43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]
    # Add more subject IDs when needed
}
"""

# strict exclusion
excl_dict = {
    'yc01': [0, 1, 2, 10, 15, 18, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42,
             43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc02': [0, 2, 3, 5, 8, 19, 20, 21, 23, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 43, 44, 45,
             46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc03': [0, 2, 3, 4, 6, 7, 11, 15, 16, 17, 18, 19, 21, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39,
             40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc04': [0, 1, 2, 9, 10, 11, 12, 13, 14, 16, 17, 22, 23, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 37, 41, 43, 44,
             49, 51],
    'yc05': [0, 1, 3, 5, 7, 11, 17, 18, 19, 20, 21, 22, 23, 24, 28, 29, 30, 32, 33, 34, 35, 38, 39, 40, 41, 42, 43, 44,
             45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc06': [],
    'yc07': [0, 1, 2, 27, 28, 30, 34, 35, 36, 37, 39, 40, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc08': [0, 4, 6, 9, 11, 16, 17, 18, 19, 21, 22, 24, 26, 27, 29, 30, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44,
             45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc09': [0, 1, 2, 3, 6, 15, 16, 17, 23, 24, 25, 26, 27, 29, 30, 31, 32, 34, 35, 37, 38, 40, 43, 44, 45, 46, 47, 48,
             49, 50, 51, 52, 53, 54],
    'yc10': [0, 1, 4, 10, 18, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
             43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc11': [0, 1, 3, 6, 26, 29, 30, 31, 32, 33, 35, 36, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
             54],
    'yc12': [0, 2, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 31, 36, 37, 38, 39, 40, 41, 42, 43,
             44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc13': [0, 14, 16, 18, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
             44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc14': [0, 2, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38, 39, 40, 41, 42, 43, 44,
             45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc15': [0, 1, 2, 3, 4, 5, 7, 10, 11, 14, 15, 16, 17, 18, 19, 23, 24, 25, 26, 27, 28, 30, 31, 33, 34, 35, 36, 37,
             38, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc16': [0, 1, 5, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 37, 38, 39, 42, 43, 44, 45, 46,
             47, 48, 49, 50, 51, 52, 53, 54],
    'yc17': [0, 1, 2, 12, 15, 16, 18, 19, 20, 23, 25, 26, 27, 28, 29, 32, 33, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45,
             46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc18': [0, 3, 12, 25, 27, 28, 29, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
             52, 53, 54],
    'yc19': [0, 1, 2, 3, 4, 5, 8, 9, 10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 31, 34, 35, 36, 37,
             39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc20': [0, 1, 4, 8, 13, 15, 19, 20, 21, 25, 28, 29, 30, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
             47, 48, 49, 50, 51, 52, 53, 54],
    'yc21': [0, 2, 15, 18, 19, 21, 22, 25, 26, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47,
             48, 49, 50, 51, 52, 53, 54],
    'yc22': [0, 1, 2, 12, 13, 14, 17, 19, 21, 22, 25, 26, 27, 28, 29, 30, 32, 35, 36, 37, 38, 40, 41, 42, 43, 44, 46,
             47, 48, 49, 50, 51, 52, 53, 54],
    'yc23': [0, 1, 2, 4, 7, 9, 10, 12, 14, 15, 16, 19, 20, 21, 22, 23, 25, 27, 31, 32, 33, 35, 36, 37, 38, 39, 40, 41,
             42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc24': [0, 3, 4, 9, 10, 11, 14, 16, 18, 22, 26, 28, 30, 31, 32, 34, 35, 38, 39, 41, 42, 43, 44, 45, 47, 48, 49, 50,
             51, 52, 53, 54],
    'yc25': [0, 1, 3, 13, 15, 16, 17, 18, 19, 21, 22, 23, 26, 27, 28, 30, 31, 32, 33, 36, 37, 41, 45, 46, 47, 48, 49,
             50, 51, 52, 53, 54],
    'yc26': [0, 1, 3, 4, 5, 6, 7, 10, 14, 15, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 30, 31, 33, 34, 35, 37, 39, 40,
             41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc27': [0, 2, 9, 12, 16, 19, 20, 21, 22, 23, 24, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 41, 42, 44,
             45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc28': [],
    'yc29': [],
    'yc30': [0, 1, 7, 14, 15, 16, 17, 21, 23, 25, 26, 27, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 42, 43, 44, 45, 46,
             47, 48, 49, 50, 51, 52, 53, 54],
    'yc31': [0, 1, 2, 4, 7, 8, 9, 19, 20, 24, 25, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 40, 41, 42, 43, 47, 48, 49,
             50, 51, 52, 53, 54],
    'yc32': [0, 3, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
             36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc33': [0, 1, 4, 14, 15, 16, 18, 19, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 37, 40, 41, 42, 43,
             44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc34': [0, 1, 2, 10, 12, 16, 17, 19, 20, 21, 22, 23, 24, 26, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
             43, 45, 46, 47, 48, 52, 53, 54],
    'yc35': [0, 1, 12, 13, 14, 16, 23, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
             50, 51, 52, 53, 54],
    'yc36': [0, 4, 6, 21, 26, 29, 30, 31, 32, 35, 36, 37, 38, 40, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc37': [0, 1, 2, 3, 4, 9, 12, 13, 15, 16, 19, 20, 21, 22, 24, 25, 26, 29, 31, 32, 33, 36, 37, 38, 40, 41, 42, 43,
             44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc38': [0, 1, 9, 11, 13, 15, 16, 17, 19, 20, 21, 23, 26, 27, 28, 29, 30, 31, 34, 35, 36, 37, 38, 39, 40, 41, 42,
             43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc39': [0, 3, 5, 7, 8, 9, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 26, 27, 28, 29, 30, 32, 33, 35, 37, 38, 39, 40,
             41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc40': [0, 4, 7, 10, 11, 12, 13, 15, 18, 19, 20, 21, 23, 24, 26, 27, 29, 30, 33, 34, 35, 36, 37, 39, 40, 42, 43,
             44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]
}

# apply ICA & plot signal overlay
for subj in subj_names:
    if subj in subj_excl:
        continue

    print(f"Processing participant {subj}")

    # load EEG data & ICA, downsample to 250 Hz
    cont = raw_dict[subj]
    cont.resample(250, npad='auto')
    ica = mne.preprocessing.read_ica(dir_eeg % subj + fname_ica % subj)

    # exclude components in excl_dict
    cont_ica = cont.copy()
    cont_ica = ica.apply(cont_ica, exclude=excl_dict[subj])

    # plot overlay of original and cleaned eeg signal
    ica.plot_overlay(cont, exclude=excl_dict[subj], show=False)
    plt.savefig(dir_ica + plt_overlay % subj)

    # save cleaned EEG data
    cont_ica.save(dir_eeg % subj + fname_filt_ica % subj, overwrite=True)

################################################################################
# 5) Epoch, add behavioural metadata, baseline

epo_dict = {}
behav = pd.read_csv(dir_behav + fname_behav)
gr_av = []

for subj in subj_names:
    if subj in subj_excl:
        continue

    if subj == 'yc22':
        print('Processing participant yc22...')
        # subset behav data frame to ID & select eeg data from raw_dict
        metadata = behav[behav.ID == 22]
        cont = raw_dict[subj]
        # get events and delete 2 events from array
        events, event_id = mne.events_from_annotations(cont)
        events = np.delete(events, 433, 0)
        events = np.delete(events, 432, 0)
        event_id = {'Stimulus/S  1': 3, 'Stimulus/S  2': 4}
        # epoch & save
        epochs = mne.Epochs(cont, events, event_id,
                            tmin=-0.4, tmax=0.5,
                            baseline=(-0.1, 0),
                            metadata=metadata,
                            preload=True)
        epochs.save(dir_eeg % subj + fname_epo % subj, overwrite=True)
        epo_dict[subj] = epochs

    elif subj == 'yc26':
        print('Processing participant yc26...')
        # subset behav data frame to ID & select eeg data from raw_dict
        metadata = behav[behav.ID == 26]
        cont = raw_dict[subj]
        # get events and delete 2 events from array
        events, event_id = mne.events_from_annotations(cont)
        events = np.delete(events, 577, 0)
        events = np.delete(events, 576, 0)
        event_id = {'Stimulus/S  1': 3, 'Stimulus/S  2': 4}
        # epoch & save
        epochs = mne.Epochs(cont, events, event_id,
                            tmin=-0.4, tmax=0.5,
                            baseline=(-0.1, 0),
                            metadata=metadata,
                            preload=True)
        epochs.save(dir_eeg % subj + fname_epo % subj, overwrite=True)
        epo_dict[subj] = epochs

    else:
        print('Processing participant %s...' % subj)
        # find index of subj in subj_names
        i = subj_names.index(subj)
        # subset behav data frame to current ID
        metadata = behav[behav.ID == i + 1]
        # select current participant from raw_dict
        cont = raw_dict[subj]
        # epochs, downsample, add behavioural metadata
        events, event_id = mne.events_from_annotations(cont)
        epochs = mne.Epochs(cont, events, event_id,
                            tmin=-0.4, tmax=0.5,
                            baseline=(-0.1, 0),
                            metadata=metadata,
                            preload=True)
        # save epochs
        epochs.save(dir_eeg % subj + fname_epo % subj, overwrite=True)
        epo_dict[subj] = epochs

#################################################################################################################
# 6) Behavioural pre-processing of epochs

# number of signal trials left per participant after behavioural exclusion:
trials = []
#['yc01', 'yc02', 'yc03', 'yc04', 'yc05', 'yc06', 'yc07', 'yc08', 'yc09', 'yc10', 'yc11', 'yc12', 'yc13', 'yc14', 'yc15', 'yc16', 'yc17', 'yc18', 'yc19', 'yc20', 'yc21', 'yc22', 'yc23', 'yc24', 'yc25', 'yc26', 'yc27', 'yc28', 'yc29', 'yc30', 'yc31', 'yc32', 'yc33', 'yc34', 'yc35', 'yc36', 'yc37', 'yc38', 'yc39', 'yc40']
#[394,     473,    470,    349,    88,             598,    0,      351,    126,    577,    304,    396,    489,    264,    118,    273,    194,    228,    446,    463,    507,    205,    199,    354,    340,    357,                    234,    191,    483,    598,    199,    591,    415,    182,    573,    465,    505]
for subj in subj_names:
    if subj in subj_excl:
        continue

    epochs = epo_dict[subj]

    # reject by response time & button press (filter column of metadata)
    epochs = epochs[epochs.metadata.resp_filter == 1]

    # create columns for  Detection Rates (F1 & F2, anatomical position), Detection Rate difference, overall False Alarm Rate
    epochs.metadata = epochs.metadata.assign(DR_F1_anat='NaN',
                                             DR_F2_anat='NaN',
                                             DR_diff='NaN',
                                             FAR_overall='NaN')

    # subset only anatomical trials
    anat_epo = epochs[epochs.metadata.block_type == 1]

    # iterate over blocks of current participant
    for i in np.unique(anat_epo.metadata.block):
        # count hits & misses in anatomical position
        hitsF1 = sum((anat_epo.metadata.block == i) & (anat_epo.metadata.det == 'hitF1'))
        hitsF2 = sum((anat_epo.metadata.block == i) & (anat_epo.metadata.det == 'hitF2'))
        # count false alarms overall
        FAs = sum((epochs.metadata.block == i) & (epochs.metadata.det == 'FA'))

        # calculate anatomical detection rate & detection rate difference, and overall FAR (NOT signal detection theory)
        DR_F1 = hitsF1 / sum((anat_epo.metadata.block == i) & (anat_epo.metadata.stim == 1))
        DR_F2 = hitsF2 / sum((anat_epo.metadata.block == i) & (anat_epo.metadata.stim == 2))
        Diff = abs(DR_F1 - DR_F2)
        FAR = FAs / sum((epochs.metadata.block == i) & (epochs.metadata.stim == 0))

        # assign calculated values to metadata
        cond = epochs.metadata['block'] == i
        epochs.metadata.loc[cond, 'DR_F1_anat'] = DR_F1
        epochs.metadata.loc[cond, 'DR_F2_anat'] = DR_F2
        epochs.metadata.loc[cond, 'DR_diff'] = Diff
        epochs.metadata.loc[cond, 'FAR_overall'] = FAR

    del DR_F1, DR_F2, Diff, FAR, FAs, hitsF1, hitsF2, anat_epo

    # reject blocks based on DR_F1_/F2_anat, DR_diff, FAR_overall
    epochs = epochs[(epochs.metadata.DR_F1_anat < 0.8) &
                    (epochs.metadata.DR_F1_anat > 0.2) &
                    (epochs.metadata.DR_F2_anat < 0.8) &
                    (epochs.metadata.DR_F2_anat > 0.2) &
                    (epochs.metadata.DR_diff < 0.4) &
                    (epochs.metadata.FAR_overall < 0.4)]

    signal_trials = epochs[epochs.metadata.stim != 0]
    trials.append(len(signal_trials))

    # re-order channels, so channel order is the same for all participants:
    epochs.reorder_channels(['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'ECG', 'CP5', 'CP1', 'Pz', 'P3',
                             'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'VEOG', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6',
                             'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'C1', 'C5', 'TP7',
                             'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8',
                             'C6', 'C2', 'FC4', 'FT8', 'F6', 'AF8', 'AF4', 'F2', 'Iz'])

    # save behaviourally pre-processed epochs
    epochs.save(dir_eeg % subj + fname_epo_behav % subj, overwrite=True)
    epo_dict[subj] = epochs
