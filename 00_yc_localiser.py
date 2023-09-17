"""
Analysis of localizer EEG data
ring finger supra-threshold electrical stimulation (passive, eyes open)

1. Pre-processing: interpolate, re-reference, filter (FIR bandpass 1 & 45Hz, notch 50Hz)
2. Downsampling to 250 Hz, ICA (extended algorithm, 55 PCs, ICLabel classification)
3. Epoching (from -0.1s to 0.5s to stimulus onset), baseline correction (-0.1s to 0s to stimulus onset)
4. Grand-average ERP in C4, CP4, C6, CP6

"""

import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from csv import writer
import pickle

# raw eeg data
dir_eeg = r'/data/pt_02438/numbtouch_neglect/young_controls/yc_eeg/%s/'
# ICA (ICA object, component plots etc)
dir_ica = r'/data/pt_02438/numbtouch_neglect/young_controls/yc_eeg/ica_loc/'
# save plots to
dir_plots = r'/data/pt_02438/numbtouch_neglect/young_controls/yc_eeg/plots/localizer/'

# raw eeg file name (%s = participant ID)
fname_loc = r'numbtouch_neglect_EEG_%s_loc.vhdr'
# for yc01, yc02, yc03 use this instead (had to change electrode names/positions):
fname_loc_switch = r'%s_loc_raw.fif'
# filtered data
fname_filt = r'%s_loc_filt_raw.fif'

# ICA object
fname_ica = r'%s_loc_ica.fif'
# ICs plot
plt_comp = r'%s_loc_components.pdf'
plt_prop = r'%s_loc_IC_properties.pdf'
plt_overlay = r'%s_loc_overlay.png'
# .csv file with calculated probability values from ICLabel
fname_csv = r'ICLabels_loc_%s.csv'
# ica corrected data
fname_filt_ica = r'%s_loc_filt_ica_raw.fif'

# epochs
fname_epo = r'%s_loc_epo.fif'
# cleaned epochs
fname_epo_ica = r'%s_loc_clean_epo.fif'

# list of participant IDs:
subj_names = ['yc01', 'yc02', 'yc03', 'yc04', 'yc05', 'yc06', 'yc07', 'yc08', 'yc09', 'yc10', 'yc11', 'yc12', 'yc13',
              'yc14', 'yc15', 'yc16', 'yc17', 'yc18', 'yc19', 'yc20', 'yc21', 'yc22', 'yc23', 'yc24', 'yc25', 'yc26',
              'yc27', 'yc28', 'yc29', 'yc30', 'yc31', 'yc32', 'yc33', 'yc34', 'yc35', 'yc36', 'yc37', 'yc38', 'yc39',
              'yc40']
subj_excl = ['yc08'] # yc08: no localizer recorded
subj_switch = ['yc01', 'yc02', 'yc03'] # electrode positions switched around




#############################################################################################################
# 1) Pre-processing: interpolate, re-reference, filter

eeg_dict = {}

for subj in subj_names:

    if subj in subj_excl:
        continue

    elif subj in subj_switch:
        eeg_dict[subj] = mne.io.read_raw_fif(dir_eeg % subj + fname_loc_switch % subj, preload=True)

    else:
        eeg_dict[subj] = mne.io.read_raw_brainvision(dir_eeg % subj + fname_loc % subj, preload=True)

    # read & load raw data
    raw = eeg_dict[subj]

    # set channel types for VEOG, ECG, TP7, CPz electrodes
    raw.set_channel_types({'VEOG': 'eog', 'ECG': 'ecg', 'TP7': 'eeg', 'CPz': 'eeg'}, verbose=False)

    # set montage
    ten_ten_montage = mne.channels.make_standard_montage('easycap-M1')
    raw.set_montage(ten_ten_montage)

    # exclude very noisy sensors & interpolate them
    raw.plot(n_channels=64)
    plt.show(block=True)

# Dictionary to store the bad channels for each subject ID
bads_dict = {
    'yc01': ['FT10'],
    'yc02': [],
    'yc03': ['AF7', 'Iz', 'T7', 'T8', 'TP7'],
    'yc04': ['AF7'],
    'yc05': ['Iz'],
    'yc06': ['FT10', 'AF7'],
    'yc07': [],
    'yc08': [],
    'yc09': [],
    'yc10': ['T7', 'T8', 'Iz'],
    'yc11': ['FT9', 'T7', 'P7'],
    'yc12': [],
    'yc13': [],
    'yc14': ['AF7', 'T7', 'TP7'],
    'yc15': ['Iz', 'T7', 'T8', 'TP7'],
    'yc16': [],
    'yc17': ['AF7'],
    'yc18': ['T7', 'TP7'],
    'yc19': ['CP5'], # ECG lost
    'yc20': ['FT10', 'TP7'],
    'yc21': [],
    'yc22': [],
    'yc23': ['F4'],
    'yc24': [],
    'yc25': ['FT10', 'FC6'],
    'yc26': ['T7', 'T8', 'TP7', 'TP8', 'FT7'],
    'yc27': ['T7', 'C5', 'TP7'],
    'yc28': [],
    'yc29': ['FT10'],
    'yc30': [],
    'yc31': ['T7', 'FC5', 'Iz'],
    'yc32': [],
    'yc33': ['T7', 'TP7'],
    'yc34': [],
    'yc35': ['FC1', 'C3', 'Iz'],
    'yc36': ['T7'],
    'yc37': ['T7', 'T8', 'Iz'],
    'yc38': [],
    'yc39': ['T7'],
    'yc40': ['T7', 'T8', 'TP7', 'Pz']
    # Add more subject IDs and bad channels if needed
}

for subj in subj_names:
    if subj in subj_excl:
        continue

    # interpolate bad channels
    raw = eeg_dict[subj]
    bads = bads_dict.get(subj, [])
    raw.info['bads'] = bads
    raw.interpolate_bads(reset_bads=False)

    # set reference
    raw.set_eeg_reference(ref_channels='average', ch_type='eeg')

    # highpass filter at 1 Hz to remove slow drifts, lowpass filter at 45 Hz
    bandpass = raw.copy().filter(l_freq=1, h_freq=45, method='fir')

    # notch filter at 50 Hz to remove power line artefacts
    picks = mne.pick_types(raw.info, eeg=True)
    freqs = (50, 100, 150, 200, 250)
    notch = bandpass.copy().notch_filter(freqs=freqs)

    # save concatenated filtered data, update eeg_dict
    notch.save(dir_eeg % subj + fname_filt % subj, overwrite=True)
    eeg_dict[subj] = notch


#############################################################################################################
# 2) Downsampling & ICA

# define function for saving ICs & properties in 1 PDF file per participant
def save_components(filename):
    p = PdfPages(filename)  # PdfPages is a wrapper around pdf file so there is no clash and create files with no error.
    for fig in figs:  # iterating over the numbers in list and saving the files
        fig.savefig(p, format='pdf')
    p.close()  # close the object


for subj in subj_names:
    if subj in subj_excl:
        continue

    print(f'Processing participant {subj}...')

    # load & resample to 250 Hz
    cont = eeg_dict[subj]
    cont.resample(250, npad='auto')

    # initialize ICA object with extended Infomax algorithm and 55 components
    ica = ICA(method='infomax',
              fit_params=dict(extended=True),
              n_components=55)

    # fit to continuous data & save ICA solution
    ica.fit(cont)
    ica.save(dir_eeg % subj + fname_ica % subj, overwrite=True)

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
excl_dict = {
    'yc01': [0, 1, 2, 3, 4, 10, 15, 22, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
             45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc02': [0, 2, 4, 7, 14, 20, 21, 23, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
             47, 48, 49, 50, 51, 52, 53, 54],
    'yc03': [0, 2, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
             32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc04': [0, 2, 4, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
             35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc05': [0, 1, 2, 3, 4, 8, 9, 11, 12, 13, 14, 15, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
             38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc06': [0, 1, 8, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
             45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc07': [6, 28, 29, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc08': [],
    'yc09': [0, 1, 2, 4, 10, 17, 22, 26, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
             48, 49, 50, 51, 52, 53, 54],
    'yc10': [0, 1, 3, 4, 5, 7, 11, 12, 16, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
             39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc11': [0, 1, 3, 11, 12, 13, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
             38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc12': [0, 2, 8, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
             37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc13': [0, 3, 4, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
             36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc14': [0, 1, 2, 8, 11, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
             40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc15': [0, 1, 2, 4, 6, 7, 9, 10, 11, 12, 13, 14, 16, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
             34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc16': [0, 9, 11, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
             38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc17': [0, 10, 13, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
             42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc18': [0, 15, 16, 17, 22, 24, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
             47, 48, 49, 50, 51, 52, 53, 54],
    'yc19': [0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
             31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc20': [0, 8, 9, 11, 14],
    'yc21': [0, 14, 16, 17, 19, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
             44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc22': [0, 2, 3, 4, 6, 11, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
             36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc23': [0, 1, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
             34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc24': [0, 1, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
             32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc25': [0, 1, 7, 8, 10, 11, 13, 14, 16, 18, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
             39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc26': [0, 1, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
             35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc27': [0, 9, 12, 13, 16, 17, 18, 19, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
             41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc28': [0, 1, 12, 14, 15, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
             43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc29': [0, 4, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
             37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc30': [0, 3, 5, 9, 12, 14, 15, 16, 17, 18, 20, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
             40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc31': [0, 1, 2, 13, 15, 16, 17, 18, 19, 20, 21, 22, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
             42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc32': [0, 1, 2, 3, 5, 6, 7, 8, 11, 13, 14, 16, 17, 18, 19, 20, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
             38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc33': [0, 2, 5, 12, 18, 19, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
             44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc34': [0, 4, 5, 12, 13, 16, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
             41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc35': [0, 3, 9, 10, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
             38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc36': [0, 29, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc37': [0, 1, 3, 11, 12, 13, 14, 15, 17, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
             38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc38': [0, 1, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
             32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc39': [0, 1, 2, 5, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
             33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'yc40': [0, 13, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]
# Add more IDs when needed
}

# apply ICA & plot signal overlay
for subj in subj_names:
    if subj in subj_excl:
        continue

    print(f"Processing participant {subj}")

    # load EEG data & ICA, downsample to 250 Hz
    cont = eeg_dict[subj]
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

#############################################################################################################
# 3) Epoch
eeg_dict = {}
epo_dict = {}

subj_excl = ['yc08']

for subj in subj_names:
    if subj in subj_excl:
        continue

    #eeg_dict[subj] = mne.io.read_raw_fif(dir_eeg % subj + fname_filt_ica % subj, preload=True)
    loc = eeg_dict[subj]

    # epochs, downsample to 250 Hz, add behavioural metadata
    events, event_id = mne.events_from_annotations(loc)
    events = events[2:]
    event_id = {'Stimulus/S  3': 3}
    epochs = mne.Epochs(loc, events, event_id,
                        tmin=-0.1, tmax=0.5,
                        baseline=(-0.1, 0),
                        preload=True)

    # save epochs & write into epo_dict
    #epochs.save(dir_eeg % subj + fname_epo % subj, overwrite=True)
    epo_dict[subj] = epochs


#######################################################################################################################
# Plot Grand Average ERP

sensor = ['C4', 'C6', 'CP4', 'CP6']
gr_av = []

for subj in subj_names:
    if subj in subj_excl:
        continue

    epochs = epo_dict[subj]
    gr_av.append(epochs.average())

GrAv = mne.grand_average(gr_av)
fig = mne.viz.plot_compare_evokeds(GrAv, picks=sensor, legend='lower right', ylim=dict(eeg=[-0.8, 0.8]))
filename = r'loc_ERP_C4.pickle'
pickle.dump(fig, open(dir_plots + filename, 'wb'))
# open pickled interactive plot:
#fig = pickle.load(open(dir_plots + filename, 'rb'))


"""
#####################################################################
# ICLabel ONLY exclusion

eeg_dict = {}
epo_dict = {}

for subj in subj_names:
    if subj in subj_excl:
        continue

    eeg_dict[subj] = mne.io.read_raw_fif(dir_eeg % subj + fname_filt % subj, preload=True)
    ica = mne.preprocessing.read_ica(dir_eeg % subj + fname_ica % subj)

    cont = eeg_dict[subj]
    cont.resample(250, npad='auto')

    # classify every IC & calculate probability values, shorten label names
    ic_labels = label_components(cont, ica, method="iclabel")
    ic_labels['labels'] = ['eye' if x == 'eye blink' else x for x in ic_labels['labels']]
    ic_labels['labels'] = ['muscle' if x == 'muscle artifact' else x for x in ic_labels['labels']]
    ic_labels['labels'] = ['channel' if x == 'channel noise' else x for x in ic_labels['labels']]
    ic_labels['labels'] = ['heart' if x == 'heart beat' else x for x in ic_labels['labels']]
    # extract labels & probability values
    labels = ic_labels["labels"]
    y_pred_proba = ic_labels["y_pred_proba"]
    # exclude all ICs that are NOT "brain" with >60% probability
    exclude_idx = [idx for idx, (label, prob) in enumerate(zip(labels, y_pred_proba))
                   if ((label not in "brain") & (prob > 0.6))]
    print(f"Excluding these ICA components: {exclude_idx}")
    cont_ica = ica.apply(cont, exclude=exclude_idx)

    # plot overlay of original and cleaned eeg signal
    ica.plot_overlay(cont_ica, exclude=exclude_idx, show=False)
    plt.savefig(dir_ica + 'auto/' + plt_overlay % subj)

    # save cleaned EEG data & update eeg_dict
    cont_ica.save(dir_eeg % subj + '%s_auto-ica-filt_raw.fif' % subj, overwrite=True)
    cont_ica = eeg_dict[subj]

    # epoch
    events, event_id = mne.events_from_annotations(cont_ica)
    events = events[2:]
    event_id = {'Stimulus/S  3': 3}
    epochs = mne.Epochs(cont_ica, events, event_id,
                        tmin=-0.1, tmax=0.5,
                        baseline=(-0.1, 0),
                        preload=True)

    # save epochs & write into epo_dict
    epochs.save(dir_eeg % subj + '%s_auto-ica_epo.fif' % subj, overwrite=True)
    epo_dict[subj] = epochs
"""



















