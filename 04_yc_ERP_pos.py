"""
author: Lena Lange
last update: 24.04.2023
Version: Python 3.9

- 2 finger (left ring and index finger) electrical stimulation near the perceptual threshold (50% detection rate)
- Y/N detection task, 2AFC localisation task
- anatomical and crossed hand position (left hand in right hemispace)

Plotting Grand Average ERPs in anatomical vs crossed

"""

import mne
import pandas as pd
import matplotlib.pyplot as plt
import pickle

dir_eeg = r'/data/pt_02438/numbtouch_neglect/young_controls/yc_eeg/%s/'
dir_plots = r'/data/pt_02438/numbtouch_neglect/young_controls/yc_eeg/plots/continuous/localisation/'
fname_epo_behav = r'%s_cont_behav_epo.fif'

subj_names = ['yc01', 'yc02', 'yc03', 'yc04', 'yc05', 'yc06', 'yc07', 'yc08', 'yc09', 'yc10', 'yc11', 'yc12', 'yc13',
              'yc14', 'yc15', 'yc16', 'yc17', 'yc18', 'yc19', 'yc20', 'yc21', 'yc22', 'yc23', 'yc24', 'yc25', 'yc26']
subj_excl = ['yc06', 'yc08']  # yc06: BufferOverflow error; yc08: no blocks survived exclusion

sensors = ['C4']
# ['AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4',
#  'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'AF8', 'AF4', 'F2', 'Iz', 'Fp1', 'Fz', 'F3',
#  'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'ECG', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'VEOG',
#  'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2']

# enter YN outcome (detected only - 1, undetected only - 2, all trials - 3)
det = 1

# enter 2AFC outcome (corr. localised - 1, incorr. localised - 2, both - 3)
loc = 3


#############################################################################################################
# Replace epochs metadata with modified data frame & save
#
# data frame containing behavioural data
#behav = pd.read_csv(dir_behav + fname_behav)
#
# enter subject ID and number of subj ID as integer:
#subj = 'yc01'
#i = 1
#
# subset behav data frame to current ID
#metadata = behav[behav.ID == i]
#
#epochs = mne.read_epochs(dir_eeg % subj + fname_epo_ica % subj, preload=True)
#epochs.metadata = metadata
#epochs.save(dir_eeg % subj + fname_epo_ica % subj, overwrite=True)


#############################################################################################################
# Compute Grand Average ERPs

# lists for Grand Averages
det_loc_anat = []
det_loc_cross = []

undet_loc_anat = []
undet_loc_cross = []

catch_anat = []
catch_cross = []

# check trial count per participant (before equalizing):
#trials_det_anat = [] #   [141, 169, 124, 230, 143, 200, 191, 182, 201, 128]
#trials_det_cross = [] #  [81, 101, 69, 130, 78, 113, 113, 112, 100, 79]
#trials_undet_anat = [] # [176, 155, 155, 177, 102, 126, 232, 155, 112, 130]
#trials_undet_cross = [] #[76, 74, 86, 84, 46, 80, 123, 92, 62, 66]

# check trial count per participant (after equalizing):
trials_det_anat = [] # [81, 101, 69, 130, 78, 113, 113, 112, 100, 79]
trials_det_cross = [] # [81, 101, 69, 130, 78, 113, 113, 112, 100, 79]
trials_undet_anat = [] # [76, 74, 86, 84, 46, 80, 123, 92, 62, 66]
trials_undet_cross = [] # [76, 74, 86, 84, 46, 80, 123, 92, 62, 66]


for subj in subj_names:

    # skip participants in subj_excl
    if subj in subj_excl:
        continue

    # read & load epochs
    epochs = mne.read_epochs(dir_eeg % subj + fname_epo_behav % subj, preload=True)

    # Select epochs (anatomical, crossed)

    epo_catch_anat = epochs[(epochs.metadata.stim == 0) & (epochs.metadata.block_type == 1)]
    epo_catch_cross = epochs[(epochs.metadata.stim == 0) & (epochs.metadata.block_type == 2)]

    # detected
    epo_det_loc_anat = epochs[(epochs.metadata.det == 'hitF1') | (epochs.metadata.det == 'hitF2') &
                              ((epochs.metadata.loca == 'hit') | (epochs.metadata.loca == 'CR')) &
                              (epochs.metadata.block_type == 1)]
    epo_det_loc_cross = epochs[((epochs.metadata.det == 'hitF1') | (epochs.metadata.det == 'hitF2')) &
                               ((epochs.metadata.loca == 'hit') | (epochs.metadata.loca == 'CR')) &
                               (epochs.metadata.block_type == 2)]

    # undetected
    epo_undet_loc_anat = epochs[(epochs.metadata.det == 'missF1') | (epochs.metadata.det == 'missF2') &
                              ((epochs.metadata.loca == 'hit') | (epochs.metadata.loca == 'CR')) &
                              (epochs.metadata.block_type == 1)]
    epo_undet_loc_cross = epochs[((epochs.metadata.det == 'missF1') | (epochs.metadata.det == 'missF2')) &
                               ((epochs.metadata.loca == 'hit') | (epochs.metadata.loca == 'CR')) &
                               (epochs.metadata.block_type == 2)]

    # Equalize number of epochs you want to plot against each other
    mne.epochs.equalize_epoch_counts([epo_det_loc_anat, epo_det_loc_cross])
    mne.epochs.equalize_epoch_counts([epo_undet_loc_anat, epo_undet_loc_cross])

    # Create Evokeds and append to lists for Grand Averages:
    # anatomical
    det_loc_anat.append(epo_det_loc_anat.average())
    undet_loc_anat.append(epo_undet_loc_anat.average())
    catch_anat.append(epo_catch_anat.average())

    # crossed
    det_loc_cross.append(epo_det_loc_cross.average())
    undet_loc_cross.append(epo_undet_loc_cross.average())
    catch_cross.append(epo_catch_cross.average())

    trials_det_anat.append(len(epo_det_loc_anat))
    trials_det_cross.append(len(epo_det_loc_cross))
    trials_undet_anat.append(len(epo_undet_loc_anat))
    trials_undet_cross.append(len(epo_undet_loc_cross))

# Compute Grand Averages and subtract Catch Grand Average
# anatomical
GrAv_det_loc_anat = mne.grand_average(det_loc_anat)
GrAv_undet_loc_anat = mne.grand_average(undet_loc_anat)
GrAv_catch_anat = mne.grand_average(catch_anat)

#GrAv_det_loc_anat = mne.combine_evoked([GrAv_det_loc_anat, GrAv_catch_anat], weights=[1, -1])

# crossed
GrAv_det_loc_cross = mne.grand_average(det_loc_cross)
GrAv_undet_loc_cross = mne.grand_average(undet_loc_cross)

GrAv_catch_cross = mne.grand_average(catch_cross)

#GrAv_det_loc_cross = mne.combine_evoked([GrAv_det_loc_cross, GrAv_catch_cross], weights=[1, -1])


#############################################################################################################
# Plot ERPs against each other

### ANATOMICAL vs CROSSED
# detected
fig = mne.viz.plot_compare_evokeds({'anatomical': GrAv_det_loc_anat,
                                    'crossed':GrAv_det_loc_cross},
                                   title='Grand Average: anatomical vs crossed (detected) %s' % sensors,
                                   picks=sensors,
                                   ylim=dict(eeg=[-4, 4]),
                                   legend='lower right')
filename = r'det_C4_n10.pickle'
pickle.dump(fig, open(dir_plots + filename, 'wb'))
# open pickled interactive plot:
#fig = pickle.load(open(dir_plots + filename, 'rb'))

fig = mne.viz.plot_compare_evokeds({'anatomical': det_loc_anat,
                                    'crossed': det_loc_cross},
                                   title='Grand Average: anatomical vs crossed (detected) %s' % sensors,
                                   picks=sensors,
                                   ylim=dict(eeg=[-4, 4]),
                                   legend='lower right',
                                   ci=True)
filename = r'det_ci_C4_n10.pickle'
pickle.dump(fig, open(dir_plots + filename, 'wb'))
# open pickled interactive plot:
#fig = pickle.load(open(dir_plots + filename, 'rb'))


# undetected
fig = mne.viz.plot_compare_evokeds({'anatomical': GrAv_undet_loc_anat,
                                    'crossed': GrAv_undet_loc_cross},
                                   title='Grand Average: anatomical vs crossed (undetected) %s' % sensors,
                                   picks=sensors,
                                   ylim=dict(eeg=[-4, 4]),
                                   legend='lower right')
filename = r'undet_C4_n10.pickle'
pickle.dump(fig, open(dir_plots + filename, 'wb'))
# open pickled interactive plot:
#fig = pickle.load(open(dir_plots + filename, 'rb'))

fig = mne.viz.plot_compare_evokeds({'anatomical': undet_loc_anat,
                                    'crossed': undet_loc_cross},
                                   title='Grand Average: anatomical vs crossed (undetected) %s' % sensors,
                                   picks=sensors,
                                   ylim=dict(eeg=[-4, 4]),
                                   legend='lower right',
                                   ci=True)
filename = r'undet_ci_C4_n10.pickle'
pickle.dump(fig, open(dir_plots + filename, 'wb'))
# open pickled interactive plot:
#fig = pickle.load(open(dir_plots + filename, 'rb'))

