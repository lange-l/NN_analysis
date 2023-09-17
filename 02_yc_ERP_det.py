"""
author: Lena Lange
last update: 24.04.2023
Version: Python 3.9

- 2 finger (left ring and index finger) electrical stimulation near the perceptual threshold (50% detection rate)
- Y/N detection task, 2AFC localisation task
- anatomical and crossed hand position (left hand in right hemispace)

Plotting Grand Average ERPs of Hits/Misses/CRs/FAs in the Y/N detection task

"""


import mne
import pandas as pd
import matplotlib.pyplot as plt
import pickle

dir_eeg = r'/data/pt_02438/numbtouch_neglect/young_controls/yc_eeg/%s/'
dir_plots = r'/data/pt_02438/numbtouch_neglect/young_controls/yc_eeg/plots/newfilt/'
#fname_epo = r'%s_newfilt_noica_epo.fif'
fname_epo = r'%s_newfilt_behav_epo.fif'

subj_names = ['yc01', 'yc02', 'yc03', 'yc04', 'yc05', 'yc06', 'yc07', 'yc08', 'yc09', 'yc10', 'yc11', 'yc12', 'yc13',
              'yc14', 'yc15', 'yc16', 'yc17', 'yc18', 'yc19', 'yc20', 'yc21', 'yc22', 'yc23', 'yc24', 'yc25', 'yc26',
              'yc27', 'yc28', 'yc29', 'yc30', 'yc31', 'yc32', 'yc33', 'yc34', 'yc35', 'yc36', 'yc37', 'yc38', 'yc39',
              'yc40']
subj_excl = ['yc06', 'yc08', 'yc28', 'yc29']
# yc06: BufferOverflow error
# yc08: no blocks survived exclusion
# yc28: weird boundaries
# yc29: too many noisy channels

# enter hand position (anatomical - 1, crossed - 2, overall - 3)
pos = 2

#n_trials_anat:  [95, 95, 118, 84, 14, 127, 78, 25, 131, 63, 81, 126, 68, 26, 63, 39, 39, 108, 104, 127, 42, 33, 85, 69, 76, 52, 39, 119, 125, 51, 139, 100, 43, 138, 112, 88]
# participant ID: 01  02   03  04  05  07   09  10   11  12  13   14  15  16  17  18  19   20   21   22  23  24  25  26  27  30  31  32   33   34  35   36   37  38   39   40
#n_trials_cross: [78, 95, 109, 66, 16, 149, 86, 28, 109, 56, 90, 101, 57, 21, 61, 30, 35, 109, 111, 118, 46, 44, 73, 62, 65, 52, 23, 120, 103, 47, 135, 99,  41, 129, 89,  93]


# enter ERPs to plot (hits - 0, misses - 1, CRs - 2, FAs - 3, hits & CRs - 4, misses & FAs - 5)
cond1 = 0
cond2 = 1


#############################################################################################################

epo_dict = {}
# read & load epochs of all participants, store in dictionary epo_dict
for subj in subj_names:
    if subj in subj_excl:
        continue
    epo_dict[subj] = mne.read_epochs(dir_eeg % subj + fname_epo % subj, preload=True)


plot_titles = ['Hits', 'Misses', 'CRs', 'FAs', 'Hits/CRs', 'Misses/FAs']
evokeds_for_GrAv = [[], [], [], [], [], []] # lists for grand averaging (hits, misses, CRs, FAs, hits/CRs, misses/FAs)
n_trials = []

for subj in subj_names:
    if subj in subj_excl:
        continue

    epochs = epo_dict[subj]

    if pos == 1:
        position = "anatomical"
        conditions = [
            epochs[((epochs.metadata.det == 'hitF1') | (epochs.metadata.det == 'hitF2')) &
                   (epochs.metadata.block_type == 1)],  # hits
            epochs[((epochs.metadata.det == 'missF1') | (epochs.metadata.det == 'missF2')) &
                   (epochs.metadata.block_type == 1)],  # misses
            epochs[(epochs.metadata.det == 'CR') & (epochs.metadata.block_type == 1)],  # CRs
            epochs[(epochs.metadata.det == 'FA') & (epochs.metadata.block_type == 1)],  # FAs
            epochs[(((epochs.metadata.det == 'hitF1') | (epochs.metadata.det == 'hitF2')) |
                    (epochs.metadata.det == 'CR')) & (epochs.metadata.block_type == 1)],  # hits & CRs
            epochs[(((epochs.metadata.det == 'missF1') | (epochs.metadata.det == 'missF2')) |
                    (epochs.metadata.det == 'FA')) & (epochs.metadata.block_type == 1)]  # misses & FAs
        ]

    elif pos == 2:
        position = "crossed"
        conditions = [
            epochs[((epochs.metadata.det == 'hitF1') | (epochs.metadata.det == 'hitF2')) &
                   (epochs.metadata.block_type == 2)],  # hits
            epochs[((epochs.metadata.det == 'missF1') | (epochs.metadata.det == 'missF2')) & (epochs.metadata.block_type == 2)],  # misses
            epochs[(epochs.metadata.det == 'CR') & (epochs.metadata.block_type == 2)],  # CRs
            epochs[(epochs.metadata.det == 'FA') & (epochs.metadata.block_type == 2)],  # FAs
            epochs[(((epochs.metadata.det == 'hitF1') | (epochs.metadata.det == 'hitF2')) |
                    (epochs.metadata.det == 'CR')) & (epochs.metadata.block_type == 2)],  # hits & CRs
            epochs[(((epochs.metadata.det == 'missF1') | (epochs.metadata.det == 'missF2')) |
                    (epochs.metadata.det == 'FA')) & (epochs.metadata.block_type == 2)]  # misses & FAs
        ]

    else:
        position = "overall"
        conditions = [
            epochs[((epochs.metadata.det == 'hitF1') | (epochs.metadata.det == 'hitF2'))],  # hits
            epochs[((epochs.metadata.det == 'missF1') | (epochs.metadata.det == 'missF2'))], # misses
            epochs[(epochs.metadata.det == 'CR')],   # CRs
            epochs[(epochs.metadata.det == 'FA')],   # FAs
            epochs[((epochs.metadata.det == 'hitF1') | (epochs.metadata.det == 'hitF2')) |
                   (epochs.metadata.det == 'CR')], # hits & CRs
            epochs[((epochs.metadata.det == 'missF1') | (epochs.metadata.det == 'missF2')) |
                   (epochs.metadata.det == 'FA')]   # misses & FAs
        ]
    print('Processing participant %s' % subj)
    if (len(conditions[cond1]) != 0) & (len(conditions[cond2]) != 0):
        # equalize epoch count between conditions you want to plot against each other
        mne.epochs.equalize_epoch_counts([conditions[cond1], conditions[cond2]])
        # count trials that are left
        n_trials.append(len(conditions[cond1]))
        # average epoch object for each condition & append to list for GrAv
        av_epochs = [epochs.average() for epochs in conditions]
        for cond, average in enumerate(av_epochs):
            evokeds_for_GrAv[cond].append(average)
    else:
        print('Skipping participant %s because no trials in one of the conditions' % subj)

# For each condition get Grand Average
GrAvs = [mne.grand_average(evoked) for evoked in evokeds_for_GrAv]
print('Number of included participants: ' + str(len(evokeds_for_GrAv[0])))

sensors = ['C4', 'C6', 'CP4', 'CP6']
# ['AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4',
#  'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'AF8', 'AF4', 'F2', 'Iz', 'Fp1', 'Fz', 'F3',
#  'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'ECG', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'VEOG',
#  'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2']

fig = mne.viz.plot_compare_evokeds({plot_titles[cond1]: evokeds_for_GrAv[cond1], # for CIs enter "evokeds_for_GrAv", otherwise "GrAvs"
                                    plot_titles[cond2]: evokeds_for_GrAv[cond2]},
                                   title='Grand Average: %s vs %s in Detection Task (%s)'
                                         % (plot_titles[cond1], plot_titles[cond2], position),
                                   picks=sensors,
                                   legend='lower right',
                                   ylim=dict(eeg=[-2.0, 2.0]),
                                   ci=True)

# pickle interactive plot
#filename = r'anat_n10.pickle'
#pickle.dump(fig, open(dir_plots + filename, 'wb'))
# open pickled interactive plot:
#fig = pickle.load(open(dir_plots + filename, 'rb'))


##################################################################################################################
# Compare Difference Waves:

anat_hits = evokeds_for_GrAv[0]
anat_misses = evokeds_for_GrAv[1]
anat_diff = mne.combine_evoked([mne.grand_average(anat_hits), mne.grand_average(anat_misses)], weights=[1,-1])

cross_hits = evokeds_for_GrAv[0]
cross_misses = evokeds_for_GrAv[1]
cross_diff = mne.combine_evoked([mne.grand_average(cross_hits), mne.grand_average(cross_misses)], weights=[1,-1])

fig = mne.viz.plot_compare_evokeds({'hits': anat_hits, # for CIs enter "evokeds_for_GrAv", otherwise "GrAvs"
                                    'misses': anat_misses,
                                    'difference wave': anat_diff},
                                   title='Grand Average: Hits vs Misses in Detection Task (anatomical)',
                                   picks=sensors,
                                   legend='lower right',
                                   ylim=dict(eeg=[-3, 3]),
                                   ci=True)

fig = mne.viz.plot_compare_evokeds({'hits': cross_hits, # for CIs enter "evokeds_for_GrAv", otherwise "GrAvs"
                                    'misses': cross_misses,
                                    'difference wave': cross_diff},
                                   title='Grand Average: Hits vs Misses in Detection Task (crossed)',
                                   picks=sensors,
                                   legend='lower right',
                                   ylim=dict(eeg=[-3, 3]),
                                   ci=True)

fig = mne.viz.plot_compare_evokeds({'anatomical': anat_diff, # for CIs enter "evokeds_for_GrAv", otherwise "GrAvs"
                                    'crossed': cross_diff},
                                   title='Difference Waves: Hits minus Misses in Detection Task',
                                   picks=sensors,
                                   legend='lower right',
                                   ylim=dict(eeg=[-3, 3]),
                                   ci=True)


n_trials_anat = [95, 95, 118, 84, 14, 0, 127, 0, 78, 25, 131, 63, 81, 126, 68, 26, 63, 39, 39, 108, 104, 127, 42, 33, 85, 69, 76, 0, 0, 52, 39, 119, 125, 51, 139, 100, 43, 138, 112, 88]
#participant ID: 01  02   03  04  05  06 07   08 09  10   11  12  13   14  15  16  17  18  19   20   21   22  23  24  25  26  27 28  29 30  31  32   33   34  35   36   37  38   39   40
n_trials_cross = [78, 95, 109, 66, 16, 0, 149, 0, 86, 28, 109, 56, 90, 101, 57, 21, 61, 30, 35, 109, 111, 118, 46, 44, 73, 62, 0, 0, 65, 52, 23, 120, 103, 47, 135, 99,  41, 129, 89,  93]
n_trials_all =  [184, 190, 227, 150, 30, 0, 277, 0, 164, 53, 240, 119, 171, 231, 128, 53, 124, 69, 74, 217, 218, 245, 88, 77, 165, 131, 141, 0, 0, 116, 62, 239, 228, 99, 293, 199, 84, 267, 201, 181]

import matplotlib.pyplot as plt
import numpy as np

IDs = np.arange(40)

width = 0.2

# plot data in grouped manner of bar type
plt.bar(IDs - 0.2, n_trials_anat, width)
plt.bar(IDs, n_trials_cross, width)
plt.bar(IDs + 0.2, n_trials_all, width)
#plt.xticks(IDs, ['Team A', 'Team B', 'Team C', 'Team D', 'Team E'])
plt.xlabel("participants")
plt.ylabel("number of trials")
plt.legend(["anatomical", "crossed", "overall"])
plt.axhline(y = 100, color = 'r', linestyle = '--')
plt.xticks(ticks=IDs, labels=IDs+1)
plt.viridis()
plt.show()
