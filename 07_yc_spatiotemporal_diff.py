"""
author: Lena Lange
last update: 13.104.2022
Version: Python 3.9

from https://mne.tools/dev/auto_tutorials/stats-sensor-space/75_cluster_ftest_spatiotemporal.html

"Spatiotemporal permutation F-test on full sensor data:
    Tests for differential evoked responses in at least one condition using a permutation clustering test. The FieldTrip
    neighbor templates will be used to determine the adjacency between sensors. This serves as a spatial prior to the
    clustering. Spatiotemporal clusters will then be visualized using custom matplotlib code.
    Here, the unit of observation is epochs from a specific study subject. However, the same logic applies when the unit
    observation is a number of study subject each of whom contribute their own averaged data (i.e., an average of their
    epochs). This would then be considered an analysis at the “2nd level”."

-per condition (see list below) creates 2D array of shape (n_ch, n_times) for every participant
-stores these 2D arrays in a list to make a 3D matrix of shape (n_subjects, n_ch, n_times)

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.stats
import mne
from mne.stats import spatio_temporal_cluster_test, combine_adjacency
from mne.datasets import sample
from mne.channels import find_ch_adjacency
from mne.viz import plot_compare_evokeds
from mne.time_frequency import tfr_morlet

# enter time window (P50: 0-100ms; N140: 100-200ms; P300: 200-500ms)
tmin = 0.0
tmax = 0.5

subj_names = ['yc01', 'yc02', 'yc03', 'yc04', 'yc05', 'yc06', 'yc07', 'yc08', 'yc09', 'yc10', 'yc11', 'yc12', 'yc13',
              'yc14', 'yc15', 'yc16', 'yc17', 'yc18', 'yc19', 'yc20', 'yc21', 'yc22', 'yc23', 'yc24', 'yc25', 'yc26',
              'yc27', 'yc28', 'yc29', 'yc30', 'yc31', 'yc32', 'yc33', 'yc34', 'yc35', 'yc36', 'yc37', 'yc38', 'yc39',
              'yc40']
subj_excl = ['yc06', 'yc08', 'yc28', 'yc29']
# yc06: BufferOverflow error
# yc08: no blocks survived exclusion
# yc28: weird boundaries
# yc29: too many noisy channels

sensors = ['AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz',
           'PO4', 'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'AF8', 'AF4', 'F2', 'Iz',
           'Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz',
           'O2', 'P4', 'P8', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2']

dir_eeg = r'/data/pt_02438/numbtouch_neglect/young_controls/yc_eeg/%s/'
dir_plots = r'/data/pt_02438/numbtouch_neglect/young_controls/yc_eeg/plots/newfilt/spatiotemporal_strictICA/'
fname_epo = r'%s_newfilt_behav_epo.fif'

epo_dict = {}
# read & load epochs of all participants, store in dictionary epo_dict
for subj in subj_names:
    if subj in subj_excl:
        continue
    epo_dict[subj] = mne.read_epochs(dir_eeg % subj + fname_epo % subj, preload=True)

# six empty lists, one per condition
evokeds_for_GrAv = [[], [], [], []]  # 3D matrices
matrices_for_matrix = [[], [], [], []]  # 2D arrays

for subj in subj_names:
    if subj in subj_excl:
        continue

    # read epochs, crop, drop EOG/ECG
    epochs = epo_dict[subj]
    epochs = epochs.crop(tmin=tmin, tmax=tmax)
    epochs = epochs.pick(sensors)

    # Select epochs:
    # anatomical
    epo_det_anat = epochs[(((epochs.metadata.det == 'hitF1') | (epochs.metadata.det == 'hitF2')) &
                          (epochs.metadata.block_type == 1))]
    epo_undet_anat = epochs[(((epochs.metadata.det == 'missF1') | (epochs.metadata.det == 'missF2')) &
                            (epochs.metadata.block_type == 1))]

    # crossed
    epo_det_cross = epochs[(((epochs.metadata.det == 'hitF1') | (epochs.metadata.det == 'hitF2')) &
                           (epochs.metadata.block_type == 2))]
    epo_undet_cross = epochs[(((epochs.metadata.det == 'missF1') | (epochs.metadata.det == 'missF2')) &
                             (epochs.metadata.block_type == 2))]

    # equalize epoch count between conditions you want to plot against each other
    mne.epochs.equalize_epoch_counts(epo_det_anat, epo_undet_anat)
    mne.epochs.equalize_epoch_counts(epo_det_cross, epo_undet_cross)

    # average epoch object for each condition & append to list for GrAv
    evokeds_for_GrAv[0].append(epo_det_anat.average)
    evokeds_for_GrAv[1].append(epo_undet_anat.average)
    evokeds_for_GrAv[2].append(epo_det_cross.average)
    evokeds_for_GrAv[3].append(epo_undet_cross.average)

    # create 2D array (n_ch, n_times) for each condition & append to list for 3D matrix
    subj_matrix_1 = epo_det_anat.average().get_data()
    subj_matrix_2 = epo_undet_anat.average().get_data()
    subj_matrix_3 = epo_det_cross.average().get_data()
    subj_matrix_4 = epo_undet_cross.average().get_data()

    matrices_for_matrix[0].append(subj_matrix_1)
    matrices_for_matrix[1].append(subj_matrix_2)
    matrices_for_matrix[2].append(subj_matrix_3)
    matrices_for_matrix[3].append(subj_matrix_4)


Diff1 = mne.combine_evoked([mne.grand_average(evokeds_for_GrAv[cond1]), mne.grand_average(evokeds_for_GrAv[cond2])],
                           weights=[1, -1])
Diff2 = mne.combine_evoked([mne.grand_average(evokeds_for_GrAv[cond3]), mne.grand_average(evokeds_for_GrAv[cond4])],
                           weights=[1, -1])

# For each Difference Wave obtain data as a 3D matrix (from list of 2D matrices)
Evokeds = [np.array(Diff1), np.array(Diff2)]

# Transpose it such that the dimensions are as expected for the cluster permutation test (n_subj × n_times × n_ch)
X = [np.transpose(evoked, (0, 2, 1)) for evoked in Evokeds]

# Find the FieldTrip neighbor definition to setup sensor adjacency
adjacency, ch_names = find_ch_adjacency(epochs.info, ch_type='eeg')
# print(type(adjacency))  # it's a sparse matrix!
# mne.viz.plot_ch_adjacency(epochs.info, adjacency, ch_names)

# Compute permutation statistic
# We are running an F test, so we look at the upper tail
# see also: https://stats.stackexchange.com/a/73993
# "F tests are most commonly used for two purposes: 1. in ANOVA, for testing equality of means
#                                                   2. in testing equality of variances"
tail = 1

# We want to set a critical test statistic (here: F), to determine when clusters are being formed. Using Scipy's percent point function of
# the F distribution, we can conveniently select a threshold that corresponds to some alpha level that we arbitrarily pick.
alpha_cluster_forming = 0.005

# For an F test we need the degrees of freedom for the numerator
# (number of conditions - 1) and the denominator (number of observations
# - number of conditions):
n_conditions = 2
n_observations = len(subj_names)
dfn = n_conditions - 1
dfd = n_observations - n_conditions

# Note: we calculate 1 - alpha_cluster_forming to get the critical value
# on the right tail
f_thresh = scipy.stats.f.ppf(1 - alpha_cluster_forming, dfn=dfn, dfd=dfd)

# run the cluster based permutation analysis
cluster_stats = spatio_temporal_cluster_test(X, n_permutations=1000,
                                             threshold=f_thresh, tail=tail,
                                             n_jobs=None, buffer_size=None,
                                             adjacency=adjacency)
F_obs, clusters, p_values, _ = cluster_stats

del adjacency, ch_names, tail, alpha_cluster_forming, n_conditions, n_observations, dfn, dfd, f_thresh, cluster_stats, \
    _, Evokeds, X, evokeds_for_GrAv, matrices_for_matrix, subj_names

# Visualize clusters
# We subselect clusters that we consider significant at an arbitrarily picked alpha level: "p_accept".
# NOTE: remember the caveats with respect to "significant" clusters that we mentioned in the introduction of this tutorial!
p_accept = 0.05
good_cluster_inds = np.where(p_values < p_accept)[0]

# configure variables for visualization
colors = {cond_names[cond1]: 'crimson', cond_names[cond2]: 'crimson'}
linestyles = {cond_names[cond1]: '-', cond_names[cond2]: '--'}

# organize data for plotting (this has to be a dict with string:evoked pairs)
GrAvs_dict = {cond_name: GrAv for cond_name, GrAv in zip(cond_names, GrAvs)}
# slice dict for wanted conditions:
GrAvs_dict = {key: GrAvs_dict[key] for key in conds_to_plot}

# loop over clusters
for i_clu, clu_idx in enumerate(good_cluster_inds):

    # unpack cluster information (for timepoints and sensors):
    time_inds, space_inds = np.squeeze(clusters[clu_idx])
    # get unique indices from unpacked arrays:
    ch_inds = np.unique(space_inds)
    time_inds = np.unique(time_inds)

    # get topography for F stat
    # (= calculating mean F value (of F values at sign. timepoints) for each channel)
    f_map = F_obs[time_inds, ...].mean(axis=0)

    # get signals at the sensors contributing to the cluster
    # (= get list of significant timepoints in seconds)
    sig_times = epochs.times[time_inds]

    # create spatial mask
    # 1. create array of shape (n_channels, 1) with all values=False
    # 2. mark significant sensors True
    mask = np.zeros((f_map.shape[0], 1), dtype=bool)
    mask[ch_inds, :] = True

    # initialize figure
    fig, ax_topo = plt.subplots(1, 1, figsize=(10, 3))

    # plot average test statistic and mark significant sensors:
    # f_map.shape: (102,)
    # f_map[:, np.newaxis].shape: (102, 1) --> have all values in 1 column
    f_evoked = mne.EvokedArray(f_map[:, np.newaxis], epochs.info, tmin=0)
    f_evoked.plot_topomap(times=0, mask=mask, axes=ax_topo, cmap='Reds',
                          vmin=np.min, vmax=np.max, show=False,
                          colorbar=False, mask_params=dict(markersize=10))
    image = ax_topo.images[0]

    # remove the title that would otherwise say "0.000 s"
    ax_topo.set_title("")

    # create additional axes (for ERF and colorbar)
    divider = make_axes_locatable(ax_topo)

    # add axes for colorbar
    ax_colorbar = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(image, cax=ax_colorbar)
    ax_topo.set_xlabel(
        'Averaged F-map ({:0.3f} - {:0.3f} s)'.format(*sig_times[[0, -1]]))

    # add new axis for time courses and plot time courses
    ax_signals = divider.append_axes('right', size='300%', pad=1.2)
    title = 'Cluster #{0}, {1} sensor'.format(i_clu + 1, len(ch_inds))
    if len(ch_inds) > 1:
        title += "s (mean)"
    plot_compare_evokeds(GrAvs_dict, title=title, picks=ch_inds, axes=ax_signals,
                         colors=colors, linestyles=linestyles, show=False,
                         split_legend=True, truncate_yaxis='auto')

    # plot temporal cluster extent
    ymin, ymax = ax_signals.get_ylim()
    ax_signals.fill_betweenx((ymin, ymax), sig_times[0], sig_times[-1],
                             color='orange', alpha=0.3)

    # clean up viz
    mne.viz.tight_layout(fig=fig)
    fig.subplots_adjust(bottom=.05)
    plt.show()

    # save
    plt.savefig(dir_plots + filename + '_%s.png' % clu_idx)
    plt.close('all')
