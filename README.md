# NN_eeg_analysis
Analysis of electrophysiological data 

## Pre-processing & Analysis of EEG data from passive supra-threshold localiser sequence (00_yc_localiser.py)

- visual inspection for bad channels, interpolation
- re-reference to average
- Hamming-windowed FIR bandpass filter (1 Hz and 45 Hz) and notch filter (50 Hz)
- downsampling to 250 Hz
- ICA (extended, 55 PCs), classification with ICLabel (reject anything other than “brain” or “other” with >60% probability, include "brain" with >60% probability, visual confirmation)
- epoch to trigger (-100 ms - 500 ms), baseline correct (-100 ms - 0 ms to trigger)

- plot grand-average somatosensory evoked potential (SEP) in C4, CP4, C6 and CP6

## Pre-processing of EEG data from experiemental blocks (01_yc_prepro.py)

- blocks are cropped to 1 s before first and 2 s after last trigger
- concatenate blocks per ID to obtain 1 continuous .fif file per ID
- visual inspection for bad channels, interpolation
- re-referencing to average
- Hamming-windowed FIR bandpass filter (1 Hz and 45 Hz) and notch filter (50 Hz)
- downsampling to 250 Hz
- ICA (extended infomax, passing 55 PCs), classification with ICLabel (reject anything other than “brain” or “other” with >60% probability, include "brain" with >60% probability, visual confirmation)
- epoch to -400 ms - 500 ms to trigger, add behavioural metadata, baseline correction (-100 - 0 ms to trigger)
- reject epochs based on behavioual pre-processing criteria: FAR > 40%, DR_F1/_F2 >80% or <20%, difference between DR_F1 and DR_F2 >40%

## ERPs & difference waves

- detected trials against undetected trials (02_yc_ERP_det.py)
- undetected correctly localsied trials against undetected incorrectly localised trials (03_yc_ERP_loc.py)
- detected anatomical trials against detected crossed trials (04_yc_ERP_pos.py)

## spatiotemporal permutation F-tests

- detected vs undetected (05_yc_spatiotemporal_det.py)
- undetected correctly vs incorrectly localised (06_yc_spatiotemporal_loc.py)
- detected anatomical vs crossed (07_yc_spatiotemporal_diff.py)
