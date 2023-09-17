# NN_analysis
Analysis of behavioural & electrophysiological data from an experiment about somatosensory perception. Near-threshold intensity paradigm, stimulating 2 fingers of the same hand, Yes/No detection and 2-alternative-forced-choice localisation task. 

## Experiment

Electrical stimuli are applied at near-threshold intensity (50% detection rate) to the index (F2) or ring finger (F1) of the left hand. Only 1 finger at a time is stimulated. Each trial contains a visual cue & a stimulus, followed by a Y/N detection task ("Did you feel the stimulus? Yes or No") and 2AFC localisation task ("Where was the stimulus applied? Ring or index finger"). Every participant completed 5 experimental blocks. All blocks had the same number of trials, trial numbers between ring and index finger were balanced, order of stimulation was randomised. Every 12 trials participants adjusted the position of the left hand (anatomical or crossed). Within every sub-block there were 2 catch trials where no stimulation was applied. In addition to the 5 experimental blocks, every participants completes a passive localiser sequence where F1 is stimulated 100 times at suprathreshold intensity. EEG & ECG data were recorded.

## Behavioural analysis

Every trials is given two labels according to signal detection theory (hit/miss/false alarm/correct rejection), one for the Y/N task, one for the 2AFC task. Based on these labels we calculate performance, sensitivity d' and response bias c for every block individually (in both tasks). Detection performance, detection sensitivity d' and detection response bias c are calculated for hand position × finger (overall/anatomical/crossed × both/F1/F2). Localisation performance, localisation sensitivity d' and localisation response bias c are calculated for detection × hand position (overall/detected only/undetected only × overall/anatomical/both). d’ and c are calculated with log-linear correction (Hautus, 1995). 

- detection performance = (hits + correct rejections) / total number of trials
- localisation performance = (hits + correct rejections) / total number of stimulation trials
- d' = zHR - zFAR
- c = -0.5*(zHR + zFAR)
- (see https://mvuorre.github.io/posts/2017-10-09-bayesian-estimation-of-signal-detection-theory-models/#ref-macmillan_detection_2005) 

We investigate
- the effect of finger (F1 or F2) and hand position (anatomical or crossed) on detection performance, d' and c
- the effect of detection (detected or undetected) and hand position (anatomical or crossed) on localisation performance, d' and c
- if there is above-chance correct localisation in undetected trials (loc_perf > 50 %, d' > 0)

We compared the means of within-subject averages of performance scores and SDT measures between conditions. 
We pooled blocks of all participants together and compared the means of performance scores and SDT measures of the pooled blocks between conditions.

## EEG analysis

We calculate the grand-average ERPs for 
- the passive supra-threshold localiser sequence
- near-threshold detected, undetected, undetected correctly localized, and undetected incorrectly localized trials (across both hand positions)
- near-threshold detected trials in anatomical hand position and in crossed hand position

ERPs are contrasted using a spatiotemporal permutation F-Test to identify clusters of significant difference in activity between conditions:
- detected vs undetected
- undetected correctly localised vs undetected incorrectly localised
- detected anatomical vs detected crossed
