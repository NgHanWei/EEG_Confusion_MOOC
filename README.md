# EEG_Confusion_MOOC
Codes for running analysis on students' confusion levels via EEG-based deep neural network classsification. The leave-one-subject-and-video-out (LOSVO) paradigm is introduced here as a means to accurately simulate a plug-and-play implementation of the proposed classification framework into real-world classrooms.

## Results Overview

## Resources
Dataset:

EEGNet:

## Dependences

## Run
```
usage: python LOSVO_DCN.py

Obtain list of subjects to use as training data for subject-indepdendent baseline model

Positional Arguments:
    DATAPATH                            Datapath for the pre-processed EEG signals file

Optional Arguments:
    -start START                        Set start of range for subjects, minimum 1 and maximum 54
    -end END                            Set end of range for subjects, minimum 2 and maximum 55
    -subj SUBJ                          Set the subject number to run feature extraction on, will override the -start and -end functions if used
    -trial TRIAL                        Set the number of test trials from target subject to create baseline. Set number of trials to 0 to use target validation data
```

run `LOSVO_DCN_all.py` to run all subjects.
