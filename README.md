# EEG_Confusion_MOOC
Codes for running analysis on students' confusion levels via EEG-based deep neural network classsification. The leave-one-subject-and-video-out (LOSVO) paradigm is introduced here as a means to accurately simulate a plug-and-play implementation of the proposed classification framework into real-world classrooms.

## Results Overview

### Non-Normalized EEG Data
| Videos| 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | Average |
|-|-|-|-|-|-|-|-|-|-|-|-|
| Subject 1 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| 2 | 100.00 | 72.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 97.00 | 86.90 |
| 3 | 100.00 | 91.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 83.00 | 97.40 |
| 4 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 86.00 | 98.60 |
| 5 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| 6 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| 7 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 0.00 | 100.00 | 90.00 |
| 8 | 100.00 | 100.00 | 100.00 | 64.00 | 100.00 | 100.00 | 100.00 | 100.00 | 0.00 | 100.00 | 86.40 |
| 9 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00| 100.00 |
| 10 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 82.00 | 98.20 |
| **Average** | 100.00 | 96.30 | 100.00 | 96.40 | 100.00 | 100.00 | 100.00 | 90.00 | 80.00 | 94.80 | 95.75 |

### Normalized EEG Data
| Videos| 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | Average |
|-|-|-|-|-|-|-|-|-|-|-|-|
| Subject 1 | 62.00 | 100.00 | 100.00 | 48.00 | 65.00 | 100.00 | 100.00 | 95.00 | 100.00 | 100.00 | 87.00 |
| 2         | 95.00 | 100.00| 100.00 | 100.00 | 97.00 | 100.00 | 100.00 | 88.00 | 76.00 | 87.00 | 94.30 |
| 3         | 97.00 | 95.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 99.20 |
| 4         | 100.00 | 100.00 | 100.00 | 97.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 86.00 | 99.70 |
| 5         | 35.00 | 100.00 | 100.00 | 100.00 | 100.00 | 82.00 | 100.00 | 100.00 | 88.00 | 100.00 | 90.50 |
| 6         | 95.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 99.50 |
| 7         | 70.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 94.00 | 96.40 |
| 8         | 100.00 | 100.00 | 100.00 | 96.00 | 100.00 | 42.00 | 100.00 | 100.00 | 100.00 | 100.00 | 93.80 |
| 9         | 100.00 | 100.00 | 100.00 | 46.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00| 94.60 |
| 10        | 13.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 95.00 | 90.80 |
| **Average** | 76.70 | 99.50 | 100.00 | 88.70 | 96.20 | 92.40 | 100.00 | 98.30 | 96.40 | 97.60 | 94.58 |
## Resources
Dataset: [Link](https://www.kaggle.com/datasets/wanghaohan/confused-eeg)

EEGNet Framework: [Link](https://github.com/braindecode/braindecode)

## Dependences

## Run

Download the input folder from the dataset link.

```
usage: python LOSVO_DCN.py

Trains and Evaluates a binary classifier for students' confusion levels based on the EEGNet framework under the leave-one-subject-and-video-out (LOSVO) paradigm.

Optional Arguments:
    -vid VID                            Set the video ID which data will be removed from training the classifier
    -subj SUBJ                          Set the subject whose data will be removed from training the classifier

```

Alternatively, to obtain the LOSVO accuracy for all subjects for each individual video, run `LOSVO_DCN_all.py`.
