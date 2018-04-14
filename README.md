# HeartBeat

This is a project where we seek to classify sound files containing human heartbeats.


# Data:

The data is provided by: https://www.kaggle.com/kinguistics/heartbeat-sounds

It has two sets: A and B. A is gathered by an iPhone app ( iStethoscope Pro ), and B is gather from clinical trials.

There are 3 .csv files.

* set_a.csv and set_b.csv contain the classes for each file

* set_a_timing contains the segmantation for different sounds during a heartbeat.


# Branch Description:

In this branch, the data is a neighbourhood of values around a particular sound in the recordings. These sounds are then transformed from a 1-D signal to a 2D signal using a melspectrogram. This, in essence, transforms th sound classification task into a image classification task. A Convnet would then be trained on this data. After the convnet is sufficiently classified, it will be iteratively applied to windows to segment the recordings.
