# HeartBeat

This is a project where we seek to classify sound files containing human heartbeats.


# Data:

The data is provided by: https://www.kaggle.com/kinguistics/heartbeat-sounds

It has two sets: A and B. A is gathered by an iPhone app ( iStethoscope Pro ), and B is gather from clinical trials.

There are 3 .csv files.

* set_a.csv and set_b.csv contain the classes for each file

* set_a_timing contains the segmantation for different sounds during a heartbeat.


# Current state:

The sound samples are now being cut into smaller pieces before being feed into the network. These samples are extracted using a sliding window scheme.

# Plans:

* Train the LSTM and check performance

* Try out a convnet version
