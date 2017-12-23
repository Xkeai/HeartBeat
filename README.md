# HeartBeat

This is a project where we seek to classify sound files containing human heartbeats.


# Data:

The data is provided by: https://www.kaggle.com/kinguistics/heartbeat-sounds

It has two sets: A and B. A is gathered by an iPhone app ( iStethoscope Pro ), and B is gather from clinical trials.

There are 3 .csv files.

* set_a.csv and set_b.csv contain the classes for each file

* set_a_timing contains the segmantation for different sounds during a heartbeat.


# Plans:

Now that the NaN loss issue is resolved, I can try a few things.

The current network is an LSTM whose output is passed through a single layer neural network to reduce its size. Unforunately, as of today (23.12.2017), I do not have access to machine powerful enough to train my network. So I do not know the full performance of this network. In the coming weeks, I will look into cloud. 

What can be done:

* Increase the size of the LSTM.

* Increase the size of the single-layer neurons.

* Filter the data (with the possibility of filtering using a trainable convnet)

* A seperate network in Fourier space
