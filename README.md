# HeartBeat

This is a project where we seek to classify sound files containing human heartbeats.


# Data:

The data is provided by: https://www.kaggle.com/kinguistics/heartbeat-sounds

It has two sets: A and B. A is gathered by an iPhone app ( iStethoscope Pro ), and B is gather from clinical trials.

There are 3 .csv files.

* set_a.csv and set_b.csv contain the classes for each file

* set_a_timing contains the segmantation for different sounds during a heartbeat.


# Current state:

I approached the timing task as a segmentation tasks. I attempted to use a both RNNs and CNNs for this and both have show disappoiting results.

So I am going to change my approach. I am going to treat the timing task as a categorisation task on patches of the sound recording. Thus by iteratively sliding a patch on the input recording, I can use the neural network to segment the input. I also plan on turning the problem into a image classification problem by transforming the sounds into an image. There are several transformation that do this so I will be researching for a bit. 
