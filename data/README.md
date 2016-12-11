# Data:

The data is provided by: https://www.kaggle.com/kinguistics/heartbeat-sounds

It has two sets: A and B. A is gathered by an iPhone app ( iStethoscope Pro ), and B is gather from clinical trials.

There are 3 .csv files.

* set_a.csv and set_b.csv contain the classes for each file. They contain 4 fields: dataset (a or b), fname, label, sublabel

* set_a_timing contains the segmantation for different sounds during a heartbeat. It contains the following fields: fname, cycle, sound(S1 or S2), and location of the sound.


The format of the soundfiles is .wav