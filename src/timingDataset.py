# This file contains the definition of class and related function
# that allows us to interface with the dataset
# The dataset has two main components:
# Sound files that contain recordings of heartbeat.
# A csv file that contains the location of specific sounds in the sound files


import csv
import numpy as np

import wave
import struct

# getTimingInfo:
# A parser for the csv file.
# Input:
########
# FNAME: filename of a csv
# The csv contains the filename of the recordings,
# and the location and type of the sounds
# Output:
#########
# timingInfo: A dictionary
# -> keys: filename of the soundfile
# -> value: a dictionary whose keys are sound types and
# the value is a list of the locations


def getTimingInfo(FNAME, fnamePrefix=''):
    timingInfo = {}
    with open(FNAME, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            soundFname = fnamePrefix + row['fname']
            # If this file doesn't have already have a dict,
            # we need to create one.
            # This is caused by the way python deals with lists, not dicts
            if soundFname not in timingInfo:
                timingInfo[soundFname] = {'S1': list(), 'S2': list()}
            soundType = row['sound']
            soundLocation = row['location']
            timingInfo[soundFname][soundType].append(soundLocation)
    return timingInfo

# importWavFile:
# import the sound .wav file
# Input:
########
# FNAME: filename of the sound file
# Output:
#########
# A simple list
# Note:  This function is adapted from this notebook:
# https://www.kaggle.com/kinguistics/loading-and-manipulating-heartbeat-audio


def importWavFile(FNAME):
    f = wave.open(FNAME)
    frames = f.readframes(-1)
    samples = struct.unpack('h' * f.getnframes(), frames)
    return samples

# findMaxLength:
# Goes through the files and checks their lengths. Returns the max
# Input:
########
# filenames: A list of strings that contain the path to a wav file
# Output:
########
# The maximum length of all the wav files


def findMaxLength(filenames):
    lengths = list()
    for fname in filenames:
        recording = importWavFile(fname)
        lengths.append(len(recording))
    return max(lengths)

# getSoundData:
# import the recordings from files and puts them in the right format
# Input:
########
# FNAME: file name of recording
# max_length: the length of the maximum recordings
# Output:
#########
# A numpy array/tensor of shape [1,max_length,1]


def getSoundData(FNAME, max_length):
    ret = np.empty([1, max_length, 1], dtype=np.float16)
    samples = importWavFile(FNAME)
    for i in range(len(samples)):
        ret[0, i, 0] = samples[i]
    return ret
