# This file contains the definition of class and related function
# that allows us to interface with the dataset
# The dataset has two main components:
# Sound files that contain recordings of heartbeat.
# A csv file that contains the location of specific sounds in the sound files
import os

import csv
import numpy as np

import wave
import struct

from math import isnan


def getTimingInfo(FNAME, fnamePrefix=''):
    """ getTimingInfo:
    A parser for the csv file.
    Input:
    FNAME: filename of a csv
    The csv contains the filename of the recordings,
    and the location and type of the sounds
    Output:
    timingInfo: A dictionary
    -> keys: filename of the soundfile
    -> value: a dictionary whose keys are sound types and
    the value is a list of the locations"""
    timingInfo = {}
    with open(FNAME, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            soundFname = os.path.join(fnamePrefix, row['fname'])
            # If this file doesn't have already have a dict,
            # we need to create one.
            # This is caused by the way python deals with lists, not dicts
            if soundFname not in timingInfo:
                timingInfo[soundFname] = {'S1': list(), 'S2': list()}
            soundType = row['sound']
            soundLocation = row['location']
            timingInfo[soundFname][soundType].append(int(soundLocation))
    return timingInfo


def importWavFile(FNAME):
    """ importWavFile:
    import the sound .wav file
    Input:
    FNAME: filename of the sound file
    Output:
    A simple list of numerical values
    Note:  This function is adapted from this notebook:
    https://www.kaggle.com/kinguistics/loading-and-manipulating-heartbeat-audio"""

    f = wave.open(FNAME)
    frames = f.readframes(-1)
    samples = struct.unpack('h' * f.getnframes(), frames)
    return samples


def findMaxLength(filenames):
    """ findMaxLength:
    Goes through the files and checks their lengths. Returns the max
    Input:
    filenames: A list of strings that contain the path to a wav file
    Output:
    The maximum length of all the wav files"""
    lengths = list()
    for fname in filenames:
        recording = importWavFile(fname)
        lengths.append(len(recording))
    return max(lengths)


def getSoundData(FNAME, max_length=-1):
    """ getSoundData:
    import the recordings from files and puts them in the right format
    Input:
    FNAME: file name of recording
    max_length: the length of the maximum recordings
    Output:
    A numpy array/tensor of shape [1,max_length,1]"""
    samples = importWavFile(FNAME)
    if(max_length == -1):
        max_length = len(samples)
    ret = np.zeros([1, max_length, 1], dtype=np.float32)
    ret[0, :len(samples), 0] = samples[:]
    return ret


def createLabel(locationsByType, max_length):
    """ createLabel
    creates the label for based the location of various sounds
    Input:
    locationsByType: A dictionary whose keys are the types and the values are lists of location
    max_length: The length of the output tensor
    Output:
    A numpy array of shape [1,max_length, number of sound types]
    It contains 0 in all points except in points where the locations is """
    soundTypes = locationsByType.keys()
    label = np.zeros([1, max_length, len(soundTypes)], np.float32)
    i = 0
    for t in soundTypes:
        locations = locationsByType[t]
        for l in locations:
            label[0, l, i] = 1
        i = i + 1
    return label


def testSoundFiles(filenames):
    """ testSoundFiles
    Imports the sounds files and runs a few test on them.
    Input:
    filenames: A list of strings
    Output:
    True if all test ae successful, False if not
    Side-effect:
    If there is a problem, prints out a text explaining the problem.
    """
    noProblem = True

    is_nan_vec = np.vectorize(isnan)
    is_None_vec = np.vectorize(lambda x: x is None)

    for fname in filenames:
        data = getSoundData(fname)
        if(data is None):
            print("problem importing:" + fname)
            noProblem = False
            break
        if(is_nan_vec(data).any()):
            print("Invalid data 'nan' in:" + fname)
            noProblem = False
        if(is_None_vec(data).any()):
            print("Invalid data 'None' in:" + fname)
            noProblem = False
    if(noProblem):
        print('There are no problems in recordings')
    return noProblem


class timingDataset:
    """
    This is an object to manage the data for the timing task

    """

    def __init__(self, FNAME, fnamePrefix='../data/', normalise=True):
        # Importing the meta data
        self.timingInfo = getTimingInfo(FNAME, fnamePrefix=fnamePrefix)
        # Counting the data points
        self.dataset_size = len(self.timingInfo)
        # And dividing them into train and valid set (with a ration of 80/20
        self.no_train = int(self.dataset_size * 0.8)
        self.no_valid = self.dataset_size - self.no_train
        # I am going to import data as it is needed.
        # My computer is fairly low on RAM.
        # On a bigger machine, it might be better to import it all in one go.
        self.keys_train = self.timingInfo.keys()[:self.no_train]
        self.keys_valid = self.timingInfo.keys()[self.no_train:]
        # We need to have the max_length for we construct the tensors
        self.max_length = findMaxLength(self.timingInfo.keys())
        # Testing the soundFiles
        if(not testSoundFiles(self.timingInfo.keys())):
            raise ValueError("Invalid values for the recordings")
        # The user can choose the size of the data they want.
        # To keep track of which data to return we have two cursors
        # The first one keep track of which sample we are using
        # The second keeps track of where we are in the sample
        self.index_train = 0
        self.index_sample_train = 0
        self.current_data_train = None
        self.current_label_train = None

        # Naturally we have a second set for validation
        self.index_valid = 0
        self.index_sample_valid = 0
        self.current_data_valid = None
        self.current_label_valid = None

        self.epochs = 0
        self.get_new_permutation()

        # Lastly, parameter for pre-treatement of data
        self.normalise = normalise

    def reset(self):
        """
        Reseting the dataset
        """
        self.index_train = 0
        self.index_valid = 0
        self.epochs = 0
        self.get_new_permutation()

    def get_new_permutation(self):
        """
        We want to go through the training data in a random manner
        So each time an epoch is completed the other,
        we go through is reshuffled
        """
        self.perm = np.arange(self.no_train)
        np.random.shuffle(self.perm)

    def get_new_samples(self, key):
        # First getting the data
        data = getSoundData(key)
        if(self.normalise):
            data = (data - np.amin(data, axis=1))
            data = data / np.amax(data, axis=1)
        length = data.shape[1]
        # Then the labels
        locationsByType = self.timingInfo[key]
        soundTypes = locationsByType.keys()
        label = np.zeros([1, length, len(soundTypes)], dtype=np.float32)
        i = 0
        for t in soundTypes:
            locations = locationsByType[t]
            for l in locations:
                label[0, l, i] = 1
            i = i + 1

        return (data, label)

    def next_batch_train(self, batch_size=1, data_length=int(1e5), step_size=int(1e3)):
        """
        Get the batch for the training
        batch_size is the number of elements we want
        """
        data = np.zeros([batch_size, data_length, 1], dtype=np.float32)
        labels = np.zeros([batch_size, data_length, 2], dtype=np.float32)

        b = 0
        ready = True
        while(b < batch_size):
            # If we are at the end of the current epoch
            if(self.index_train >= self.no_train):
                # Emptying the current samples
                self.current_data_train = None
                self.current_label_train = None
                # Resetting the indicators and incrementing the epoch counter
                self.index_train = 0
                self.index_sample_train = 0
                self.get_new_permutation()
                self.epochs += 1
            # If the samples are not imported yet, we need to import them
            if self.current_data_train is None:
                # Getting the key to get the samples
                key = self.perm[self.index_train]
                key = self.keys_train[key]
                # Getting new samples
                data_train, label_train = self.get_new_samples(key)
                self.current_data_train = data_train
                self.current_label_train = label_train

            # Setting the limits
            start = self.index_sample_train
            end = start + data_length
            ready = True
            # We are at the end of the current samples
            if (end > self.current_data_train.shape[1]):
                # Incrementing and resetting the indexes
                self.index_train += 1
                self.index_sample_train = 0
                # Emptying the samples
                self.current_data_train = None
                self.current_label_train = None
                ready = False
            # When we are ready, we get the data
            if(ready):
                data[b, :, :] = self.current_data_train[0, start:end, :]
                labels[b, :, :] = self.current_label_train[0, start:end, :]
                self.index_sample_train += step_size
                b += 1

        return(data, labels)

    def next_batch_valid(self, batch_size=1, data_length=int(1e5), step_size=int(1e3)):
        """
        Get the batch for the training
        batch_size is the number of elements we want
        diff_n is the lag we want for the differentiation
        """
        data = np.zeros([batch_size, data_length, 1], dtype=np.float32)
        labels = np.zeros([batch_size, data_length, 2], dtype=np.float32)

        b = 0
        ready = True
        while(b < batch_size):
            # If we are at the end of the current epoch
            if(self.index_valid >= self.no_valid):
                # Emptying the current samples
                self.current_data_train = None
                self.current_label_valid = None
                # Resetting the indicators and incrementing the epoch counter
                self.index_valid = 0
                self.index_sample_valid = 0

                return(-1, -1, -1)
            # If the samples are not imported yet, we need to import them
            if self.current_data_valid is None:
                # Getting the key to get the samples
                key = self.index_valid
                key = self.keys_valid[key]
                # Getting new samples
                data_valid, label_valid = self.get_new_samples(key)
                self.current_data_valid = data_valid
                self.current_label_valid = label_valid

            # Setting the limits
            start = self.index_sample_valid
            end = start + data_length
            ready = True
            # We are at the end of the current samples
            if (end > self.current_data_valid.shape[1]):
                # Incrementing and resetting the indexes
                self.index_valid += 1
                self.index_sample_valid = 0
                # Emptying the samples
                self.current_data_valid = None
                self.current_label_valid = None
                ready = False
            # When we are ready, we get the data
            if(ready):
                data[b, :, :] = self.current_data_valid[0, start: end, :]
                labels[b, :, :] = self.current_label_valid[0, start: end, :]
                self.index_sample_valid += step_size
                b += 1

        return(data, labels, 1)
