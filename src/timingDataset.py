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


def getTimingInfo(FNAME, fnamePrefix=''):
    """ getTimingInfo:
    A parser for the csv file.
    Input:
    FNAME: name of a csv file
    The csv contains the filename of the recordings,
    and the location and type of the sounds
    Output:
    timingInfo: A list which contains 3 dimensionals tuples:
    (soundFname, location, category)
    """
    timingInfo = list()
    with open(FNAME, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            soundFname = os.path.join(fnamePrefix, row['fname'])
            soundLocation = int(row['location'])
            soundType = row['sound']
            datapoint = (soundFname, soundLocation, soundType)
            timingInfo.append(datapoint)
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
    return np.array(samples)


def getNeighbourhoodFromSample(samples, loc, nbefore, nafter):
    """ getNeighbourhoodFromSample
    Extract a neighboorhood of values around a location in a sample
    Input:
    sample: a 1-D numpy array containing the values of the recording
    loc: The location that the neighbourhood is going to be around
    nbefore: the number of values to be taken before loc
    nafter: the number of values to be taken after loc
    Output:
    A numpy array of shape [size].
    The data is zero-padded in the end as necessary.
    """
    V = np.zeros(shape=[nbefore + nafter])
    start = loc - nbefore
    if start < 0:
        start = 0
    end = loc + nafter
    if end > samples.shape[0]:
        end = samples.shape[0]
    V[0:(end - start)] = samples[start:end]
    return V


def produceDataPoint(timingInfo, nbefore, nafter):
    """ produceDataPoint:
    Produce the datapoint from the tuples
    Input:
    timingInfo: tuple that contains the file name, location and type of sound
    size: Size of the data
    Output:
    A 2-dimensional tuple whose first elements is data and the second is the label
    """
    # Extracting the information
    fname = timingInfo[0]
    loc = timingInfo[1]
    category = timingInfo[2]
    # Creating the data
    samples = importWavFile(fname)
    data = getNeighbourhoodFromSample(samples, loc, nbefore, nafter)
    # Creating the label
    # S1 = [1,0]
    # S2 = [0,1]
    label = np.zeros([2])
    category = int(category[1])
    label[category - 1] = 1

    return (data, label)


class timingDataset:
    """
    This is an object to manage the data for the timing task
    """

    def __init__(self, fname, prefix, nbefore, nafter, seed=None):
        # Saving the parameters for the constructor
        self.fname = fname
        self.prefix = prefix
        self.nbefore = nbefore
        self.nafter = nafter
        self.size = nbefore + nafter
        # Get the timingInfo
        self.timingInfo = getTimingInfo(self.fname, self.prefix)
        # Get some metadata on the timingInfo
        self.set_size = len(self.timingInfo)
        self.distinct_recordings = list(set(map(
            lambda x: x[0],
            self.timingInfo)))
        # Seperating the data into training and validation subsets
        cuttoff_index = int(self.set_size * 0.8)

        index = np.arange(self.set_size)
        if seed is not None:
            np.random.seed(seed)
            np.random.shuffle(index)

        self.info_train = [self.timingInfo[i]
                           for i in index[:cuttoff_index]]
        self.info_valid = [self.timingInfo[i]
                           for i in index[cuttoff_index:]]

        self.no_train = len(self.info_train)
        self.no_valid = len(self.info_valid)
        # Setting the cursors
        self.index_train = 0
        self.index_valid = 0

        self.epochs = 0

        self.get_new_permutation()

    def reset(self):
        self.index_train = 0
        self.index_valid = 0

        self.epochs = 0

    def get_new_permutation(self):
        """
        We want to go through the training data in a random manner
        So each time an epoch is completed the other,
        we go through is reshuffled
        """
        self.perm = np.arange(self.no_train)
        np.random.shuffle(self.perm)

    def next_batch_train(self, batch_size=1):
        """
        Gives the next batch in the training subset
        """

        start = self.index_train
        self.index_train += batch_size
        # If the index exceeds the the subset size,
        # we reset the index and get a new permutation.
        if(self.index_train > self.no_train):
            self.epochs += 1
            start = 0
            self.index_train = batch_size
            self.get_new_permutation()
        end = self.index_train
        # Extract from the permutation the indexes we need
        curr_perm = self.perm[start:end]
        # Get the information we need
        info = [self.info_train[i] for i in curr_perm]
        # We iteratively produce the data points and store them in np arrays
        data = np.zeros(shape=[batch_size, self.size])
        label = np.zeros(shape=[batch_size, 2])
        n = 0
        for i in info:
            d, l = produceDataPoint(i, self.nbefore, self.nafter)
            data[n, :] = d
            label[n, :] = l
            n += 1
        return (data, label)

    def next_batch_valid(self, batch_size=1):
        """
        Gives the next batch in the valid subset
        """
        start = self.index_valid
        self.index_valid += batch_size
        # If the index exceeds the the subset size,
        # we reset the index and return (None,None)
        if(self.index_valid > self.no_valid):
            self.index_valid = 0
            return (None, None)
        end = self.index_valid
        # Get the information we need
        info = self.info_valid[start:end]
        # We iteratively produce the data points and store them in np arrays
        data = np.zeros(shape=[batch_size, self.size])
        label = np.zeros(shape=[batch_size, 2])
        n = 0
        for i in info:
            d, l = produceDataPoint(i, self.nbefore, self.nafter)
            data[n, :] = d
            label[n, :] = l
            n += 1
        return (data, label)
