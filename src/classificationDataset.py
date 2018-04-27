# Pandas for csv importation, and to manage the dataframe
import pandas as pd
# Numpy to manage the arrays
import numpy as np
# Re for regex to manage the filenames
import re

# Wave and struct for importation of the wav
import wave
import struct


def importWavFile(FNAME, length=None):
    # Import the wav file as a np array
    f = wave.open(FNAME)
    frames = f.readframes(-1)
    samples = struct.unpack('h' * f.getnframes(), frames)
    samples = np.array(samples)
    if length is None:
        return samples
    a = np.zeros(length)
    a[:samples.shape[0]] = samples
    return a


class classificationDataset:

    def __init__(self, csvFname, dataPrefix, seed=None):
        # Importing and cleaning up the csv
        df = pd.read_csv(csvFname)
        df = df[~df.label.isnull()]
        self.labels = df.label.unique()

        # Cleaning the fnames
        # The filenames are from the csv.
        # To access from where the script is we need to move
        df.fname = df.fname.apply(
            lambda fname: dataPrefix + fname)
        # There is an artefact in the csv file.
        # The fname in csv are different from real names
        # Removing those
        df.fname = df.fname.apply(
            lambda fname: re.sub(r'Btraining_', '', fname))
        # Normally for when there is no sublabel, the fname should be:
        # <label>__<digits>.csv
        # but in the csv, it is:
        # <label>_<digits>
        # which needs correction.
        df.fname = df.fname.apply(
            lambda fname: re.sub(r'(/[a-z]+)_([0-9])', r'\1__\2', fname))
        # Checking the files and checking the greatest length
        self.max_length = 0
        for index, row in df.iterrows():
            sample = importWavFile(row.fname)
            if(sample.shape[0] > self.max_length):
                self.max_length = sample.shape[0]
            if(np.any(~np.isfinite(sample))):
                print("There is a problem in the file %s" % (row.fname))
                return None

        # The dataset is going to be divided in to train and valid sets
        # Random Partition (using the seed if specified)
        # Partition is label-wise to have similar ratios of labels
        self.train_set = []
        self.valid_set = []
        if seed is not None:
            np.random.seed(seed)
        for l in self.labels:
            # Extracting the sub-dataframe
            sdf = df[df.label == l]
            # Defining the the partition
            N = sdf.shape[0]  # Getting the length
            cuttoff_index = int(N * 0.8)  # Defining the limit of partition
            index = np.arange(N)
            np.random.shuffle(index)
            # Making the partition
            train_df = sdf.iloc[index[:cuttoff_index]]
            valid_df = sdf.iloc[index[cuttoff_index:]]
            self.train_set.append(train_df)
            self.valid_set.append(valid_df)
        # Transforming the list of df's to a single df
        self.train_set = pd.concat(self.train_set)
        self.valid_set = pd.concat(self.valid_set)

        self.no_train = self.train_set.shape[0]
        self.no_valid = self.valid_set.shape[0]
        # cursors/indexes to follow where the next batch is
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
        To avoid going around in circles during training, 
        we will shuffle the order we give the train steps 
        """
        self.perm = np.arange(self.no_train)
        np.random.shuffle(self.perm)

    def next_batch_train(self, batch_size=1):
        start = self.index_train
        self.index_train += batch_size
        # If the index is above the number of data points,
        # then we are at the end of the epoch
        if(self.index_train >= self.no_train):
            self.epochs += 1
            start = 0
            self.index_train = batch_size
            self.get_new_permutation()
        end = self.index_train
        # Extract the index of the rows
        curr_perm = self.perm[start:end]
        # Extract the relevant rows
        df = self.train_set.iloc[curr_perm]
        # Creating the label
        label = map(lambda x: 1 * (x == self.labels), df.label)
        label = np.array(label)
        # Creating the data
        data = map(lambda x: importWavFile(x, self.max_length), df.fname)
        data = np.array(data)
        return (data, label)

    def next_batch_valid(self, batch_size=1):
        start = self.index_valid
        self.index_valid += batch_size
        if(self.index_valid >= self.no_valid):
            self.index_valid = 0
            return(None, None)
        end = self.index_valid
        # Extract the relevant rows
        df = self.valid_set[start:end]
        # Creating the label
        label = map(lambda x: 1 * (x == self.labels), df.label)
        label = np.array(label)
        # Creating the data
        data = map(lambda x: importWavFile(x, self.max_length), df.fname)
        data = np.array(data)
        return (data, label)
