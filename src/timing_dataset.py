from scipy.io import wavfile

import csv
import numpy as np


# A utility class to manage the data from the set_a_timing.csv file

class timing_dataset:

    def __init__(self, fname):
        csvfile = open(fname)
        reader = csv.DictReader(csvfile)
        # I don't have enough RAM so I am going to import data when I need it
        data_list = []
        label_lists = []
        for row in reader:
            fname = '../data/' + row['fname']
            if((len(data_list) == 0)):
                data_list.append(fname)
                label_lists.append([])
            if(data_list[-1] != fname):
                data_list.append(fname)
                label_lists.append([])
            l = label_lists[-1]
            label = (row['sound'], int(row['location']))
            l.append(label)
        # We seperate the data into two datasets: train, and valid
        self.data_train = []
        self.label_train = []

        self.data_valid = []
        self.label_valid = []
        l = len(data_list)
        for i in range(l):
            if (i < 0.8 * l):
                self.data_train.append(data_list[i])
                self.label_train.append(label_lists[i])
            else:
                self.data_valid.append(data_list[i])
                self.label_valid.append(label_lists[i])

        # We need to find out themax length among the files as all out put
        # np.array have to be the same size
        max_length = 0
        for fname in data_list:
            fs, w = wavfile.read(fname)
            length = w.shape[0]
            if(max_length < length):
                max_length = length
        self.max_length = max_length
        # We are going to add some flow control variables
        self.index_train = 0
        self.no_train = len(self.data_train)

        self.index_valid = 0
        self.no_valid = len(self.data_valid)
        self.epochs = 0

    # This imports data from the disk to the memory
    # It returns two np arrays.
    # The first is the data. Its shape is 1xmax_lengthx1
    # The second is the label. Its shape is max_lengthx2.
    # The first col is for S1. The second is for S2.
    def import_data(self, fname, label_l):
        print "Importing %s" % (fname)
        fs, sound = wavfile.read(fname)

        data = np.zeros((1, self.max_length, 1), dtype=np.float64)
        data[0, 0:sound.shape[0], 0] = sound

        label = np.zeros((self.max_length, 2))
        for l in label_l:
            if(l[0] == 'S1'):
                i = 0
            else:
                i = 1
            label[l[1], i] = 1
        return data, label

    # This function returns the data from the training set one by one.
    def next_train(self):
        index = self.index_train
        self.index_train += 1
        if self.index_train > self.no_train:
            self.epochs += 1
            self.index_train = 0
        fname = self.data_train[index]
        label_l = self.label_train[index]
        return self.import_data(fname, label_l)

    def next_valid(self):
        index = self.index_valid
        self.index_valid += 1
        if self.index_valid > self.no_valid:
            self.index_valid = 0
            return -1, -1
        fname = self.data_valid[index]
        label_l = self.label_valid[index]
        return self.import_data(fname, label_l)
