import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from timingDataset import timingDataset

# Getting the dataset
dataset = timingDataset("../data/set_a_timing.csv")

# Some variables

num_hidden = 16
learning_rate = 10**-6
batch_size = 1

diff_n = 0
# Getting one samples
data, label = dataset.next_batch_train()

# Transforming the data

t = range(data.shape[1])
s1_loc = np.where(label[0, :, 0] == 1)[0]
s2_loc = np.where(label[0, :, 1] == 1)[0]

# Plotting
fig, ax = plt.subplots()
ax.plot(t, data[0, :, 0], 'black')
ax.vlines(s1_loc, 0, 1, transform=ax.get_xaxis_transform(), colors='r')
ax.vlines(s2_loc, 0, 1, transform=ax.get_xaxis_transform(), colors='b')

ax.grid()

plt.show()


batches_valid = 0
while True:
    valid_data, valid_label, ended = dataset.next_batch_valid()
    if ended == -1:
        break
    batches_valid += 1

print(batches_valid)
dataset.reset()
batches_train = 0
while True:
    data, label = dataset.next_batch_train()
    if dataset.epochs > 1:
        break
    batches_train += 1

print(batches_train)
