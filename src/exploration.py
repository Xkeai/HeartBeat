import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

import wave
import struct
import re


def importWavFile(FNAME):
    f = wave.open(FNAME)
    frames = f.readframes(-1)
    samples = struct.unpack('h' * f.getnframes(), frames)
    return np.array(samples)

# Importing the csv
df = pd.read_csv("../data/set_b.csv")
prefix = "../data/"
# Removing the missing labels
df = df[~df.label.isnull()]
# Getting all the different categories
labels = df.label.unique()
n_cat = len(labels)
sublabels = df.sublabel.dropna().unique()
# We get a list that has an example for each category
examples = []
for l in labels:
    fname = prefix + df[df.label == l].iloc[1].fname
    fname = re.sub("Btraining_", "", fname)
    fname = re.sub("(?<=" + l + ")_", "__", fname)

    examples.append((l, fname))
# Sublabels
for s in sublabels:
    fname = prefix + df[df.sublabel == s].iloc[1].fname
    fname = re.sub("Btraining_", "", fname)
    # fname = re.sub("(?<=(?!noisy)" + l + ")_", "__", fname)

    #examples.append((s, fname))

# Plotting the results
n_cat = len(examples)

for i in range(n_cat):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    # Importing the sound
    label, fname = examples[i]
    sample = importWavFile(fname)
    # Plotting normal
    ax1.plot(sample)
    ax1.set(title="Example of a %s" % (label))

    #
    t, f, Zxx = signal.stft(sample, nperseg=2**10, noverlap=2**10 - 2**7)
    Zxx = np.abs(Zxx)
    ax2.pcolormesh(f, t, Zxx)

plt.show()
