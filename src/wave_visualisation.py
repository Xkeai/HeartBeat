import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

from classificationDataset import classificationDataset


dataset = classificationDataset("../data/set_b.csv", "../data/", seed=123)

N = 2
data, label = dataset.next_batch_valid()

t, f, Zxx = signal.stft(data, nperseg=2**9, noverlap=2**4)
print(Zxx.shape)
