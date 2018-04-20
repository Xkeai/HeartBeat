import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

from timingDataset import timingDataset
from preprocessing import stft_sig2image, normalize

# Getting the dataset
dataset = timingDataset(fname="../data/set_a_timing.csv",
                        prefix="../data",
                        nbefore=2**10,
                        nafter=2**10,
                        seed=1234)


# Getting a test batch
N = 2
data, label = dataset.next_batch_train(N)
data = normalize(data)
SXX = stft_sig2image(data, nperseg=8, nfft=512)
t = range(data.shape[1])
print(label)
# Plotting the signal
fig, axes = plt.subplots(N, 2)

for n in range(N):
    ax1 = axes[n, 0]
    ax2 = axes[n, 1]

    t = range(data.shape[1])
    lines = ax1.plot(t, data[n, :], 'black')
    ax1.grid()

    # Plotting the STFT of the signal
    im = ax2.pcolormesh(SXX[n, :, :, 0])
    fig.colorbar(im, ax=ax2)

plt.show()
