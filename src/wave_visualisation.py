import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

from timingDataset import timingDataset

# Getting the dataset
dataset = timingDataset(fname="../data/set_a_timing.csv",
                        prefix="../data",
                        nbefore=2**10,
                        nafter=2**10,
                        seed=123)


# Getting a test batch
N = 4
nperseg = 2**11
data, label = dataset.next_batch_valid(N)
t, f, SXX = signal.stft(data,
                        nperseg=nperseg,
                        nfft=None, noverlap=nperseg - 8,
                        axis=1)
SXX = np.abs(SXX[:, :, :, np.newaxis])
mu = np.mean(SXX, axis=(1, 2))
sigma = np.std(SXX, axis=(1, 2))
for n in range(N):
    SXX[n, :, :, 0] = (SXX[n, :, :, 0] - mu[n]) / sigma[n]

t = range(data.shape[1])
print(label)
# Plotting the signal
fig, axes = plt.subplots(N, 2)
axes = np.reshape(axes, (N, 2))

for n in range(N):
    ax1 = axes[n, 0]
    ax2 = axes[n, 1]
    lines = ax1.plot(t, data[n, :], 'black')
    ax1.grid()

    # Plotting the STFT of the signal
    im = ax2.contourf(SXX[n, :, :, 0])
    fig.colorbar(im, ax=ax2)
plt.show()
