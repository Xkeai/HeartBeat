import numpy as np
import scipy.signal as signal

# The purpose of this file to contain functions to pre-process the data
# The functions here expect the data to be  numpy array of shape:
# [batch_size, sample_length]


def stft_sig2image(data, nperseg, nfft):
    SXX = []
    for i in range(data.shape[0]):
        f, t, sxx = signal.stft(data[i, :], nperseg=nperseg, nfft=nfft)
        SXX.append(np.abs(sxx))
    SXX = np.array(SXX)
    SXX = SXX[:, :, :, np.newaxis]
    return SXX


def normalize(data):
    for i in range(data.shape[0]):
        mu = np.mean(data[i])
        data[i] = data[i] - mu
        m = np.amin(data[i])
        M = np.amax(data[i])
        data[i] = data[i] / (M - m)
    return data
