from scipy.signal import butter, lfilter, cheb2ord, cheby2
import matplotlib.pyplot as plt
import os

def cheby_lowpass(wp, ws, fs, gpass, gstop):
    wp = wp / fs
    ws = ws / fs
    order, wn = cheb2ord(wp, ws, gpass, gstop)
    b, a = cheby2(order, gstop, wn)
    return b, a


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def cheby_lowpass_filter(data, cutoff, band=35, disp=10, fs=500, gpass=1, gstop=10):
    # Filter requirements.
    # band : bandwidth
    # disp : displacement
    # fs : sample rate, Hz
    # cutoff : desired cutoff frequency of the filter, Hz

    cheby_freq = [(cutoff + disp) - band / 2, (cutoff + disp) + band / 2]
    b, a = cheby_lowpass(cheby_freq[0], cheby_freq[1], fs, gpass, gstop)
    y = lfilter(b, a, data)
    return y


def butter_lowpass_filter(data, cutoff=50, fs=500, order=5):
    # Filter requirements.
    # band : bandwidth
    # disp : displacement
    # fs : sample rate, Hz
    # cutoff : desired cutoff frequency of the filter, Hz

    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

channels = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

def plot(data, data_filted, save_file):
    fig, axs = plt.subplots(12, 1, sharey=True, figsize=(50, 50))

    for i in range(12):
        axs[i].plot(data[i,:5000])
        axs[i].plot(data_filted[i, :5000], color = 'red')
        axs[i].set_title(channels[i])
        axs[i].autoscale(enable=True, axis='both', tight=True)

    plt.savefig(save_file)
    plt.close()