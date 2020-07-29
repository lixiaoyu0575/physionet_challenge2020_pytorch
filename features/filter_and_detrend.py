import sys
import pandas as pd
import numpy as np
import scipy.io as scio
import os
import wfdb
from scipy import signal
import csv
import pywt
def WTfilt_1d(sig):
    coeffs = pywt.wavedec(data=sig, wavelet='db5', level=9)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs
    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    # 将高频信号cD1、cD2置零
    cD1.fill(0)
    cD2.fill(0)
    # 将其他中低频信号按软阈值公式滤波
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)
    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
    return rdata


def ECG_Filter_and_Detrend(path):
    ##input: The absolute path of data that you want to be filtered
    def readname(filePath):
        name = os.listdir(filePath)
        name.sort()
        return name

    file_colletion = readname(path)
    print(file_colletion)
    dat_collection = []
    for i in range(0, len(file_colletion)):
        if file_colletion[i].find('.mat') >= 0:
            dat_collection.append(file_colletion[i].strip('.mat'))

    f = open('error_data.csv', 'w', encoding='utf-8')
    csv_writer = csv.writer(f)
    for j in range(0, len(dat_collection)):
        print(dat_collection[j])

        data = scio.loadmat(path+dat_collection[j])
        filtered_data = pd.DataFrame()
        for k in range(12):
            try:
                filtered_data[k] = signal.detrend(WTfilt_1d(data['val'][k]))
            except ValueError:
                ##有些数据全是0，记录下来，无法进行detrend处理
                csv_writer.writerow([dat_collection[j]])
                filtered_data[k] = WTfilt_1d(data['val'][k])
        scio.savemat(dat_collection[j] + '.mat', {'val': filtered_data.values.T})


    f.close()
ECG_Filter_and_Detrend('/home/weiyuhua/Data/challenge2020/All_data_resampled_to_500HZ/')