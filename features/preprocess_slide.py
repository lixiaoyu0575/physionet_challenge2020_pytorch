import csv
import scipy.io as scio
import biosppy
import wfdb
import pywt
import numpy as np
import h5py
import pandas as pd
from scipy import signal

#####filter
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
def get_preprocessed_data_slide(filie_name,data):
    ## set the sample frequency
    FS = 500
    patient_info = []
    long_pid = []
    long_data0 = []
    long_data1 = []
    long_data2 = []
    long_data3 = []
    long_data4 = []
    long_data5 = []
    long_data6 = []
    long_data7 = []
    long_data8 = []
    long_data9 = []
    long_data10 = []
    long_data11 = []
    qrs_info = []
    short_data0 = []
    short_data1 = []
    short_data2 = []
    short_data3 = []
    short_data4 = []
    short_data5 = []
    short_data6 = []
    short_data7 = []
    short_data8 = []
    short_data9 = []
    short_data10 = []
    short_data11 = []
    short_pid0 = []
    short_pid1 = []
    short_pid2 = []
    short_pid3 = []
    short_pid4 = []
    short_pid5 = []
    short_pid6 = []
    short_pid7 = []
    short_pid8 = []
    short_pid9 = []
    short_pid10 = []
    short_pid11 = []

    print('get preprocessed data '+filie_name)
    record = wfdb.rdrecord('/DATASET/challenge2020/All_data/'+ filie_name)
    ###the loaded data shape is  like 12*7500
    resample_data= pd.DataFrame()
    sampleNum = record.__dict__['sig_len']
    resample_num = int(sampleNum * (FS / record.__dict__['fs']))
    filtered_data = pd.DataFrame()
    for k in range(12):
        resample_data[k] = signal.resample(data[k], resample_num, axis=0, window=None)
        try:
            filtered_data[k] = signal.detrend(WTfilt_1d(resample_data[k]))
        except ValueError:
            ##有些数据全是0，记录下来，无法进行detrend处理
            filtered_data[k] = WTfilt_1d(resample_data[k])
    data = (filtered_data.values).T
    ##patient
    patient_info_each = []
    patient_info_each.append(filie_name)
    patient_info_each.append(record.__dict__['comments'][0][5:])
    patient_info_each.append(record.__dict__['comments'][1][5:])
    patient_info.append(patient_info_each)
    ###LONG
    long_pid.append(filie_name)
    long_data0.append(data[0].tolist())
    long_data1.append(data[1].tolist())
    long_data2.append(data[2].tolist())
    long_data3.append(data[3].tolist())
    long_data4.append(data[4].tolist())
    long_data5.append(data[5].tolist())
    long_data6.append(data[6].tolist())
    long_data7.append(data[7].tolist())
    long_data8.append(data[8].tolist())
    long_data9.append(data[9].tolist())
    long_data10.append(data[10].tolist())
    long_data11.append(data[11].tolist())
    ###QRS_INFO
    signal_data = data[1]
    rpeaks_indices_1 = biosppy.signals.ecg.hamilton_segmenter(signal=signal_data, sampling_rate = FS)
    rpeaks_indices_2 = biosppy.signals.ecg.correct_rpeaks(signal=signal_data, rpeaks=rpeaks_indices_1[0],sampling_rate = FS)
    qrs_info.append(np.diff(rpeaks_indices_2[0]).tolist())
    ###SHORT
    rpeaks_indices_0 = biosppy.signals.ecg.hamilton_segmenter(signal=data[0], sampling_rate=FS)
    rpeaks_indices_1 = biosppy.signals.ecg.hamilton_segmenter(signal=data[1], sampling_rate=FS)
    rpeaks_indices_2 = biosppy.signals.ecg.hamilton_segmenter(signal=data[2], sampling_rate=FS)
    rpeaks_indices_3 = biosppy.signals.ecg.hamilton_segmenter(signal=data[3], sampling_rate=FS)
    rpeaks_indices_4 = biosppy.signals.ecg.hamilton_segmenter(signal=data[4], sampling_rate=FS)
    rpeaks_indices_5 = biosppy.signals.ecg.hamilton_segmenter(signal=data[5], sampling_rate=FS)
    rpeaks_indices_6 = biosppy.signals.ecg.hamilton_segmenter(signal=data[6], sampling_rate=FS)
    rpeaks_indices_7 = biosppy.signals.ecg.hamilton_segmenter(signal=data[7], sampling_rate=FS)
    rpeaks_indices_8 = biosppy.signals.ecg.hamilton_segmenter(signal=data[8], sampling_rate=FS)
    rpeaks_indices_9 = biosppy.signals.ecg.hamilton_segmenter(signal=data[9], sampling_rate=FS)
    rpeaks_indices_10 = biosppy.signals.ecg.hamilton_segmenter(signal=data[10], sampling_rate=FS)
    rpeaks_indices_11 = biosppy.signals.ecg.hamilton_segmenter(signal=data[11], sampling_rate=FS)
    tol_para = 0.1
    rpeaks_indices_0_c = biosppy.signals.ecg.correct_rpeaks(signal=data[0], rpeaks=rpeaks_indices_0[0],sampling_rate=FS, tol=tol_para)
    rpeaks_indices_1_c = biosppy.signals.ecg.correct_rpeaks(signal=data[1], rpeaks=rpeaks_indices_1[0],sampling_rate=FS, tol=tol_para)
    rpeaks_indices_2_c = biosppy.signals.ecg.correct_rpeaks(signal=data[2], rpeaks=rpeaks_indices_2[0],sampling_rate=FS, tol=0.05)
    rpeaks_indices_3_c = biosppy.signals.ecg.correct_rpeaks(signal=data[3], rpeaks=rpeaks_indices_3[0],sampling_rate=FS, tol=0.05)
    rpeaks_indices_3_c_avR = []
    for each_R in range(1, len(rpeaks_indices_3_c[0])):
        R_P = rpeaks_indices_3_c[0][each_R]
        R_P_correct = np.argmin(data[3][R_P - 30:R_P + 20]) + R_P - 30
        rpeaks_indices_3_c_avR.append(R_P_correct)
    rpeaks_indices_3_c_avR = np.array(rpeaks_indices_3_c_avR)
    rpeaks_indices_4_c = biosppy.signals.ecg.correct_rpeaks(signal=data[4], rpeaks=rpeaks_indices_4[0],sampling_rate=FS, tol=0.05)
    rpeaks_indices_5_c = biosppy.signals.ecg.correct_rpeaks(signal=data[5], rpeaks=rpeaks_indices_5[0],sampling_rate=FS, tol=tol_para)
    rpeaks_indices_6_c = biosppy.signals.ecg.correct_rpeaks(signal=data[6], rpeaks=rpeaks_indices_6[0],sampling_rate=FS, tol=0.05)
    rpeaks_indices_7_c = biosppy.signals.ecg.correct_rpeaks(signal=data[7], rpeaks=rpeaks_indices_7[0],sampling_rate=FS, tol=tol_para)
    rpeaks_indices_8_c = biosppy.signals.ecg.correct_rpeaks(signal=data[8], rpeaks=rpeaks_indices_8[0],sampling_rate=FS, tol=tol_para)
    rpeaks_indices_9_c = biosppy.signals.ecg.correct_rpeaks(signal=data[9], rpeaks=rpeaks_indices_9[0],sampling_rate=FS, tol=tol_para)
    rpeaks_indices_10_c = biosppy.signals.ecg.correct_rpeaks(signal=data[10], rpeaks=rpeaks_indices_10[0],sampling_rate=FS, tol=tol_para)
    rpeaks_indices_11_c = biosppy.signals.ecg.correct_rpeaks(signal=data[11], rpeaks=rpeaks_indices_11[0],sampling_rate=FS, tol=tol_para)
    for j in range(len(rpeaks_indices_0_c[0]) - 1):
        short_pid0.append(filie_name)
        short_data0.append(data[0].tolist()[rpeaks_indices_0_c[0][j]:rpeaks_indices_0_c[0][j + 1]])
    for j in range(len(rpeaks_indices_1_c[0]) - 1):
        short_pid1.append(filie_name)
        short_data1.append(data[1].tolist()[rpeaks_indices_1_c[0][j]:rpeaks_indices_1_c[0][j + 1]])
    for j in range(len(rpeaks_indices_2_c[0]) - 1):
        short_pid2.append(filie_name)
        short_data2.append(data[2].tolist()[rpeaks_indices_2_c[0][j]:rpeaks_indices_2_c[0][j + 1]])
    for j in range(len(rpeaks_indices_3_c_avR) - 1):
        short_pid3.append(filie_name)
        short_data3.append(data[3].tolist()[rpeaks_indices_3_c_avR[j]:rpeaks_indices_3_c_avR[j + 1]])
    for j in range(len(rpeaks_indices_4_c[0]) - 1):
        short_pid4.append(filie_name)
        short_data4.append(data[4].tolist()[rpeaks_indices_4_c[0][j]:rpeaks_indices_4_c[0][j + 1]])
    for j in range(len(rpeaks_indices_5_c[0]) - 1):
        short_pid5.append(filie_name)
        short_data5.append(data[5].tolist()[rpeaks_indices_5_c[0][j]:rpeaks_indices_5_c[0][j + 1]])
    for j in range(len(rpeaks_indices_6_c[0]) - 1):
        short_pid6.append(filie_name)
        short_data6.append(data[6].tolist()[rpeaks_indices_6_c[0][j]:rpeaks_indices_6_c[0][j + 1]])
    for j in range(len(rpeaks_indices_7_c[0]) - 1):
        short_pid7.append(filie_name)
        short_data7.append(data[7].tolist()[rpeaks_indices_7_c[0][j]:rpeaks_indices_7_c[0][j + 1]])
    for j in range(len(rpeaks_indices_8_c[0]) - 1):
        short_pid8.append(filie_name)
        short_data8.append(data[8].tolist()[rpeaks_indices_8_c[0][j]:rpeaks_indices_8_c[0][j + 1]])
    for j in range(len(rpeaks_indices_9_c[0]) - 1):
        short_pid9.append(filie_name)
        short_data9.append(data[9].tolist()[rpeaks_indices_9_c[0][j]:rpeaks_indices_9_c[0][j + 1]])
    for j in range(len(rpeaks_indices_10_c[0]) - 1):
        short_pid10.append(filie_name)
        short_data10.append(data[10].tolist()[rpeaks_indices_10_c[0][j]:rpeaks_indices_10_c[0][j + 1]])
    for j in range(len(rpeaks_indices_11_c[0]) - 1):
        short_pid11.append(filie_name)
        short_data11.append(data[11].tolist()[rpeaks_indices_11_c[0][j]:rpeaks_indices_11_c[0][j + 1]])
    return patient_info,short_data0,short_data1,short_data2,short_data3,short_data4,short_data5,short_data6,short_data7,short_data8,short_data9,short_data10,short_data11,long_data0,long_data1,long_data2,long_data3,long_data4,long_data5,long_data6,long_data7,long_data8,long_data9,long_data10,long_data11,qrs_info, long_pid, short_pid0,short_pid1,short_pid2,short_pid3,short_pid4,short_pid5,short_pid6,short_pid7,short_pid8,short_pid9,short_pid10,short_pid11