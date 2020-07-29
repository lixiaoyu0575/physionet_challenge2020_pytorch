import csv
import scipy.io as scio
import biosppy
import wfdb
import pywt
import numpy as np
import h5py
import pandas as pd
from scipy import signal
with open('/home/hanhaochen/physionet_challenge2020_pytorch/features/Preprocess_Data/REFERENCE.csv', 'r') as f_r:
    reader = csv.reader(f_r)
    result = list(reader)
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
## set the sample frequency
FS = 500
##### write long

hf_long = h5py.File('/DATASET/challenge2020/Preprocess_Data/long.h5', 'w')
l0 = hf_long.create_group('lead0')
l1 = hf_long.create_group('lead1')
l2 = hf_long.create_group('lead2')
l3 = hf_long.create_group('lead3')
l4 = hf_long.create_group('lead4')
l5 = hf_long.create_group('lead5')
l6 = hf_long.create_group('lead6')
l7 = hf_long.create_group('lead7')
l8 = hf_long.create_group('lead8')
l9 = hf_long.create_group('lead9')
l10 = hf_long.create_group('lead10')
l11 = hf_long.create_group('lead11')
for i in range(len(result)):
    print(result[i][0])
    record = wfdb.rdrecord('/DATASET/challenge2020/All_data/'+ result[i][0])
    data = record.__dict__['p_signal']
    data = np.transpose(data)
    resample_data= pd.DataFrame()
    sampleNum = record.__dict__['sig_len']
    resample_num = int(sampleNum * (FS / record.__dict__['fs']))
    for k in range(12):
        resample_data[k] = signal.resample(data[k], resample_num, axis=0, window=None)
    data = (resample_data.values).T
    l0.create_dataset(result[i][0], data = WTfilt_1d(data[0]).tolist())
    l1.create_dataset(result[i][0], data = WTfilt_1d(data[1]).tolist())
    l2.create_dataset(result[i][0], data = WTfilt_1d(data[2]).tolist())
    l3.create_dataset(result[i][0], data = WTfilt_1d(data[3]).tolist())
    l4.create_dataset(result[i][0], data = WTfilt_1d(data[4]).tolist())
    l5.create_dataset(result[i][0], data = WTfilt_1d(data[5]).tolist())
    l6.create_dataset(result[i][0], data = WTfilt_1d(data[6]).tolist())
    l7.create_dataset(result[i][0], data = WTfilt_1d(data[7]).tolist())
    l8.create_dataset(result[i][0], data = WTfilt_1d(data[8]).tolist())
    l9.create_dataset(result[i][0], data = WTfilt_1d(data[9]).tolist())
    l10.create_dataset(result[i][0], data = WTfilt_1d(data[10]).tolist())
    l11.create_dataset(result[i][0], data = WTfilt_1d(data[11]).tolist())
hf_long.close()

##### write QRS_info

hf_qrsinfo = h5py.File('/DATASET/challenge2020/Preprocess_Data/QRSinfo.h5', 'w')
for i in range(len(result)):
    print(result[i][0])
    record = wfdb.rdrecord('/DATASET/challenge2020/All_data/'+ result[i][0])
    data = record.__dict__['p_signal']
    data = np.transpose(data)
    resample_data= pd.DataFrame()
    sampleNum = record.__dict__['sig_len']
    resample_num = int(sampleNum * (FS / record.__dict__['fs']))
    for k in range(12):
        resample_data[k] = signal.resample(data[k], resample_num, axis=0, window=None)
    data = (resample_data.values).T
    signal_data = WTfilt_1d(data[1])
    rpeaks_indices_1 = biosppy.signals.ecg.hamilton_segmenter(signal=signal_data, sampling_rate = FS)
    rpeaks_indices_2 = biosppy.signals.ecg.correct_rpeaks(signal=signal_data, rpeaks=rpeaks_indices_1[0],sampling_rate = FS)
    hf_qrsinfo.create_dataset(result[i][0], data=np.diff(rpeaks_indices_2[0]).tolist())
hf_qrsinfo.close()

#### write short_data
hf_short = h5py.File('/DATASET/challenge2020/Preprocess_Data/short.h5', 'w')
s0 = hf_short.create_group('lead0')
s1 = hf_short.create_group('lead1')
s2 = hf_short.create_group('lead2')
s3 = hf_short.create_group('lead3')
s4 = hf_short.create_group('lead4')
s5 = hf_short.create_group('lead5')
s6 = hf_short.create_group('lead6')
s7 = hf_short.create_group('lead7')
s8 = hf_short.create_group('lead8')
s9 = hf_short.create_group('lead9')
s10 = hf_short.create_group('lead10')
s11 = hf_short.create_group('lead11')
for i in range(len(result)):
    print(result[i][0])
    record = wfdb.rdrecord('/DATASET/challenge2020/All_data/'+ result[i][0])
    data = record.__dict__['p_signal']
    data = np.transpose(data)
    resample_data= pd.DataFrame()
    sampleNum = record.__dict__['sig_len']
    resample_num = int(sampleNum * (FS / record.__dict__['fs']))
    for k in range(12):
        resample_data[k] = signal.resample(data[k], resample_num, axis=0, window=None)
    data = (resample_data.values).T
    rpeaks_indices_0 = biosppy.signals.ecg.hamilton_segmenter(signal=WTfilt_1d(data[0]), sampling_rate = FS)
    rpeaks_indices_1 = biosppy.signals.ecg.hamilton_segmenter(signal=WTfilt_1d(data[1]), sampling_rate = FS)
    rpeaks_indices_2 = biosppy.signals.ecg.hamilton_segmenter(signal=WTfilt_1d(data[2]), sampling_rate = FS)
    rpeaks_indices_3 = biosppy.signals.ecg.hamilton_segmenter(signal=WTfilt_1d(data[3]), sampling_rate = FS)
    rpeaks_indices_4 = biosppy.signals.ecg.hamilton_segmenter(signal=WTfilt_1d(data[4]), sampling_rate = FS)
    rpeaks_indices_5 = biosppy.signals.ecg.hamilton_segmenter(signal=WTfilt_1d(data[5]), sampling_rate = FS)
    rpeaks_indices_6 = biosppy.signals.ecg.hamilton_segmenter(signal=WTfilt_1d(data[6]), sampling_rate = FS)
    rpeaks_indices_7 = biosppy.signals.ecg.hamilton_segmenter(signal=WTfilt_1d(data[7]), sampling_rate = FS)
    rpeaks_indices_8 = biosppy.signals.ecg.hamilton_segmenter(signal=WTfilt_1d(data[8]), sampling_rate = FS)
    rpeaks_indices_9 = biosppy.signals.ecg.hamilton_segmenter(signal=WTfilt_1d(data[9]), sampling_rate = FS)
    rpeaks_indices_10 = biosppy.signals.ecg.hamilton_segmenter(signal=WTfilt_1d(data[10]), sampling_rate = FS)
    rpeaks_indices_11 = biosppy.signals.ecg.hamilton_segmenter(signal=WTfilt_1d(data[11]), sampling_rate = FS)
    tol_para = 0.1
    rpeaks_indices_0_c = biosppy.signals.ecg.correct_rpeaks(signal=WTfilt_1d(data[0]), rpeaks=rpeaks_indices_0[0],sampling_rate=FS, tol=tol_para)
    rpeaks_indices_1_c = biosppy.signals.ecg.correct_rpeaks(signal=WTfilt_1d(data[1]), rpeaks=rpeaks_indices_1[0],sampling_rate=FS, tol=tol_para)
    rpeaks_indices_2_c = biosppy.signals.ecg.correct_rpeaks(signal=WTfilt_1d(data[2]), rpeaks=rpeaks_indices_2[0], sampling_rate=FS, tol=0.05)
    rpeaks_indices_3_c = biosppy.signals.ecg.correct_rpeaks(signal=WTfilt_1d(data[3]), rpeaks=rpeaks_indices_3[0],sampling_rate=FS, tol=0.05)
    rpeaks_indices_3_c_avR = []
    for each_R in range(1,len(rpeaks_indices_3_c[0])):
        R_P = rpeaks_indices_3_c[0][each_R]
        R_P_correct = np.argmin(WTfilt_1d(data[3])[R_P - 30:R_P + 20]) + R_P - 30
        rpeaks_indices_3_c_avR.append(R_P_correct)
    rpeaks_indices_3_c_avR = np.array(rpeaks_indices_3_c_avR)
    rpeaks_indices_4_c = biosppy.signals.ecg.correct_rpeaks(signal=WTfilt_1d(data[4]), rpeaks=rpeaks_indices_4[0],sampling_rate=FS, tol=0.05)
    rpeaks_indices_5_c = biosppy.signals.ecg.correct_rpeaks(signal=WTfilt_1d(data[5]), rpeaks=rpeaks_indices_5[0],sampling_rate=FS, tol=tol_para)
    rpeaks_indices_6_c = biosppy.signals.ecg.correct_rpeaks(signal=WTfilt_1d(data[6]), rpeaks=rpeaks_indices_6[0],sampling_rate=FS, tol=0.05)
    rpeaks_indices_7_c = biosppy.signals.ecg.correct_rpeaks(signal=WTfilt_1d(data[7]), rpeaks=rpeaks_indices_7[0],sampling_rate=FS, tol=tol_para)
    rpeaks_indices_8_c = biosppy.signals.ecg.correct_rpeaks(signal=WTfilt_1d(data[8]), rpeaks=rpeaks_indices_8[0],sampling_rate=FS, tol=tol_para)
    rpeaks_indices_9_c = biosppy.signals.ecg.correct_rpeaks(signal=WTfilt_1d(data[9]), rpeaks=rpeaks_indices_9[0],sampling_rate=FS, tol=tol_para)
    rpeaks_indices_10_c = biosppy.signals.ecg.correct_rpeaks(signal=WTfilt_1d(data[10]), rpeaks=rpeaks_indices_10[0],sampling_rate=FS, tol=tol_para)
    rpeaks_indices_11_c = biosppy.signals.ecg.correct_rpeaks(signal=WTfilt_1d(data[11]), rpeaks=rpeaks_indices_11[0],sampling_rate=FS, tol=tol_para)

    for j in range(len(rpeaks_indices_0_c[0])-1):
        s0.create_dataset(result[i][0]+str(j), data = WTfilt_1d(data[0]).tolist()[rpeaks_indices_0_c[0][j]:rpeaks_indices_0_c[0][j + 1]])
    for j in range(len(rpeaks_indices_1_c[0])-1):
        s1.create_dataset(result[i][0]+str(j), data = WTfilt_1d(data[1]).tolist()[rpeaks_indices_1_c[0][j]:rpeaks_indices_1_c[0][j + 1]])
    for j in range(len(rpeaks_indices_2_c[0])-1):
        s2.create_dataset(result[i][0]+str(j),data = WTfilt_1d(data[2]).tolist()[rpeaks_indices_2_c[0][j]:rpeaks_indices_2_c[0][j + 1]])
    for j in range(len(rpeaks_indices_3_c_avR)-1):
        s3.create_dataset(result[i][0]+str(j), data = WTfilt_1d(data[3]).tolist()[rpeaks_indices_3_c_avR[j]:rpeaks_indices_3_c_avR[j + 1]])
    for j in range(len(rpeaks_indices_4_c[0])-1):
        s4.create_dataset(result[i][0]+str(j),data = WTfilt_1d(data[4]).tolist()[rpeaks_indices_4_c[0][j]:rpeaks_indices_4_c[0][j + 1]])
    for j in range(len(rpeaks_indices_5_c[0])-1):
        s5.create_dataset(result[i][0]+str(j),data=WTfilt_1d(data[5]).tolist()[rpeaks_indices_5_c[0][j]:rpeaks_indices_5_c[0][j + 1]])
    for j in range(len(rpeaks_indices_6_c[0]) - 1):
        s6.create_dataset(result[i][0]+str(j), data = WTfilt_1d(data[6]).tolist()[rpeaks_indices_6_c[0][j]:rpeaks_indices_6_c[0][j + 1]])
    for j in range(len(rpeaks_indices_7_c[0])-1):
        s7.create_dataset(result[i][0]+str(j), data=WTfilt_1d(data[7]).tolist()[rpeaks_indices_7_c[0][j]:rpeaks_indices_7_c[0][j + 1]])
    for j in range(len(rpeaks_indices_8_c[0])-1):
        s8.create_dataset(result[i][0]+str(j),data= WTfilt_1d(data[8]).tolist()[rpeaks_indices_8_c[0][j]:rpeaks_indices_8_c[0][j + 1]])
    for j in range(len(rpeaks_indices_9_c[0])-1):
        s9.create_dataset(result[i][0]+str(j),data=WTfilt_1d(data[9]).tolist()[rpeaks_indices_9_c[0][j]:rpeaks_indices_9_c[0][j + 1]])
    for j in range(len(rpeaks_indices_10_c[0])-1):
        s10.create_dataset(result[i][0]+str(j), data=WTfilt_1d(data[10]).tolist()[rpeaks_indices_10_c[0][j]:rpeaks_indices_10_c[0][j + 1]])
    for j in range(len(rpeaks_indices_11_c[0])-1):
        s11.create_dataset(result[i][0]+str(j),data=WTfilt_1d(data[11]).tolist()[rpeaks_indices_11_c[0][j]:rpeaks_indices_11_c[0][j + 1]])
hf_short.close()

