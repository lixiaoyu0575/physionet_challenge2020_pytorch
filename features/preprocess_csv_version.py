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
FS = 500

# ##### write long
# f_l0 = open('/DATASET/challenge2020/Prepreocess_Data_CSV/long0.csv','w',encoding='utf-8',newline='')
# f_l1 = open('/DATASET/challenge2020/Prepreocess_Data_CSV/long1.csv','w',encoding='utf-8',newline='')
# f_l2 = open('/DATASET/challenge2020/Prepreocess_Data_CSV/long2.csv','w',encoding='utf-8',newline='')
# f_l3 = open('/DATASET/challenge2020/Prepreocess_Data_CSV/long3.csv','w',encoding='utf-8',newline='')
# f_l4 = open('/DATASET/challenge2020/Prepreocess_Data_CSV/long4.csv','w',encoding='utf-8',newline='')
# f_l5 = open('/DATASET/challenge2020/Prepreocess_Data_CSV/long5.csv','w',encoding='utf-8',newline='')
# f_l6 = open('/DATASET/challenge2020/Prepreocess_Data_CSV/long6.csv','w',encoding='utf-8',newline='')
# f_l7 = open('/DATASET/challenge2020/Prepreocess_Data_CSV/long7.csv','w',encoding='utf-8',newline='')
# f_l8 = open('/DATASET/challenge2020/Prepreocess_Data_CSV/long8.csv','w',encoding='utf-8',newline='')
# f_l9 = open('/DATASET/challenge2020/Prepreocess_Data_CSV/long9.csv','w',encoding='utf-8',newline='')
# f_l10 = open('/DATASET/challenge2020/Prepreocess_Data_CSV/long10.csv','w',encoding='utf-8',newline='')
# f_l11 = open('/DATASET/challenge2020/Prepreocess_Data_CSV/long11.csv','w',encoding='utf-8',newline='')
#
# for i in range(len(result)):
#     print(i)
#     print(result[i][0])
#     record = wfdb.rdrecord('/DATASET/challenge2020/All_data/' + result[i][0])
#     data = record.__dict__['p_signal']
#     data = np.transpose(data)
#     resample_data = pd.DataFrame()
#     sampleNum = record.__dict__['sig_len']
#     resample_num = int(sampleNum * (FS / record.__dict__['fs']))
#     for k in range(12):
#         resample_data[k] = signal.resample(data[k], resample_num, axis=0, window=None)
#     data = (resample_data.values).T
#     write_long0 = csv.writer(f_l0)
#     write_long1 = csv.writer(f_l1)
#     write_long2 = csv.writer(f_l2)
#     write_long3 = csv.writer(f_l3)
#     write_long4 = csv.writer(f_l4)
#     write_long5 = csv.writer(f_l5)
#     write_long6 = csv.writer(f_l6)
#     write_long7 = csv.writer(f_l7)
#     write_long8 = csv.writer(f_l8)
#     write_long9 = csv.writer(f_l9)
#     write_long10 = csv.writer(f_l10)
#     write_long11 = csv.writer(f_l11)
#     write_long0.writerow([result[i][0], 'DX'] + WTfilt_1d(data[0]).tolist())
#     write_long1.writerow([result[i][0], 'DX'] + WTfilt_1d(data[1]).tolist())
#     write_long2.writerow([result[i][0], 'DX'] + WTfilt_1d(data[2]).tolist())
#     write_long3.writerow([result[i][0], 'DX'] + WTfilt_1d(data[3]).tolist())
#     write_long4.writerow([result[i][0], 'DX'] + WTfilt_1d(data[4]).tolist())
#     write_long5.writerow([result[i][0], 'DX'] + WTfilt_1d(data[5]).tolist())
#     write_long6.writerow([result[i][0], 'DX'] + WTfilt_1d(data[6]).tolist())
#     write_long7.writerow([result[i][0], 'DX'] + WTfilt_1d(data[7]).tolist())
#     write_long8.writerow([result[i][0], 'DX'] + WTfilt_1d(data[8]).tolist())
#     write_long9.writerow([result[i][0], 'DX'] + WTfilt_1d(data[9]).tolist())
#     write_long10.writerow([result[i][0], 'DX'] + WTfilt_1d(data[10]).tolist())
#     write_long11.writerow([result[i][0], 'DX'] + WTfilt_1d(data[11]).tolist())
# f_l0.close()
# f_l1.close()
# f_l2.close()
# f_l3.close()
# f_l4.close()
# f_l5.close()
# f_l6.close()
# f_l7.close()
# f_l8.close()
# f_l9.close()
# f_l10.close()
# f_l11.close()
#
#
# ##### write QRS_info
# f_qrsinfo = open('/DATASET/challenge2020/Prepreocess_Data_CSV/QRSinfo.csv','w',encoding='utf-8',newline='')
# for i in range(len(result)):
#     print(result[i][0])
#     record = wfdb.rdrecord('/DATASET/challenge2020/All_data/' + result[i][0])
#     data = record.__dict__['p_signal']
#     data = np.transpose(data)
#     resample_data = pd.DataFrame()
#     sampleNum = record.__dict__['sig_len']
#     resample_num = int(sampleNum * (FS / record.__dict__['fs']))
#     for k in range(12):
#         resample_data[k] = signal.resample(data[k], resample_num, axis=0, window=None)
#     data = (resample_data.values).T
#     signal_data = WTfilt_1d(data[1])
#     rpeaks_indices_1 = biosppy.signals.ecg.hamilton_segmenter(signal=signal_data, sampling_rate=FS)
#     rpeaks_indices_2 = biosppy.signals.ecg.correct_rpeaks(signal=signal_data, rpeaks=rpeaks_indices_1[0],sampling_rate=FS)
#     write_qrs = csv.writer(f_qrsinfo)
#     write_qrs.writerow([result[i][0], 'DX'] + np.diff(rpeaks_indices_2[0]).tolist())
#
# f_qrsinfo.close()


#### write short_data
f_s0 = open('/DATASET/challenge2020/Prepreocess_Data_CSV/short0.csv','w',encoding='utf-8',newline='')
f_s1 = open('/DATASET/challenge2020/Prepreocess_Data_CSV/short1.csv','w',encoding='utf-8',newline='')
f_s2 = open('/DATASET/challenge2020/Prepreocess_Data_CSV/short2.csv','w',encoding='utf-8',newline='')
f_s3 = open('/DATASET/challenge2020/Prepreocess_Data_CSV/short3.csv','w',encoding='utf-8',newline='')
f_s4 = open('/DATASET/challenge2020/Prepreocess_Data_CSV/short4.csv','w',encoding='utf-8',newline='')
f_s5 = open('/DATASET/challenge2020/Prepreocess_Data_CSV/short5.csv','w',encoding='utf-8',newline='')
f_s6 = open('/DATASET/challenge2020/Prepreocess_Data_CSV/short6.csv','w',encoding='utf-8',newline='')
f_s7 = open('/DATASET/challenge2020/Prepreocess_Data_CSV/short7.csv','w',encoding='utf-8',newline='')
f_s8 = open('/DATASET/challenge2020/Prepreocess_Data_CSV/short8.csv','w',encoding='utf-8',newline='')
f_s9 = open('/DATASET/challenge2020/Prepreocess_Data_CSV/short9.csv','w',encoding='utf-8',newline='')
f_s10 = open('/DATASET/challenge2020/Prepreocess_Data_CSV/short10.csv','w',encoding='utf-8',newline='')
f_s11 = open('/DATASET/challenge2020/Prepreocess_Data_CSV/short11.csv','w',encoding='utf-8',newline='')

for i in range(len(result)):
    write_short0 = csv.writer(f_s0)
    write_short1 = csv.writer(f_s1)
    write_short2 = csv.writer(f_s2)
    write_short3 = csv.writer(f_s3)
    write_short4 = csv.writer(f_s4)
    write_short5 = csv.writer(f_s5)
    write_short6 = csv.writer(f_s6)
    write_short7 = csv.writer(f_s7)
    write_short8 = csv.writer(f_s8)
    write_short9 = csv.writer(f_s9)
    write_short10 = csv.writer(f_s10)
    write_short11 = csv.writer(f_s11)
    print('short'+result[i][0])
    record = wfdb.rdrecord('/DATASET/challenge2020/All_data/' + result[i][0])
    data = record.__dict__['p_signal']
    data = np.transpose(data)
    resample_data = pd.DataFrame()
    sampleNum = record.__dict__['sig_len']
    resample_num = int(sampleNum * (FS / record.__dict__['fs']))
    for k in range(12):
        resample_data[k] = signal.resample(data[k], resample_num, axis=0, window=None)
    data = (resample_data.values).T
    rpeaks_indices_0 = biosppy.signals.ecg.hamilton_segmenter(signal=WTfilt_1d(data[0]), sampling_rate=FS)
    rpeaks_indices_1 = biosppy.signals.ecg.hamilton_segmenter(signal=WTfilt_1d(data[1]), sampling_rate=FS)
    rpeaks_indices_2 = biosppy.signals.ecg.hamilton_segmenter(signal=WTfilt_1d(data[2]), sampling_rate=FS)
    rpeaks_indices_3 = biosppy.signals.ecg.hamilton_segmenter(signal=WTfilt_1d(data[3]), sampling_rate=FS)
    rpeaks_indices_4 = biosppy.signals.ecg.hamilton_segmenter(signal=WTfilt_1d(data[4]), sampling_rate=FS)
    rpeaks_indices_5 = biosppy.signals.ecg.hamilton_segmenter(signal=WTfilt_1d(data[5]), sampling_rate=FS)
    rpeaks_indices_6 = biosppy.signals.ecg.hamilton_segmenter(signal=WTfilt_1d(data[6]), sampling_rate=FS)
    rpeaks_indices_7 = biosppy.signals.ecg.hamilton_segmenter(signal=WTfilt_1d(data[7]), sampling_rate=FS)
    rpeaks_indices_8 = biosppy.signals.ecg.hamilton_segmenter(signal=WTfilt_1d(data[8]), sampling_rate=FS)
    rpeaks_indices_9 = biosppy.signals.ecg.hamilton_segmenter(signal=WTfilt_1d(data[9]), sampling_rate=FS)
    rpeaks_indices_10 = biosppy.signals.ecg.hamilton_segmenter(signal=WTfilt_1d(data[10]), sampling_rate=FS)
    rpeaks_indices_11 = biosppy.signals.ecg.hamilton_segmenter(signal=WTfilt_1d(data[11]), sampling_rate=FS)
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
        write_short0.writerow([result[i][0], 'DX'] + WTfilt_1d(data[0]).tolist()[rpeaks_indices_0_c[0][j]:rpeaks_indices_0_c[0][j + 1]])
    for j in range(len(rpeaks_indices_1_c[0])-1):
        write_short1.writerow([result[i][0], 'DX'] + WTfilt_1d(data[1]).tolist()[rpeaks_indices_1_c[0][j]:rpeaks_indices_1_c[0][j + 1]])
    for j in range(len(rpeaks_indices_2_c[0])-1):
        write_short2.writerow([result[i][0], 'DX'] + WTfilt_1d(data[2]).tolist()[rpeaks_indices_2_c[0][j]:rpeaks_indices_2_c[0][j + 1]])
    for j in range(len(rpeaks_indices_3_c_avR)-1):
        write_short3.writerow([result[i][0], 'DX'] + WTfilt_1d(data[3]).tolist()[rpeaks_indices_3_c_avR[j]:rpeaks_indices_3_c_avR[j + 1]])
    for j in range(len(rpeaks_indices_4_c[0])-1):
        write_short4.writerow([result[i][0], 'DX'] + WTfilt_1d(data[4]).tolist()[rpeaks_indices_4_c[0][j]:rpeaks_indices_4_c[0][j + 1]])
    for j in range(len(rpeaks_indices_5_c[0])-1):
        write_short5.writerow([result[i][0], 'DX'] + WTfilt_1d(data[5]).tolist()[rpeaks_indices_5_c[0][j]:rpeaks_indices_5_c[0][j + 1]])
    for j in range(len(rpeaks_indices_6_c[0]) - 1):
        write_short6.writerow([result[i][0], 'DX'] + WTfilt_1d(data[6]).tolist()[rpeaks_indices_6_c[0][j]:rpeaks_indices_6_c[0][j + 1]])
    for j in range(len(rpeaks_indices_7_c[0])-1):
        write_short7.writerow([result[i][0], 'DX'] + WTfilt_1d(data[7]).tolist()[rpeaks_indices_7_c[0][j]:rpeaks_indices_7_c[0][j + 1]])
    for j in range(len(rpeaks_indices_8_c[0])-1):
        write_short8.writerow([result[i][0], 'DX'] + WTfilt_1d(data[8]).tolist()[rpeaks_indices_8_c[0][j]:rpeaks_indices_8_c[0][j + 1]])
    for j in range(len(rpeaks_indices_9_c[0])-1):
        write_short9.writerow([result[i][0], 'DX'] + WTfilt_1d(data[9]).tolist()[rpeaks_indices_9_c[0][j]:rpeaks_indices_9_c[0][j + 1]])
    for j in range(len(rpeaks_indices_10_c[0])-1):
        write_short10.writerow([result[i][0], 'DX'] + WTfilt_1d(data[10]).tolist()[rpeaks_indices_10_c[0][j]:rpeaks_indices_10_c[0][j + 1]])
    for j in range(len(rpeaks_indices_11_c[0])-1):
        write_short11.writerow([result[i][0], 'DX'] + WTfilt_1d(data[11]).tolist()[rpeaks_indices_11_c[0][j]:rpeaks_indices_11_c[0][j + 1]])
f_s0.close()
f_s1.close()
f_s2.close()
f_s3.close()
f_s4.close()
f_s5.close()
f_s6.close()
f_s7.close()
f_s8.close()
f_s9.close()
f_s10.close()
f_s11.close()
