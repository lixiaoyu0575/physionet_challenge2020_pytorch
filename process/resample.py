import sys
import pandas as pd
import numpy as np
import scipy.io as scio
import os
import wfdb
from scipy import signal
import csv

#####采样后的新mat文件已经是除过ADC_gain了
def resampleECG(resampleFS, path):
# 函数功能：对path下的数据做resample
# resampleFS: resample rate
# path: 数据所在文件夹

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
    f = open('sample_rate.csv', 'w', encoding='utf-8')

    csv_writer = csv.writer(f)

    for j in range(0, len(dat_collection)):
        print(dat_collection[j])
        csv_writer.writerow([dat_collection[j], '300'])
        record = wfdb.rdrecord(path + dat_collection[j])
        sampleNum = record.__dict__['sig_len']
        resample_num = int(sampleNum * (resampleFS / record.__dict__['fs']))
        data = pd.DataFrame(record.__dict__['p_signal'], columns=record.__dict__['sig_name'])
        resample_data_300HZ = pd.DataFrame()
        for k in record.__dict__['sig_name']:
            resample_data_300HZ[k] = signal.resample(data[k], resample_num, axis=0, window=None)
        scio.savemat(dat_collection[j] + '.mat', {'val': resample_data_300HZ.values.T})
    f.close()

resampleECG(300,'/home/weiyuhua/Data/challenge2020/All_data/')
