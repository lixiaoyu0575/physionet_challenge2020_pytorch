#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 08:50:07 2017

@author: shenda
"""
import csv
import dill
import numpy as np
import pandas as pd
import random
from collections import OrderedDict
from features_centerwave import get_centerwave_feature
from features_long import get_long_feature
from features_qrs import get_qrs_feature
from features_short import get_short_stat_wave_feature
from features_short import get_short_stat_wave_feature_from_avR
from preprocess_do_not_save_version import get_preprocessed_data

from features_centerwave import get_short_centerwave
#from features_short import get_short_feature

##################################################
### tools
##################################################
def CombineFeatures(table1, table2):
    '''
    table1 and table2 should have the same length
    '''
    table = []
    n_row = len(table1)
    for i in range(n_row):
        table.append(table1[i]+table2[i])
        
    return table

def RandomNum(ts):
    '''
    baseline feature
    '''
    return [random.random()]

##################################################
### get all features
##################################################



def GetAllFeature_test( patient_info,short_table0,short_table1,short_table2,
                        short_table3,short_table4,short_table5,
                        short_table6,short_table7,short_table8,
                        short_table9,short_table10,short_table11,
                        long_table0,long_table1, long_table2,long_table3,
                        long_table4,long_table5,long_table6,
                        long_table7,long_table8,long_table9,
                        long_table10,long_table11,
                        QRS_table, long_pid_list,
                        short_pid_0,short_pid_1,short_pid_2,
                        short_pid_3,short_pid_4,short_pid_5,
                        short_pid_6,short_pid_7,short_pid_8,
                        short_pid_9,short_pid_10,short_pid_11):
    '''
    get all features for test, without feature name, do not need precomputed center_waves
    
    input:
        data: short_table, long_table, QRS_table
        pid: long_pid_list, short_pid_list
    output:
        out_feature: 8528 rows
        
    1. centerwave_feature
    2. long_feature
    3. qrs_feature
    4. short_stat_wave_feature
    '''

    '''
    
    center_waves_lead1 = get_short_centerwave(short_table1, short_pid_list1, long_pid_list)
    center_waves_lead2 = get_short_centerwave(short_table2, short_pid_list2, long_pid_list)
    center_waves_lead3 = get_short_centerwave(short_table3, short_pid_list3, long_pid_list)
    center_waves_lead4 = get_short_centerwave(short_table4, short_pid_list4, long_pid_list)
    center_waves_lead5 = get_short_centerwave(short_table5, short_pid_list5, long_pid_list)
    center_waves_lead6 = get_short_centerwave(short_table6, short_pid_list6, long_pid_list)
    center_waves_lead7 = get_short_centerwave(short_table7, short_pid_list7, long_pid_list)
    center_waves_lead8 = get_short_centerwave(short_table8, short_pid_list8, long_pid_list)
    center_waves_lead9 = get_short_centerwave(short_table9, short_pid_list9, long_pid_list)
    center_waves_lead10 = get_short_centerwave(short_table10, short_pid_list10, long_pid_list)
    center_waves_lead11 = get_short_centerwave(short_table11, short_pid_list11, long_pid_list)
    center_waves_lead12 = get_short_centerwave(short_table12, short_pid_list12, long_pid_list)

    _, centerwave_feature_1 = get_centerwave_feature(center_waves_lead1)
    _, centerwave_feature_2 = get_centerwave_feature(center_waves_lead2)
    _, centerwave_feature_3 = get_centerwave_feature(center_waves_lead3)
    _, centerwave_feature_4 = get_centerwave_feature(center_waves_lead4)
    _, centerwave_feature_5 = get_centerwave_feature(center_waves_lead5)
    _, centerwave_feature_6 = get_centerwave_feature(center_waves_lead6)
    _, centerwave_feature_7 = get_centerwave_feature(center_waves_lead7)
    _, centerwave_feature_8 = get_centerwave_feature(center_waves_lead8)
    _, centerwave_feature_9 = get_centerwave_feature(center_waves_lead9)
    _, centerwave_feature_10 = get_centerwave_feature(center_waves_lead10)
    _, centerwave_feature_11 = get_centerwave_feature(center_waves_lead11)
    _, centerwave_feature_12 = get_centerwave_feature(center_waves_lead12)
    '''

    _, short_stat_wave_feature_0 = get_short_stat_wave_feature(short_table0, short_pid_0, long_pid_list)
    _, short_stat_wave_feature_1 = get_short_stat_wave_feature(short_table1, short_pid_1, long_pid_list)
    _, short_stat_wave_feature_2 = get_short_stat_wave_feature(short_table2, short_pid_2, long_pid_list)
    _, short_stat_wave_feature_3 = get_short_stat_wave_feature_from_avR(short_table3, short_pid_3, long_pid_list)
    _, short_stat_wave_feature_4 = get_short_stat_wave_feature(short_table4, short_pid_4, long_pid_list)
    _, short_stat_wave_feature_5 = get_short_stat_wave_feature(short_table5, short_pid_5, long_pid_list)
    _, short_stat_wave_feature_6 = get_short_stat_wave_feature(short_table6, short_pid_6, long_pid_list)
    _, short_stat_wave_feature_7 = get_short_stat_wave_feature(short_table7, short_pid_7, long_pid_list)
    _, short_stat_wave_feature_8 = get_short_stat_wave_feature(short_table8, short_pid_8, long_pid_list)
    _, short_stat_wave_feature_9 = get_short_stat_wave_feature(short_table9, short_pid_9, long_pid_list)
    _, short_stat_wave_feature_10 = get_short_stat_wave_feature(short_table10, short_pid_10, long_pid_list)
    _, short_stat_wave_feature_11 = get_short_stat_wave_feature(short_table11, short_pid_11, long_pid_list)

    _, long_feature_0 = get_long_feature(long_table0)
    _, long_feature_1 = get_long_feature(long_table1)
    _, long_feature_2 = get_long_feature(long_table2)
    _, long_feature_3 = get_long_feature(long_table3)
    _, long_feature_4 = get_long_feature(long_table4)
    _, long_feature_5 = get_long_feature(long_table5)
    _, long_feature_6 = get_long_feature(long_table6)
    _, long_feature_7 = get_long_feature(long_table7)
    _, long_feature_8 = get_long_feature(long_table8)
    _, long_feature_9 = get_long_feature(long_table9)
    _, long_feature_10 = get_long_feature(long_table10)
    _, long_feature_11 = get_long_feature(long_table11)




    _, qrs_feature = get_qrs_feature(QRS_table)
    patient_info = np.array(patient_info)

    '''
    out_feature = CombineFeatures(centerwave_feature,
                                  CombineFeatures(long_feature, 
                                                  CombineFeatures(qrs_feature, 
                                                                  short_stat_wave_feature)))
    
    ### TODO: potential bug, if last column all 0, may cause bug in xgboost
    # for feat in out_feature:
    #     if feat[-1] == 0.0:
    #         feat[-1] = 0.00000001
    '''
    # all features have the same row number:6877 
    final_array_row = qrs_feature.shape[0]
    final_array_column = patient_info.shape[1]+long_feature_0.shape[1]*12+qrs_feature.shape[1]+short_stat_wave_feature_0.shape[1]*12
    print('row,column'+str(final_array_row)+'.'+str(final_array_column))
    final_array = np.zeros((final_array_row,final_array_column),dtype=object)

    ###combine all features

    '''
    all_feature_name_list = [
        'qrs_feature','centerwave_feature_1','centerwave_feature_1','centerwave_feature_2','centerwave_feature_3',
        'centerwave_feature_4','centerwave_feature_5','centerwave_feature_6','centerwave_feature_7',
        'centerwave_feature_8','centerwave_feature_9','centerwave_feature_10','centerwave_feature_11',
        'centerwave_feature_12','long_feature_1','long_feature_2','long_feature_3','long_feature_4',
        'long_feature_5','long_feature_6','long_feature_7','long_feature_8','long_feature_9',
        'long_feature_10','long_feature_11','long_feature_12','short_stat_wave_feature_1',
        'short_stat_wave_feature_2','short_stat_wave_feature_3','short_stat_wave_feature_4',
        'short_stat_wave_feature_5','short_stat_wave_feature_6','short_stat_wave_feature_7',
        'short_stat_wave_feature_8','short_stat_wave_feature_9','short_stat_wave_feature_10',
        'short_stat_wave_feature_11','short_stat_wave_feature_12'
    ]
    '''
    all_feature_name_list = ['patient_info',
        'qrs_feature','long_feature_0','long_feature_1','long_feature_2','long_feature_3','long_feature_4',
        'long_feature_5','long_feature_6','long_feature_7','long_feature_8','long_feature_9',
        'long_feature_10','long_feature_11','short_stat_wave_feature_0','short_stat_wave_feature_1',
        'short_stat_wave_feature_2','short_stat_wave_feature_3','short_stat_wave_feature_4',
        'short_stat_wave_feature_5','short_stat_wave_feature_6','short_stat_wave_feature_7',
        'short_stat_wave_feature_8','short_stat_wave_feature_9','short_stat_wave_feature_10',
        'short_stat_wave_feature_11'
    ]

    column_sum = 0
    for each_feature in all_feature_name_list:
        final_array[:,column_sum:column_sum+eval(each_feature).shape[1]] = eval(each_feature)
        column_sum += eval(each_feature).shape[1]
    # hhc_feature = pd.DataFrame(data = final_array)
    # hhc_feature.to_csv(result_name+'_OUT_FEATURE.csv', sep=',', header=True, index=False)
    return final_array.tolist()




#########################
### main
#########################

if __name__ == '__main__':
    with open('/home/hanhaochen/physionet_challenge2020_pytorch/features/Preprocess_Data/REFERENCE.csv', 'r') as f_r:
        reader = csv.reader(f_r)
        result = list(reader)
    f = open('Out_Feature_New_Data.csv', 'w', encoding='utf-8', newline='')
    write_f= csv.writer(f)
    ###记录问题数据
    f_error = open('error_data.csv', 'w', encoding='utf-8', newline='')
    csv_writer = csv.writer(f_error)
    for i in range(len(result)):
        print('#######################'+result[i][0]+'######################')
        try:
            ecg_data = result[i:i+1]
            patient_info,short_data0,short_data1,short_data2,short_data3,short_data4,short_data5,short_data6,short_data7,short_data8,short_data9,short_data10,short_data11,long_data0,long_data1,long_data2,long_data3,long_data4,long_data5,long_data6,long_data7,long_data8,long_data9,long_data10,long_data11,qrs_info, long_pid, short_pid0,short_pid1,short_pid2,short_pid3,short_pid4,short_pid5,short_pid6,short_pid7,short_pid8,short_pid9,short_pid10,short_pid11 = get_preprocessed_data(ecg_data)
            each_feature = GetAllFeature_test(patient_info,short_data0,short_data1,short_data2,short_data3,short_data4,short_data5,short_data6,short_data7,short_data8,short_data9,short_data10,short_data11,long_data0,long_data1,long_data2,long_data3,long_data4,long_data5,long_data6,long_data7,long_data8,long_data9,long_data10,long_data11,qrs_info, long_pid, short_pid0,short_pid1,short_pid2,short_pid3,short_pid4,short_pid5,short_pid6,short_pid7,short_pid8,short_pid9,short_pid10,short_pid11)
            write_f.writerow(each_feature[0])
        except ValueError:
            csv_writer.writerow([result[i][0]])
    f.close()
    f_error.close()


