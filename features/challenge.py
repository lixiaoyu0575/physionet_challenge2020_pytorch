#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 23:00:34 2017

@author: shenda
"""

from collections import Counter
import numpy as np
import FeatureExtract
import MyEval
import ReadData
import dill
import features_all
##import challenge_encase_mimic

##############
#### load classifier
###############
#with open('model/v2.5_xgb5_all.pkl', 'rb') as my_in:
#    clf_final = dill.load(my_in)

##############
#### read and extract
###############
##short_pid1-12   short_label1-12 are same

long_pid0, long_data0, long_label0 = ReadData.ReadData( '../data1/long0.csv' )
long_pid1, long_data1, long_label1 = ReadData.ReadData( '../data1/long1.csv' )
long_pid2, long_data2, long_label2 = ReadData.ReadData( '../data1/long2.csv' )
long_pid3, long_data3, long_label3 = ReadData.ReadData( '../data1/long3.csv' )
long_pid4, long_data4, long_label4 = ReadData.ReadData( '../data1/long4.csv' )
long_pid5, long_data5, long_label5 = ReadData.ReadData( '../data1/long5.csv' )
long_pid6, long_data6, long_label6 = ReadData.ReadData( '../data1/long6.csv' )
long_pid7, long_data7, long_label7 = ReadData.ReadData( '../data1/long7.csv' )
long_pid8, long_data8, long_label8 = ReadData.ReadData( '../data1/long8.csv' )
long_pid9, long_data9, long_label9 = ReadData.ReadData( '../data1/long9.csv' )
long_pid10, long_data10, long_label10 = ReadData.ReadData( '../data1/long10.csv' )
long_pid11, long_data11, long_label11 = ReadData.ReadData( '../data1/long11.csv' )

short_pid0, short_data0, short_label0 = ReadData.ReadData( '../data1/short0.csv' )
short_pid1, short_data1, short_label1 = ReadData.ReadData( '../data1/short1.csv' )
short_pid2, short_data2, short_label2 = ReadData.ReadData( '../data1/short2.csv' )
short_pid3, short_data3, short_label3 = ReadData.ReadData( '../data1/short3.csv' )
short_pid4, short_data4, short_label4 = ReadData.ReadData( '../data1/short4.csv' )
short_pid5, short_data5, short_label5 = ReadData.ReadData( '../data1/short5.csv' )
short_pid6, short_data6, short_label6 = ReadData.ReadData( '../data1/short6.csv' )
short_pid7, short_data7, short_label7 = ReadData.ReadData( '../data1/short7.csv' )
short_pid8, short_data8, short_label8 = ReadData.ReadData( '../data1/short8.csv' )
short_pid9, short_data9, short_label9 = ReadData.ReadData( '../data1/short9.csv' )
short_pid10, short_data10, short_label0 = ReadData.ReadData( '../data1/short10.csv' )
short_pid11, short_data11, short_label11 = ReadData.ReadData( '../data1/short11.csv' )



QRS_pid, QRS_data, QRS_label = ReadData.ReadData( '../data1/QRSinfo.csv' )


#############
### feature
#############
all_feature = features_all.GetAllFeature_test(short_data0,short_data1,short_data2,short_data3,
                                            short_data4,short_data5,short_data6,
                                            short_data7,short_data8,short_data9,
                                            short_data10,short_data11,long_data0,
                                            long_data1,long_data2,long_data3,
                                            long_data4,long_data5,long_data6,
                                            long_data7,long_data8,long_data9,
                                            long_data10,long_data11,
                                             QRS_data, long_pid1, short_pid0,
                                              short_pid1,short_pid2,short_pid3,
                                              short_pid4,short_pid5,short_pid6,
                                              short_pid7,short_pid8,short_pid9,
                                              short_pid10,short_pid11)
#out_feats = features_mimic.get_mimic_feature(long_data[0])

'''
############
## classifier
############
pred = []
pred = challenge_encase_mimic.pred_one_sample(short_data, long_data, QRS_data, long_pid, short_pid)

fout= open('../answers1.txt','a')
print(pred[0])
fout.write(pred[0])
fout.write('\n')
fout.close
'''
