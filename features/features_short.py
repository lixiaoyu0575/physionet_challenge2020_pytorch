# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 21:32:52 2017

@author: hsd
"""

import numpy as np
from scipy import stats
from scipy import signal
##################################################
### tools
##################################################
def LongThresCrossing(ts, thres):
    cnt = 0
    pair_flag = 1
    pre_loc = 0
    width = []
    for i in range(len(ts)-1):
        if (ts[i] - thres) * (ts[i+1] - thres) < 0:
            cnt += 1
            if pair_flag == 1:
                width.append(i-pre_loc)
                pair_flag = 0
            else:
                pair_flag = 1
                pre_loc = i
        if ts[i] == thres and (ts[i-1] - thres) * (ts[i+1] - thres) < 0:
            cnt += 1
    
    if len(width) > 1:
        return [cnt, np.mean(width)]
    else:
        return [cnt, 0.0]
def get_S_peak(ts):
    k = -5
    ts_s = ts[:-1]
    S_peak = min(ts_s)
    S_loc = np.argmin(ts_s)
    while S_peak > ts[S_loc+1] and k >= -10:
        ts = ts[:k]
        ts_s = ts[:-1]
        S_peak = min(ts_s)
        S_loc = np.argmin(ts_s)
        k -= 5
    return S_loc,S_peak
##################################################
### get features
##################################################

def short_basic_stat(ts):
    global feature_list
    Range = max(ts) - min(ts)
    Var = np.var(ts)
    Skew = stats.skew(ts)
    Kurtosis = stats.kurtosis(ts)
    Median = np.median(ts)
    feature_list.extend(['ShortBasicStat_Range', 
                         'ShortBasicStat_Var', 
                         'ShortBasicStat_Skew', 
                         'ShortBasicStat_Kurtosis', 
                         'ShortBasicStat_Median'])
    return [Range, Var, Skew, Kurtosis, Median]

def short_zero_crossing(ts):

    global feature_list
    feature_list.extend(['short_zero_crossing_cnt'])

    cnt = 0
    for i in range(len(ts)-1):
        if ts[i] * ts[i+1] < 0:
            cnt += 1
        if ts[i] == 0 and ts[i-1] * ts[i+1] < 0:
            cnt += 1
    return [cnt]



##################################################
###  get all features
##################################################
def get_short_stat_wave_feature(table, pid_list, long_pid_list):
    '''
    short stat feature, actually long feature
    
    Electrocardiogram Feature Extraction and Pattern Recognition Using a Novel Windowing Algorithm
    
    row of out feature is 8000+
    
    TODO: more on how to detect PT waves
    '''

    print('extract GetShortStatWaveFeature begin')

    features = []
    pid_short_dic = {}
    
                
    ### no-preprocess, performs better
    for i in range(len(pid_list)):
        if pid_list[i] in pid_short_dic.keys():
            pid_short_dic[pid_list[i]].append(table[i])
        else:
            pid_short_dic[pid_list[i]] = [table[i]]
                
    step = 0
    for pid in long_pid_list:
        try:
            if pid in pid_short_dic.keys() and len(pid_short_dic[pid])-2 > 0:

                ### init
                QRS_peak_list = []
                QRS_area_list = []

                PR_duration_list = []
                PR_duration_corrected_list = []
                QRS_duration_list = []
                QRS_duration_corrected_list = []
                QT_duration_list = []
                QT_corrected_list = []
                vent_rate_list = []

                NF_list = []
                Fwidth_list = []

                RQ_amp_list = []
                RS_amp_list = []
                ST_amp_list = []
                PQ_amp_list = []
                QS_amp_list = []
                RP_amp_list = []
                RT_amp_list = []
                ST_duration_list = []
                ST_duration_corrected_list = []
                RS_duration_list = []
                RS_duration_corrected_list = []
                T_peak_list = []
                P_peak_list = []
                Q_peak_list = []
                R_peak_list = []
                S_peak_list = []
                RS_slope_list = []
                ST_slope_list = []
                PP_amp_list = []
                TT_amp_list = []
                PP_duration_list = []
                TT_duration_list = []

                ### select short data of one patient
                sub_table = pid_short_dic[pid]

                for i in range(len(sub_table)-2):
                    prev_ts = sub_table[i]
                    ts = sub_table[i+1]
                    try:
                        ### select each short data
                        T_start = round(0.15 * len(ts))
                        T_end = round(0.55 * len(ts))
                        T_p_start = round(0.15 * len(prev_ts))
                        T_p_end = round(0.55 * len(prev_ts))
                        P_start = round(0.65 * len(prev_ts))
                        P_end = round(0.95 * len(prev_ts))
                        P_n_start = round(0.65 * len(ts))
                        P_n_end = round(0.95 * len(ts))
                        T_wave = ts[T_start:T_end]
                        if T_wave == []:
                            T_wave = [0]
                        T_p_wave = prev_ts[T_p_start:T_p_end]
                        if T_p_wave == []:
                            T_p_wave = [0]
                        P_wave = prev_ts[P_start:P_end]
                        if P_wave == []:
                            P_wave = [0]
                        P_n_wave = ts[P_n_start:P_n_end]
                        T_peak = max(T_wave)
                        T_p_peak = max(T_p_wave)
                        P_n_peak = max(P_n_wave)
                        P_peak = max(P_wave)
                        Q_peak = min(prev_ts[-30:])
                        R_peak = ts[0]
                        S_loc, S_peak = get_S_peak(ts[:35])
                        T_loc = np.argmax(T_wave) + T_start
                        T_p_loc = np.argmax(T_p_wave) + T_p_start - len(prev_ts)
                        P_loc = P_start + np.argmax(P_wave) - len(prev_ts)
                        P_n_loc = np.argmax(P_n_wave) + P_n_start
                        Q_loc = -30 + np.argmin(prev_ts[-30:])
                        R_loc = 0
                    except ValueError:
                        continue
                        f_error = open('error_data.csv', 'w', encoding='utf-8', newline='')
                        error_writer = csv.writer(f_error)
                        error_writer.writerow(['error'])
                        error_writer.close()

                    ### features, recent add (2)
                    QRS_peak = max(ts)
                    QRS_area = np.sum(np.abs(prev_ts[Q_loc: 0])) + np.sum(np.abs(ts[0: S_loc]))

                    ### features (5)
                    PR_duration  = np.abs(0-P_loc)
                    PR_duration_corrected = PR_duration / len(ts)
                    QRS_duration = S_loc - Q_loc
                    QRS_duration_corrected = QRS_duration / len(ts)
                    QT_duration = T_loc - Q_loc
                    QT_corrected = QT_duration / len(ts)
                    if QRS_duration == 0:
                        vent_rate = 0
                    else:
                        vent_rate = 1 / QRS_duration

                    ### number of f waves (2)
                    QT_interval = prev_ts[Q_loc:] + ts[:T_loc]
                    thres = np.mean(QT_interval) + (T_peak - np.mean(QT_interval))/50
                    NF, Fwidth = LongThresCrossing(QT_interval, thres)

                    ### more features (16)
                    RQ_amp = R_peak - Q_peak
                    RS_amp = R_peak - S_peak
                    ST_amp = T_peak - S_peak
                    PQ_amp = P_peak - Q_peak
                    QS_amp = Q_peak - S_peak
                    RP_amp = R_peak - P_peak
                    RT_amp = R_peak - T_peak

                    ST_duration = T_loc - S_loc
                    ST_duration_corrected = ST_duration / len(ts)
                    RS_duration = S_loc - R_loc
                    RS_duration_corrected = RS_duration / len(ts)
                    if RS_duration == 0:
                        RS_slope = 0
                    else:
                        RS_slope = RS_amp / RS_duration
                    if ST_duration == 0:
                        ST_slope = 0
                    else:
                        ST_slope = ST_amp / ST_duration

                    ##PP TT
                    PP_amp = np.abs(P_n_peak - P_peak)
                    TT_amp = np.abs(T_p_peak - T_peak)
                    PP_duration = P_n_loc - P_loc
                    TT_duration = T_loc - T_p_loc
                    ### add to list
                    QRS_peak_list.append(QRS_peak)
                    QRS_area_list.append(QRS_area)

                    PR_duration_list.append(PR_duration)
                    PR_duration_corrected_list.append(PR_duration_corrected)
                    QRS_duration_list.append(QRS_duration)
                    QRS_duration_corrected_list.append(QRS_duration_corrected)
                    QT_duration_list.append(QT_duration)
                    QT_corrected_list.append(QT_corrected)
                    vent_rate_list.append(vent_rate)

                    NF_list.append(NF)
                    Fwidth_list.append(Fwidth)

                    RQ_amp_list.append(RQ_amp)
                    RS_amp_list.append(RS_amp)
                    ST_amp_list.append(ST_amp)
                    PQ_amp_list.append(PQ_amp)
                    QS_amp_list.append(QS_amp)
                    RP_amp_list.append(RP_amp)
                    RT_amp_list.append(RT_amp)
                    ST_duration_list.append(ST_duration)
                    ST_duration_corrected_list.append(ST_duration_corrected)
                    RS_duration_list.append(RS_duration)
                    RS_duration_corrected_list.append(RS_duration_corrected)
                    T_peak_list.append(T_peak)
                    P_peak_list.append(P_peak)
                    Q_peak_list.append(Q_peak)
                    R_peak_list.append(R_peak)
                    S_peak_list.append(S_peak)
                    RS_slope_list.append(RS_slope)
                    ST_slope_list.append(ST_slope)
                    PP_amp_list.append(PP_amp)
                    TT_amp_list.append(TT_amp)
                    PP_duration_list.append(PP_duration)
                    TT_duration_list.append(TT_duration)

                features_part = []
                features_part.extend([np.mean(QRS_peak_list),
                                    np.mean(QRS_area_list),
                                    np.mean(PR_duration_list),
                                    np.mean(PR_duration_corrected_list),
                                    np.mean(QRS_duration_list),
                                    np.mean(QRS_duration_corrected_list),
                                    np.mean(QT_duration_list),
                                    np.mean(QT_corrected_list),
                                    np.mean(vent_rate_list),
                                    np.mean(NF_list),
                                    np.mean(Fwidth_list),
                                    np.mean(RQ_amp_list),
                                    np.mean(RS_amp_list),
                                    np.mean(ST_amp_list),
                                    np.mean(PQ_amp_list),
                                    np.mean(QS_amp_list),
                                    np.mean(RP_amp_list),
                                    np.mean(RT_amp_list),
                                    np.mean(ST_duration_list),
                                    np.mean(ST_duration_corrected_list),
                                    np.mean(RS_duration_list),
                                    np.mean(RS_duration_corrected_list),
                                    np.mean(T_peak_list),
                                    np.mean(P_peak_list),
                                    np.mean(Q_peak_list),
                                    np.mean(R_peak_list),
                                    np.mean(S_peak_list),
                                    np.mean(RS_slope_list),
                                    np.mean(ST_slope_list),
                                    np.mean(PP_amp_list),
                                    np.mean(TT_amp_list),
                                    np.mean(PP_duration_list),
                                    np.mean(TT_duration_list),

                                    np.max(QRS_peak_list),
                                    np.max(QRS_area_list),
                                    np.max(PR_duration_list),
                                    np.max(PR_duration_corrected_list),
                                    np.max(QRS_duration_list),
                                    np.max(QRS_duration_corrected_list),
                                    np.max(QT_duration_list),
                                    np.max(QT_corrected_list),
                                    np.max(vent_rate_list),
                                    np.max(NF_list),
                                    np.max(Fwidth_list),
                                    np.max(RQ_amp_list),
                                    np.max(RS_amp_list),
                                    np.max(ST_amp_list),
                                    np.max(PQ_amp_list),
                                    np.max(QS_amp_list),
                                    np.max(RP_amp_list),
                                    np.max(RT_amp_list),
                                    np.max(ST_duration_list),
                                    np.max(ST_duration_corrected_list),
                                    np.max(RS_duration_list),
                                    np.max(RS_duration_corrected_list),
                                    np.max(T_peak_list),
                                    np.max(P_peak_list),
                                    np.max(Q_peak_list),
                                    np.max(R_peak_list),
                                    np.max(S_peak_list),
                                    np.max(RS_slope_list),
                                    np.max(ST_slope_list),
                                    np.max(PP_amp_list),
                                    np.max(TT_amp_list),
                                    np.max(PP_duration_list),
                                    np.max(TT_duration_list),

                                    np.min(QRS_peak_list),
                                    np.min(QRS_area_list),
                                    np.min(PR_duration_list),
                                    np.min(PR_duration_corrected_list),
                                    np.min(QRS_duration_list),
                                    np.min(QRS_duration_corrected_list),
                                    np.min(QT_duration_list),
                                    np.min(QT_corrected_list),
                                    np.min(vent_rate_list),
                                    np.min(NF_list),
                                    np.min(Fwidth_list),
                                    np.min(RQ_amp_list),
                                    np.min(RS_amp_list),
                                    np.min(ST_amp_list),
                                    np.min(PQ_amp_list),
                                    np.min(QS_amp_list),
                                    np.min(RP_amp_list),
                                    np.min(RT_amp_list),
                                    np.min(ST_duration_list),
                                    np.min(ST_duration_corrected_list),
                                    np.min(RS_duration_list),
                                    np.min(RS_duration_corrected_list),
                                    np.min(T_peak_list),
                                    np.min(P_peak_list),
                                    np.min(Q_peak_list),
                                    np.min(R_peak_list),
                                    np.min(S_peak_list),
                                    np.min(RS_slope_list),
                                    np.min(ST_slope_list),
                                    np.min(PP_amp_list),
                                    np.min(TT_amp_list),
                                    np.min(PP_duration_list),
                                    np.min(TT_duration_list),

                                    np.std(QRS_peak_list),
                                    np.std(QRS_area_list),
                                    np.std(PR_duration_list),
                                    np.std(PR_duration_corrected_list),
                                    np.std(QRS_duration_list),
                                    np.std(QRS_duration_corrected_list),
                                    np.std(QT_duration_list),
                                    np.std(QT_corrected_list),
                                    np.std(vent_rate_list),
                                    np.std(NF_list),
                                    np.std(Fwidth_list),
                                    np.std(RQ_amp_list),
                                    np.std(RS_amp_list),
                                    np.std(ST_amp_list),
                                    np.std(PQ_amp_list),
                                    np.std(QS_amp_list),
                                    np.std(RP_amp_list),
                                    np.std(RT_amp_list),
                                    np.std(ST_duration_list),
                                    np.std(ST_duration_corrected_list),
                                    np.std(RS_duration_list),
                                    np.std(RS_duration_corrected_list),
                                    np.std(T_peak_list),
                                    np.std(P_peak_list),
                                    np.std(Q_peak_list),
                                    np.std(R_peak_list),
                                    np.std(S_peak_list),
                                    np.std(RS_slope_list),
                                    np.std(ST_slope_list),
                                    np.std(PP_amp_list),
                                    np.std(TT_amp_list),
                                    np.std(PP_duration_list),
                                    np.std(TT_duration_list)])
                for pn in [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]:
                #for pn in [5,10,15,20,25,30,35,40,45,55,60,65,70,75,80,85,90,95]:
                    features_part.extend([
                        np.percentile(QRS_peak_list, pn),
                        np.percentile(QRS_area_list, pn),
                        np.percentile(PR_duration_list, pn),
                        np.percentile(PR_duration_corrected_list, pn),
                        np.percentile(QRS_duration_list, pn),
                        np.percentile(QRS_duration_corrected_list, pn),
                        np.percentile(QT_duration_list, pn),
                        np.percentile(QT_corrected_list, pn),
                        np.percentile(vent_rate_list, pn),
                        np.percentile(NF_list, pn),
                        np.percentile(Fwidth_list, pn),
                        np.percentile(RQ_amp_list, pn),
                        np.percentile(RS_amp_list, pn),
                        np.percentile(ST_amp_list, pn),
                        np.percentile(PQ_amp_list, pn),
                        np.percentile(QS_amp_list, pn),
                        np.percentile(RP_amp_list, pn),
                        np.percentile(RT_amp_list, pn),
                        np.percentile(ST_duration_list, pn),
                        np.percentile(ST_duration_corrected_list, pn),
                        np.percentile(RS_duration_list, pn),
                        np.percentile(RS_duration_corrected_list, pn),
                        np.percentile(T_peak_list, pn),
                        np.percentile(P_peak_list, pn),
                        np.percentile(Q_peak_list, pn),
                        np.percentile(R_peak_list, pn),
                        np.percentile(S_peak_list, pn),
                        np.percentile(RS_slope_list, pn),
                        np.percentile(ST_slope_list, pn),
                        np.percentile(PP_amp_list, pn),
                        np.percentile(TT_amp_list, pn),
                        np.percentile(PP_duration_list, pn),
                        np.percentile(TT_duration_list, pn)
                    ])
                features.append(features_part)
            else:
                features.append([0.0] * 726)
        except:
            features.append([0.0] * 726)
        step += 1
        if step % 1000 == 0:
            print('extracting ...', step)
            # break
            

    print('extract GetShortStatWaveFeature DONE')
    feature_list = []
    return feature_list, np.array(features)


def get_short_stat_wave_feature_from_avR(table, pid_list, long_pid_list):
    '''
    short stat feature, actually long feature

    Electrocardiogram Feature Extraction and Pattern Recognition Using a Novel Windowing Algorithm

    row of out feature is 8000+

    TODO: more on how to detect PT waves
    '''

    print('extract GetShortStatWaveFeature begin')

    features = []
    pid_short_dic = {}

    ### no-preprocess, performs better
    for i in range(len(pid_list)):
        if pid_list[i] in pid_short_dic.keys():
            pid_short_dic[pid_list[i]].append(table[i])
        else:
            pid_short_dic[pid_list[i]] = [table[i]]

    step = 0
    for pid in long_pid_list:
        try:
            if pid in pid_short_dic.keys() and len(pid_short_dic[pid]) - 2 > 0:

                ### init
                QRS_peak_list = []
                QRS_area_list = []

                PR_duration_list = []
                PR_duration_corrected_list = []
                QRS_duration_list = []
                QRS_duration_corrected_list = []
                QT_duration_list = []
                QT_corrected_list = []
                vent_rate_list = []

                NF_list = []
                Fwidth_list = []

                RQ_amp_list = []
                RS_amp_list = []
                ST_amp_list = []
                PQ_amp_list = []
                QS_amp_list = []
                RP_amp_list = []
                RT_amp_list = []
                ST_duration_list = []
                ST_duration_corrected_list = []
                RS_duration_list = []
                RS_duration_corrected_list = []
                T_peak_list = []
                P_peak_list = []
                Q_peak_list = []
                R_peak_list = []
                S_peak_list = []
                RS_slope_list = []
                ST_slope_list = []
                PP_amp_list = []
                TT_amp_list = []
                PP_duration_list = []
                TT_duration_list = []

                ### select short data of one patient
                sub_table = pid_short_dic[pid]

                for i in range(len(sub_table) - 2):
                    prev_ts = sub_table[i]
                    ts = sub_table[i + 1]
                    try:
                        ### select each short data
                        T_start = round(0.15 * len(ts))
                        T_end = round(0.55 * len(ts))
                        T_p_start = round(0.15 * len(prev_ts))
                        T_p_end = round(0.55 * len(prev_ts))
                        P_start = round(0.65 * len(prev_ts))
                        P_end = round(0.95 * len(prev_ts))
                        P_n_start = round(0.65 * len(ts))
                        P_n_end = round(0.95 * len(ts))
                        T_wave = ts[T_start:T_end]
                        T_p_wave = prev_ts[T_p_start:T_p_end]
                        P_wave = prev_ts[P_start:P_end]
                        P_n_wave = ts[P_n_start:P_n_end]
                        T_peak = min(T_wave)
                        T_p_peak = min(T_p_wave)
                        P_n_peak = min(P_n_wave)
                        P_peak = min(P_wave)
                        Q_peak = max(prev_ts[-30:])
                        R_peak = ts[0]
                        S_peak = max(ts[:30])
                        T_loc = np.argmin(T_wave) + T_start
                        T_p_loc = np.argmin(T_p_wave) + T_p_start - len(prev_ts)
                        P_loc = P_start + np.argmin(P_wave) - len(prev_ts)
                        P_n_loc = np.argmin(P_n_wave) + P_n_start
                        Q_loc = -30 + np.argmax(prev_ts[-30:])
                        R_loc = 0
                        S_loc = np.argmax(ts[:30])
                    except ValueError:
                        continue
                        f_error = open('error_data.csv', 'w', encoding='utf-8', newline='')
                        error_writer = csv.writer(f_error)
                        error_writer.writerow(['error'])
                        error_writer.close()
                    ### features, recent add (2)
                    QRS_peak = min(ts)
                    QRS_area = np.sum(np.abs(prev_ts[Q_loc: 0])) + np.sum(np.abs(ts[0: S_loc]))

                    ### features (5)
                    PR_duration = np.abs(0 - P_loc)
                    PR_duration_corrected = PR_duration / len(ts)
                    QRS_duration = S_loc - Q_loc
                    QRS_duration_corrected = QRS_duration / len(ts)
                    QT_duration = T_loc - Q_loc
                    QT_corrected = QT_duration / len(ts)
                    if QRS_duration == 0:
                        vent_rate = 0
                    else:
                        vent_rate = 1 / QRS_duration

                    ### number of f waves (2)
                    QT_interval = prev_ts[Q_loc:] + ts[:T_loc]
                    thres = np.mean(QT_interval) + (T_peak - np.mean(QT_interval)) / 50
                    NF, Fwidth = LongThresCrossing(QT_interval, thres)

                    ### more features (16)
                    RQ_amp = R_peak - Q_peak
                    RS_amp = R_peak - S_peak
                    ST_amp = T_peak - S_peak
                    PQ_amp = P_peak - Q_peak
                    QS_amp = Q_peak - S_peak
                    RP_amp = R_peak - P_peak
                    RT_amp = R_peak - T_peak

                    ST_duration = T_loc - S_loc
                    ST_duration_corrected = ST_duration / len(ts)
                    RS_duration = S_loc - R_loc
                    RS_duration_corrected = RS_duration / len(ts)
                    if RS_duration == 0:
                        RS_slope = 0
                    else:
                        RS_slope = RS_amp / RS_duration
                    if ST_duration == 0:
                        ST_slope = 0
                    else:
                        ST_slope = ST_amp / ST_duration

                    ##PP TT
                    PP_amp = np.abs(P_n_peak - P_peak)
                    TT_amp = np.abs(T_p_peak - T_peak)
                    PP_duration = P_n_loc - P_loc
                    TT_duration = T_loc - T_p_loc
                    ### add to list
                    QRS_peak_list.append(QRS_peak)
                    QRS_area_list.append(QRS_area)

                    PR_duration_list.append(PR_duration)
                    PR_duration_corrected_list.append(PR_duration_corrected)
                    QRS_duration_list.append(QRS_duration)
                    QRS_duration_corrected_list.append(QRS_duration_corrected)
                    QT_duration_list.append(QT_duration)
                    QT_corrected_list.append(QT_corrected)
                    vent_rate_list.append(vent_rate)

                    NF_list.append(NF)
                    Fwidth_list.append(Fwidth)

                    RQ_amp_list.append(RQ_amp)
                    RS_amp_list.append(RS_amp)
                    ST_amp_list.append(ST_amp)
                    PQ_amp_list.append(PQ_amp)
                    QS_amp_list.append(QS_amp)
                    RP_amp_list.append(RP_amp)
                    RT_amp_list.append(RT_amp)
                    ST_duration_list.append(ST_duration)
                    ST_duration_corrected_list.append(ST_duration_corrected)
                    RS_duration_list.append(RS_duration)
                    RS_duration_corrected_list.append(RS_duration_corrected)
                    T_peak_list.append(T_peak)
                    P_peak_list.append(P_peak)
                    Q_peak_list.append(Q_peak)
                    R_peak_list.append(R_peak)
                    S_peak_list.append(S_peak)
                    RS_slope_list.append(RS_slope)
                    ST_slope_list.append(ST_slope)
                    PP_amp_list.append(PP_amp)
                    TT_amp_list.append(TT_amp)
                    PP_duration_list.append(PP_duration)
                    TT_duration_list.append(TT_duration)

                features_part = []
                features_part.extend([np.mean(QRS_peak_list),
                                      np.mean(QRS_area_list),
                                      np.mean(PR_duration_list),
                                      np.mean(PR_duration_corrected_list),
                                      np.mean(QRS_duration_list),
                                      np.mean(QRS_duration_corrected_list),
                                      np.mean(QT_duration_list),
                                      np.mean(QT_corrected_list),
                                      np.mean(vent_rate_list),
                                      np.mean(NF_list),
                                      np.mean(Fwidth_list),
                                      np.mean(RQ_amp_list),
                                      np.mean(RS_amp_list),
                                      np.mean(ST_amp_list),
                                      np.mean(PQ_amp_list),
                                      np.mean(QS_amp_list),
                                      np.mean(RP_amp_list),
                                      np.mean(RT_amp_list),
                                      np.mean(ST_duration_list),
                                      np.mean(ST_duration_corrected_list),
                                      np.mean(RS_duration_list),
                                      np.mean(RS_duration_corrected_list),
                                      np.mean(T_peak_list),
                                      np.mean(P_peak_list),
                                      np.mean(Q_peak_list),
                                      np.mean(R_peak_list),
                                      np.mean(S_peak_list),
                                      np.mean(RS_slope_list),
                                      np.mean(ST_slope_list),
                                      np.mean(PP_amp_list),
                                      np.mean(TT_amp_list),
                                      np.mean(PP_duration_list),
                                      np.mean(TT_duration_list),

                                      np.max(QRS_peak_list),
                                      np.max(QRS_area_list),
                                      np.max(PR_duration_list),
                                      np.max(PR_duration_corrected_list),
                                      np.max(QRS_duration_list),
                                      np.max(QRS_duration_corrected_list),
                                      np.max(QT_duration_list),
                                      np.max(QT_corrected_list),
                                      np.max(vent_rate_list),
                                      np.max(NF_list),
                                      np.max(Fwidth_list),
                                      np.max(RQ_amp_list),
                                      np.max(RS_amp_list),
                                      np.max(ST_amp_list),
                                      np.max(PQ_amp_list),
                                      np.max(QS_amp_list),
                                      np.max(RP_amp_list),
                                      np.max(RT_amp_list),
                                      np.max(ST_duration_list),
                                      np.max(ST_duration_corrected_list),
                                      np.max(RS_duration_list),
                                      np.max(RS_duration_corrected_list),
                                      np.max(T_peak_list),
                                      np.max(P_peak_list),
                                      np.max(Q_peak_list),
                                      np.max(R_peak_list),
                                      np.max(S_peak_list),
                                      np.max(RS_slope_list),
                                      np.max(ST_slope_list),
                                      np.max(PP_amp_list),
                                      np.max(TT_amp_list),
                                      np.max(PP_duration_list),
                                      np.max(TT_duration_list),

                                      np.min(QRS_peak_list),
                                      np.min(QRS_area_list),
                                      np.min(PR_duration_list),
                                      np.min(PR_duration_corrected_list),
                                      np.min(QRS_duration_list),
                                      np.min(QRS_duration_corrected_list),
                                      np.min(QT_duration_list),
                                      np.min(QT_corrected_list),
                                      np.min(vent_rate_list),
                                      np.min(NF_list),
                                      np.min(Fwidth_list),
                                      np.min(RQ_amp_list),
                                      np.min(RS_amp_list),
                                      np.min(ST_amp_list),
                                      np.min(PQ_amp_list),
                                      np.min(QS_amp_list),
                                      np.min(RP_amp_list),
                                      np.min(RT_amp_list),
                                      np.min(ST_duration_list),
                                      np.min(ST_duration_corrected_list),
                                      np.min(RS_duration_list),
                                      np.min(RS_duration_corrected_list),
                                      np.min(T_peak_list),
                                      np.min(P_peak_list),
                                      np.min(Q_peak_list),
                                      np.min(R_peak_list),
                                      np.min(S_peak_list),
                                      np.min(RS_slope_list),
                                      np.min(ST_slope_list),
                                      np.min(PP_amp_list),
                                      np.min(TT_amp_list),
                                      np.min(PP_duration_list),
                                      np.min(TT_duration_list),

                                      np.std(QRS_peak_list),
                                      np.std(QRS_area_list),
                                      np.std(PR_duration_list),
                                      np.std(PR_duration_corrected_list),
                                      np.std(QRS_duration_list),
                                      np.std(QRS_duration_corrected_list),
                                      np.std(QT_duration_list),
                                      np.std(QT_corrected_list),
                                      np.std(vent_rate_list),
                                      np.std(NF_list),
                                      np.std(Fwidth_list),
                                      np.std(RQ_amp_list),
                                      np.std(RS_amp_list),
                                      np.std(ST_amp_list),
                                      np.std(PQ_amp_list),
                                      np.std(QS_amp_list),
                                      np.std(RP_amp_list),
                                      np.std(RT_amp_list),
                                      np.std(ST_duration_list),
                                      np.std(ST_duration_corrected_list),
                                      np.std(RS_duration_list),
                                      np.std(RS_duration_corrected_list),
                                      np.std(T_peak_list),
                                      np.std(P_peak_list),
                                      np.std(Q_peak_list),
                                      np.std(R_peak_list),
                                      np.std(S_peak_list),
                                      np.std(RS_slope_list),
                                      np.std(ST_slope_list),
                                      np.std(PP_amp_list),
                                      np.std(TT_amp_list),
                                      np.std(PP_duration_list),
                                      np.std(TT_duration_list)])
                for pn in [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]:
                #for pn in [5, 10, 15, 20, 25, 30, 35, 40, 45, 55, 60, 65, 70, 75, 80, 85, 90, 95]:
                    features_part.extend([
                        np.percentile(QRS_peak_list, pn),
                        np.percentile(QRS_area_list, pn),
                        np.percentile(PR_duration_list, pn),
                        np.percentile(PR_duration_corrected_list, pn),
                        np.percentile(QRS_duration_list, pn),
                        np.percentile(QRS_duration_corrected_list, pn),
                        np.percentile(QT_duration_list, pn),
                        np.percentile(QT_corrected_list, pn),
                        np.percentile(vent_rate_list, pn),
                        np.percentile(NF_list, pn),
                        np.percentile(Fwidth_list, pn),
                        np.percentile(RQ_amp_list, pn),
                        np.percentile(RS_amp_list, pn),
                        np.percentile(ST_amp_list, pn),
                        np.percentile(PQ_amp_list, pn),
                        np.percentile(QS_amp_list, pn),
                        np.percentile(RP_amp_list, pn),
                        np.percentile(RT_amp_list, pn),
                        np.percentile(ST_duration_list, pn),
                        np.percentile(ST_duration_corrected_list, pn),
                        np.percentile(RS_duration_list, pn),
                        np.percentile(RS_duration_corrected_list, pn),
                        np.percentile(T_peak_list, pn),
                        np.percentile(P_peak_list, pn),
                        np.percentile(Q_peak_list, pn),
                        np.percentile(R_peak_list, pn),
                        np.percentile(S_peak_list, pn),
                        np.percentile(RS_slope_list, pn),
                        np.percentile(ST_slope_list, pn),
                        np.percentile(PP_amp_list, pn),
                        np.percentile(TT_amp_list, pn),
                        np.percentile(PP_duration_list, pn),
                        np.percentile(TT_duration_list, pn)
                    ])
                features.append(features_part)
            else:
                features.append([0.0] * 726)
        except:
            features.append([0.0] * 726)
        step += 1
        if step % 1000 == 0:
            print('extracting ...', step)
            # break

    print('extract GetShortStatWaveFeature DONE')
    feature_list = []
    return feature_list, np.array(features)

def get_short_feature(table):
    '''
    rows of table is 330000+
    
    no use now
    '''

    global feature_list
    feature_list = []


    features = []
    step = 0
    for ts in table:
        row = []

        row.extend(short_basic_stat(ts))
#        row.extend(short_zero_crossing(ts))
        
        features.append(row)
        
        step += 1
        if step % 100000 == 0:
            print('extracting ...')
#            break
        
    print('extract DONE')
    
    return feature_list, features



if __name__ == '__main__':
    short_pid, short_data, short_label = ReadData.ReadData( '../../data1/short.csv' )
    QRS_pid, QRS_data, QRS_label = ReadData.ReadData( '../../data1/QRSinfo.csv' )
    tmp_features = get_short_stat_wave_feature(short_data[:10], short_pid[:10], QRS_pid[0])
    print(len(tmp_features[1][0]))
    
    
