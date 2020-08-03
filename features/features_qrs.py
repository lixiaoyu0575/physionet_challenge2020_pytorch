# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 21:38:14 2017

@author: hsd
"""

import numpy as np
from scipy import stats
import tsfresh as tsf
from minNCCE import minimumNCCE

##################################################
### tools
##################################################
def ThreePointsMedianPreprocess(ts):
    '''
    8-beat sliding window RR interval irregularity detector [21]
    '''
    new_ts = []
    for i in range(len(ts)-2):
        new_ts.append(np.median([ts[i], ts[i+1], ts[i+2]]))
    return new_ts

def autocorr(ts, t):
    return np.corrcoef(np.array([ts[0:len(ts)-t], ts[t:len(ts)]]))[0,1]

##################################################
### get features
##################################################


def bin_stat(ts):
    '''
    stat of bin Counter RR ts
    
    count, ratio
    '''
    global feature_list
    feature_list.extend(['bin_stat_'+str(i) for i in range(56)])

    try:
        bins_1 = [648,594,537,495,471,452,435,421,407,395,382,370,357,344,331,317,303,287,267,241,207,162,0]
        cnt_1 = [0.0] * len(bins_1)
        for i in ts:
            for j in range(len(bins_1)):
                if i > bins_1[j]:
                    cnt_1[j] += 1
                    break
        ratio_1 = [i/len(ts) for i in cnt_1]    

        bins_2 = [594,435,370,303,177]
        cnt_2 = [0.0] * len(bins_2)
        for i in ts:
            for j in range(len(bins_2)):
                if i > bins_2[j]:
                    cnt_2[j] += 1
                    break
        ratio_2 = [i/len(ts) for i in cnt_2]
        
        return cnt_1 + ratio_1 + cnt_2 + ratio_2
    except:
        return [0.0] * 56
    



    
def SampleEn(ts):
    '''    
    sample entropy on QRS interval
    '''
    global feature_list
    feature_list.extend(['SampleEntropy'])

    try:
        sample_entropy = tsf.feature_extraction.feature_calculators.sample_entropy(ts)
        return [sample_entropy]

    except :
        return [0]
    
    
def CDF(ts):
    '''
    analysis of cumulative distribution functions [17],
    '''
    global feature_list
    feature_list.extend(['CDF_cdf_density'])
    try:
        n_bins = 25
        hist, _ = np.histogram(ts, range=(162, 600), bins=n_bins)
        cdf = np.cumsum(hist)/len(ts)
        cdf_density = np.sum(cdf) / n_bins
        return [cdf_density]
    except:
        return [0]
    
def CoeffOfVariation(ts):
    '''
    analysis of Coeff Of Variation,
    '''
    global feature_list
    feature_list.extend(['CoeffOfVariation_coeff_ts', 'CoeffOfVariation_coeff_dts'])
    try:
        if len(ts) >= 3:
            tmp_ts = ts[1:-1]
            if np.mean(tmp_ts) == 0:
                coeff_ts = 0.0
            else:
                coeff_ts = np.std(tmp_ts) / np.mean(tmp_ts)
        else:
            coeff_ts = 0.0

        if len(ts) >= 4:
            tmp_ts = ts[1:-1]
            tmp_ts = np.diff(tmp_ts)
            if np.mean(tmp_ts) == 0:
                coeff_dts = 0.0
            else:
                coeff_dts = np.std(tmp_ts) / np.mean(tmp_ts)
        else:
            coeff_dts = 0.0

        return [coeff_ts, coeff_dts]
    except:
        return [0,0]

def MAD(ts):
    '''
    thresholding on the median absolute deviation (MAD) of RR intervals [18] 
    '''
    global feature_list
    feature_list.extend(['MAD_mad'])
    try:
        ts_median = np.median(ts)
        mad = np.median([np.abs(ii - ts_median) for ii in ts])
        return [mad]
    except:
        return [0]


def QRSBasicStat(ts):
    
    global feature_list
    feature_list.extend([   'QRSBasicStat_Mean',
                            'QRSBasicStat_HR',
                            'QRSBasicStat_Count',
                            'QRSBasicStat_Range',
                            'QRSBasicStat_Var',
                            'QRSBasicStat_Skew',
                            'QRSBasicStat_Kurtosis',
                            'QRSBasicStat_Median',
                            'QRSBasicStat_Min',
                            'QRSBasicStat_p_5',
                            'QRSBasicStat_p_10',
                            'QRSBasicStat_p_15',
                            'QRSBasicStat_p_20',
                            'QRSBasicStat_p_25',
                            'QRSBasicStat_p_30',
                            'QRSBasicStat_p_35',
                            'QRSBasicStat_p_40',
                            'QRSBasicStat_p_45',
                            'QRSBasicStat_p_55',
                            'QRSBasicStat_p_60',
                            'QRSBasicStat_p_65',
                            'QRSBasicStat_p_70',
                            'QRSBasicStat_p_75',
                            'QRSBasicStat_p_80',
                            'QRSBasicStat_p_85',
                            'QRSBasicStat_p_90',
                            'QRSBasicStat_p_95',
                            'QRSBasicStat_range_95_5',
                            'QRSBasicStat_range_75_25',
                            'QRSBasicStatCid_ce',
                            'QRSBasicStatcount_above_mean',
                            'QRSBasicStatcount_below_mean',
                            'QRSBasicStatfirst_location_of_minimum',
                            'QRSBasicStatfirst_location_of_maximum',
                            'QRSBasicStatlast_location_of_minimum',
                            'QRSBasicStatlast_location_of_maximum',
                            'QRSBasicStatmean_second_derivative_central'])
    
    try:

        Mean = np.mean(ts)
        if Mean == 0:
            HR = 0
        else:
            HR = 1 / Mean
        Count = len(ts)
        Range = max(ts) - min(ts)
        Var = np.var(ts)
        Skew = stats.skew(ts)
        Kurtosis = stats.kurtosis(ts)
        Median = np.median(ts)
        Min = min(ts)
        p_5 = np.percentile(ts, 5)
        p_10 = np.percentile(ts, 10)
        p_15 = np.percentile(ts, 15)
        p_20 = np.percentile(ts, 20)
        p_25 = np.percentile(ts, 25)
        p_30 = np.percentile(ts, 30)
        p_35 = np.percentile(ts, 35)
        p_40 = np.percentile(ts, 40)
        p_45 = np.percentile(ts, 45)
        p_55 = np.percentile(ts, 55)
        p_60 = np.percentile(ts, 60)
        p_65 = np.percentile(ts, 65)
        p_70 = np.percentile(ts, 70)
        p_75 = np.percentile(ts, 75)
        p_80 = np.percentile(ts, 80)
        p_85 = np.percentile(ts, 85)
        p_90 = np.percentile(ts, 90)
        p_95 = np.percentile(ts, 95)
        range_95_5 = p_95 - p_5
        range_75_25 = p_75 - p_25
        cid_ce = tsf.feature_extraction.feature_calculators.cid_ce(ts, True)
        count_above_mean = tsf.feature_extraction.feature_calculators.count_above_mean(ts)/len(ts)
        count_below_mean = tsf.feature_extraction.feature_calculators.count_below_mean(ts)/len(ts)
        first_location_of_minimum = tsf.feature_extraction.feature_calculators.first_location_of_minimum(ts)
        first_location_of_maximum = tsf.feature_extraction.feature_calculators.first_location_of_maximum(ts)
        last_location_of_minimum = tsf.feature_extraction.feature_calculators.last_location_of_minimum(ts)
        last_location_of_maximum = tsf.feature_extraction.feature_calculators.last_location_of_maximum(ts)
        mean_second_derivative_central = tsf.feature_extraction.feature_calculators.mean_second_derivative_central(ts)
        return [Mean, HR, Count, Range, Var, Skew, Kurtosis, Median, Min, 
                p_5, p_10, p_15, p_20, p_25, p_30, p_35, p_40, p_45, p_55,
                p_60, p_65, p_70, p_75, p_80, p_85, p_90, p_95,
                range_95_5, range_75_25,cid_ce,count_above_mean, count_below_mean,
                first_location_of_minimum,first_location_of_maximum,last_location_of_minimum,
                last_location_of_maximum,mean_second_derivative_central]
    
    except ValueError:
        return [0.0] * 37
    
def QRSBasicStatPointMedian(ts):
    ##just change ts
    ts = ThreePointsMedianPreprocess(ts)
    return QRSBasicStat(ts)

      
def QRSBasicStatDeltaRR(ts):

    # global feature_list
    # feature_list.extend(['QRSBasicStatDeltaRR_Mean',
    #                     'QRSBasicStatDeltaRR_HR',
    #                     'QRSBasicStatDeltaRR_Count',
    #                     'QRSBasicStatDeltaRR_Range',
    #                     'QRSBasicStatDeltaRR_Var',
    #                     'QRSBasicStatDeltaRR_Skew',
    #                     'QRSBasicStatDeltaRR_Kurtosis',
    #                     'QRSBasicStatDeltaRR_Median',
    #                     'QRSBasicStatDeltaRR_Min',
    #                      'QRSBasicStatDeltaRR_p_5',
    #                      'QRSBasicStatDeltaRR_p_15',
    #                      'QRSBasicStatDeltaRR_p_25',
    #                      'QRSBasicStatDeltaRR_p_35',
    #                      'QRSBasicStatDeltaRR_p_45',
    #                      'QRSBasicStatDeltaRR_p_55',
    #                      'QRSBasicStatDeltaRR_p_65',
    #                     'QRSBasicStatDeltaRR_p_75'
    #                      'QRSBasicStatDeltaRR_p_85',
    #                      'QRSBasicStatDeltaRR_p_95'])

    ts = np.diff(ts)
    return QRSBasicStat(ts)



def QRSYuxi(ts):
    '''
    pars: 
        tol = 0.05
            define if two QRS intervals are matched
    '''
    tol = 0.05
    global feature_list
    feature_list.extend(['QRSYuxi'])

    
    try:
        avg_RR = np.median(ts)
        matched = [False] * len(ts)
        
        for i in range(len(ts)):
            seg_1 = ts[i]
            if abs(seg_1 - avg_RR) / avg_RR <= tol:
                matched[i] = True
            elif abs(seg_1 - 2 * avg_RR) / (2 * avg_RR) <= tol:
                matched[i] = True
                
        for i in range(len(ts)):
            if matched[i] is False:
                if i == 0:
                    seg_2_forward = ts[i]
                else:
                    seg_2_forward = ts[i-1] + ts[i]
                if i == len(ts)-1:
                    seg_2_backward = ts[i]
                else:
                    seg_2_backward = ts[i] + ts[i+1]
                    
                if abs(seg_2_forward - 2 * avg_RR) / (2 * avg_RR) <= tol:
                    matched[i] = True
                elif abs(seg_2_forward - 3 * avg_RR) / (3 * avg_RR) <= tol:
                    matched[i] = True
                elif abs(seg_2_backward - 2 * avg_RR) / (2 * avg_RR) <= tol:
                    matched[i] = True
                elif abs(seg_2_backward - 3 * avg_RR) / (3 * avg_RR) <= tol:
                    matched[i] = True
    
        return [sum(matched) / len(matched)]

    except:
        return [0.0] * 1
    

def Variability(ts):
    '''
    Variability(Time Domain) & Poincare plot
    https://zh.wikipedia.org/wiki/%E5%BF%83%E7%8E%87%E8%AE%8A%E7%95%B0%E5%88%86%E6%9E%90
    compute SDNN, NN50 count, pNN50
    [14] Atrial fibrillation detection by heart rate variability in Poincare plot
    Stepping: the mean stepping increment of the inter-beat intervals
    Dispersion: how spread the points in Poincaré plot are distributed around the diagonal line
    '''
    global feature_list
    feature_list.extend(['Variability_SDNN', 
                        'Variability_NN50', 
                        'Variability_pNN50', 
                        'Variability_Stepping', 
                        'Variability_Dispersion'])

    try:
        ts = ts[1:-1]
        SDNN = np.std(ts)
        freq = 500
        timelen = freq * 0.05
        if len(ts) < 3:
            NN50 = 0
            pNN50 = 0
            Stepping = 0
            Dispersion = 0
        else:
            NN = [abs(ts[x + 1] - ts[x]) for x in range(len(ts) - 1)]
            NN50 = sum([x > timelen for x in NN])
            pNN50 = float(NN50) / len(ts)
            Stepping = (sum([(NN[x] ** 2 + NN[x + 1] ** 2) ** 0.5 for x in range(len(NN) - 1)]) / (len(NN) - 1)) / (sum(ts) / len(ts))
            Dispersion = (sum([x ** 2 for x in NN]) / (2 * len(NN)) - sum(NN) ** 2 / (2 * (len(NN)) ** 2)) ** 0.5 / ((-ts[0] - 2 * ts[-1] + 2 * sum(ts)) / (2 * len(NN)))
        return [SDNN, NN50, pNN50, Stepping, Dispersion]

    except:
        return [0.0] * 5


def minimum_ncce(ts):
    global feature_list
    try:
        feature_list.extend(['minNCCE', 'minCCEI'])

        return minimumNCCE(ts)
    except:
        return [0,0]
def qrs_autocorr(ts):
    feat = []
    num_lag = 3
    
    global feature_list
    feature_list.extend(['qrs_autocorr_'+str(i) for i in range(num_lag)])
    try:
        if len(ts) >= num_lag:
            for i in range(num_lag):
                feat.append(autocorr(ts, i))
        else:
            for i in range(len(ts)):
                feat.append(autocorr(ts, i))
            feat.extend([0] * (num_lag - len(ts)))

        return feat
    except:
        return [0,0,0]

##################################################
###  get all features
##################################################
def get_qrs_feature(table):
    '''
    rows of table is 8000+
    '''
    
    global feature_list
    feature_list = []

    
    print('extract QRS begin')
    features = []
    step = 0
    for ts in table:
        row = []

        ### yes
        row.extend(CDF(ts))
        row.extend(CoeffOfVariation(ts))
        row.extend(MAD(ts))
        row.extend(QRSBasicStat(ts))
        row.extend(QRSBasicStatPointMedian(ts))
        row.extend(QRSBasicStatDeltaRR(ts))
        row.extend(QRSYuxi(ts))
        row.extend(Variability(ts))
        row.extend(minimum_ncce(ts))
        row.extend(bin_stat(ts))
        row.extend(qrs_autocorr(ts))
        
    

        ### no

        
        features.append(row)
        
        step += 1
        if step % 1000 == 0:
            print('extracting ...', step)
            # break
        
    print('extract QRS DONE')
    
    return feature_list, np.array(features)


if __name__ == '__main__':
    QRS_pid, QRS_data, QRS_label = ReadData.ReadData( '../../data1/QRSinfo.csv' )
    tmp_features = get_qrs_feature(QRS_data[:1])
    print(len(tmp_features[1][0]))
    
    

