import numpy as np
import biosppy

def get_PP_interval(signal_data , rpeaks_indices):
    PP_list = []
    for idx in range(len(rpeaks_indices)-2):
        prev_ts = signal_data[rpeaks_indices[idx]:rpeaks_indices[idx+1]].tolist()
        ts = signal_data[rpeaks_indices[idx+1]:rpeaks_indices[idx+2]].tolist()
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
        P_peak = max(P_wave)
        R_peak = ts[0]
        P_loc = P_start + np.argmax(P_wave) - len(prev_ts)
        P_n_loc = np.argmax(P_n_wave) + P_n_start
        PP_list.append(P_n_loc - P_loc)
    return PP_list


####心动过缓 [index]:63
def is_Brady(data,FS=300):
    #input : data: shape(i,12,300)
    #        fs : fs is the frequency of data
    
    #output： True: It's bradycardia(心动过缓)
    #         False； It's other label
    print("now is Brady")
    print(data.shape)
    result= []
    for i in range(len(data)):
        rpeaks_indices_1 = biosppy.signals.ecg.hamilton_segmenter(signal=data[i][1],sampling_rate=FS)
        rpeaks_indices_1_c = biosppy.signals.ecg.correct_rpeaks(signal=data[i][1], rpeaks=rpeaks_indices_1[0],sampling_rate = FS, tol=0.07)
        brady_heart_rate = ([(60/(i/FS)) for i in np.diff(rpeaks_indices_1_c[0]).tolist()])
        if np.mean(brady_heart_rate)<61 and np.min(brady_heart_rate) <60:
            result.append(1)
        else:
            result.append(0)
    return np.array(result)

####窦性心动过速 [index]:70
def is_STach(data,FS=300):
    print("now is STach")
    print(data.shape)
    result= []
    for i in range(len(data)):
        rpeaks_indices_1 = biosppy.signals.ecg.hamilton_segmenter(signal=data[i][1], sampling_rate=FS)
        rpeaks_indices_1_c = biosppy.signals.ecg.correct_rpeaks(signal=data[i][1], rpeaks=rpeaks_indices_1[0],sampling_rate=FS, tol=0.07)
        sinus_tachycardia_heart_rate = [(60/(i/FS)) for i in np.diff(rpeaks_indices_1_c[0]).tolist()]
        if np.mean(sinus_tachycardia_heart_rate)>99 and np.max(sinus_tachycardia_heart_rate)>100:
            result.append(1)
        else:
            result.append(0)
    return np.array(result)

####窦性心动过缓 [index]:61
def is_SB(data,FS=300):
    print("now is SB")
    print(data.shape)
    result= []
    for i in range(len(data)):
        rpeaks_indices_1 = biosppy.signals.ecg.hamilton_segmenter(signal=data[i][1], sampling_rate=FS)
        rpeaks_indices_1_c = biosppy.signals.ecg.correct_rpeaks(signal=data[i][1], rpeaks=rpeaks_indices_1[0],sampling_rate=FS, tol=0.07)
        sinus_bradycardia_heart_rate = [(60/(i/FS)) for i in np.diff(rpeaks_indices_1_c[0]).tolist()]
        if np.mean(sinus_bradycardia_heart_rate)<61 and np.min(sinus_bradycardia_heart_rate) <60:
            result.append(1)
        else:
            result.append(0)
    return np.array(result)

####窦性心律不齐  [index]:72
def is_SA(data,FS=300):
    print("now is SA")
    print(data.shape)
    result= []
    for i in range(len(data)):
        rpeaks_indices_1 = biosppy.signals.ecg.hamilton_segmenter(signal=data[i][1], sampling_rate=FS)
        rpeaks_indices_1_c = biosppy.signals.ecg.correct_rpeaks(signal=data[i][1], rpeaks=rpeaks_indices_1[0],sampling_rate=FS, tol=0.07)
        RR_list = np.diff(rpeaks_indices_1_c[0]).tolist()
        sinus_arrhythmia_RR =(np.max(RR_list) - np.min(RR_list))/FS
        if sinus_arrhythmia_RR >0.12:
            result.append(1)
        else:
            result.append(0)
    return np.array(result)