from features_qrs import get_qrs_feature
from preprocess_do_not_save_version import get_preprocessed_data
import csv
if __name__ == '__main__':
    with open('/home/hanhaochen/physionet_challenge2020_pytorch/features/Preprocess_Data/REFERENCE.csv', 'r') as f_r:
        reader = csv.reader(f_r)
        result = list(reader)

    for i in range(1):
        print('#######################'+result[i][0]+'######################')

        ecg_data = result[i:i+1]
        patient_info,short_data0,short_data1,short_data2,short_data3,short_data4,short_data5,short_data6,short_data7,short_data8,short_data9,short_data10,short_data11,long_data0,long_data1,long_data2,long_data3,long_data4,long_data5,long_data6,long_data7,long_data8,long_data9,long_data10,long_data11,qrs_info, long_pid, short_pid0,short_pid1,short_pid2,short_pid3,short_pid4,short_pid5,short_pid6,short_pid7,short_pid8,short_pid9,short_pid10,short_pid11 = get_preprocessed_data(ecg_data)
        #each_feature = GetAllFeature_test(patient_info,short_data0,short_data1,short_data2,short_data3,short_data4,short_data5,short_data6,short_data7,short_data8,short_data9,short_data10,short_data11,long_data0,long_data1,long_data2,long_data3,long_data4,long_data5,long_data6,long_data7,long_data8,long_data9,long_data10,long_data11,qrs_info, long_pid, short_pid0,short_pid1,short_pid2,short_pid3,short_pid4,short_pid5,short_pid6,short_pid7,short_pid8,short_pid9,short_pid10,short_pid11)
        feature_name,feature_data = get_qrs_feature(qrs_info)
        print('name',len(feature_name))
        print('data', len(feature_data[0]))