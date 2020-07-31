import numpy as np
import os
from scipy.io import loadmat, savemat
from process.util import load_challenge_data, load_labels, load_label_files
import time

def slide_and_cut(X, file_name, preset_n_segment, save_path):
    n_sample = len(X)
    window_size = 3000
    short_signals = []
    short_signals_file = []
    for i in range(n_sample):
        n_segment = preset_n_segment
        length = X[i].shape[1]
        if length < window_size:
            short_signals.append(i)
            short_signals_file.append(file_name[i])
            continue
        offset = (length - window_size * n_segment) / (n_segment + 1)
        if offset >= 0:
            start = 0 + offset
        else:
            offset = (length - window_size * n_segment ) / (n_segment-1)
            start = 0
        segments = []
        for j in range(n_segment):
            ind = int(start + j * (window_size + offset))
            segment = X[i][:, ind:ind+window_size]
            segments.append(segment)
        segments = np.array(segments)
        savemat(os.path.join(save_path, file_name[i]+'.mat'), {'val': segments})
        print('{}/{}... Signal len: {}  Sliding into {} segments'.format(i + 1, n_sample, length, n_segment))
    savemat(os.path.join(save_path, 'short_signals.mat'), {'val': short_signals})
    savemat(os.path.join(save_path, 'short_signals_file.mat'), {'val': short_signals_file})

if __name__ == '__main__':
    # Define the weights, the SNOMED CT code for the normal class, and equivalent SNOMED CT codes.
    weights_file = 'weights.csv'
    normal_class = '426783006'
    equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]

    input_directory_data = '/DATASET/challenge2020/All_data_resampled_to_300HZ'
    input_directory_label = '/DATASET/challenge2020/All_data'
    save_path = '/DATASET/challenge2020/All_data_resampled_to_300HZ_and_slided_n_segment=1'

    # Find the label files.
    print('Finding label and output files...')
    label_files = load_label_files(input_directory_label)

    # Load the labels and classes.
    print('Loading labels and outputs...')
    label_classes, labels_onehot, labels = load_labels(label_files, normal_class, equivalent_classes)

    num_files = len(label_files)
    print("num_files:", num_files)

    # class_count = loadmat('/home/weiyuhua/Data/challenge2020_info/class_count.mat')['val']

    dataset = []
    file_name = []

    # Load data
    for i, (f, label) in enumerate(zip(label_files, labels)):
        print('    {}/{}...'.format(i + 1, num_files))
        file = os.path.basename(f)
        name, ext = os.path.splitext(file)
        data, header_data = load_challenge_data(file, input_directory_label, input_directory_data)
        c = []
        if np.isnan(data).any():
            print(i)
            c.append(i)
        dataset.append(data)
        file_name.append(name)

    slide_and_cut(dataset, file_name, preset_n_segment=1, save_path=save_path)
    print("Done")


