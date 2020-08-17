import numpy as np
import os
from scipy.io import loadmat, savemat
from process.util import load_challenge_data, load_labels, load_label_files, load_weights

if __name__ == '__main__':

    # Define the weights, the SNOMED CT code for the normal class, and equivalent SNOMED CT codes.
    weights_file = '../evaluation/weights.csv'
    normal_class = '426783006'
    equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]

    input_directory_data = '/DATASET/challenge2020/new_data/All_data_new'
    input_directory_label = '/DATASET/challenge2020/new_data/All_data_new'

    # Find the label files.
    print('Finding label and output files...')
    label_files = load_label_files(input_directory_label)

    # Load the labels and classes.
    print('Loading labels and outputs...')
    classes, labels_onehot, labels = load_labels(label_files, normal_class, equivalent_classes)

    # Load the weights for the Challenge metric.
    print('Loading weights...')
    weights = load_weights(weights_file, classes)

    # Classes that are scored with the Challenge metric.
    indices = np.any(weights, axis=0)  # Find indices of classes in weight matrix.

    num_files = len(label_files)
    print("num_files:", num_files)

    ###################################

    name = ['心动过缓', '窦性心动过速', '窦性心动过缓', '窦性心律不齐']
    dx = ['bradycardia', 'sinus tachycardia', 'sinus bradycardia', 'sinus arrhythmia']
    abb = ['Brady', 'STach', 'SB', 'SA']
    code = ['426627000', '427084000', '426177001', '427393009']
    code2 = ['426627000', '427084000', '426177001', '427393009']

    index = list()
    for c in code2:
        index.append(classes.index(c))
    index = np.array(index)

    # index = np.zeros((len(classes)))
    # for c in code2:
    #     index[classes.index(c)] = 1

    savemat('./index.mat', {'val': index})


