import os
import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from process.util import load_challenge_data, load_labels, load_label_files
from augmentation.transformers import Jitter, Scaling, MagWarp, TimeWarp, Permutation, RandSampling
import torch

def plot_aug(data, data_aug, header_data, label, name, j, aug, save_path):
    fig, axs = plt.subplots(12, 1, sharey=True, figsize=(50, 50))

    for i in range(12):
        axs[i].plot(data[i])
        axs[i].plot(data_aug[i], color = 'red')
        axs[i].set_title(header_data[i+1])
        axs[i].autoscale(enable=True, axis='both', tight=True)

    label = list(label)
    save_path_label = label[0]
    if len(label) > 1:
        for i in range(len(label)-1):
            save_path_label += ' %s' %(label[i+1])

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    save_path_label = os.path.join(save_path, save_path_label)

    if not os.path.exists(save_path_label):
        os.mkdir(save_path_label)

    plt.savefig(os.path.join(save_path_label, '%s_%d_%s.png' %(name, j, aug)))
    plt.show()
    plt.close()

if __name__ == '__main__':

    # Define the weights, the SNOMED CT code for the normal class, and equivalent SNOMED CT codes.
    weights_file = 'weights.csv'
    normal_class = '426783006'
    equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]

    input_directory_data = '/home/weiyuhua/Data/All_data_resampled_to_500HZ_and_filtered_slided_n_segment=1_meanIR=100'
    input_directory_label = '/home/weiyuhua/Data/challenge2020'

    save_path = '/home/weiyuhua/Data/challenge2020_plots/aug'

    # Find the label files.
    print('Finding label and output files...')
    label_files = load_label_files(input_directory_label)

    # Load the labels and classes.
    print('Loading labels and outputs...')
    label_classes, labels_onehot, labels = load_labels(label_files, normal_class, equivalent_classes)

    num_files = len(label_files)
    print("num_files:", num_files)

    # Load data and plot
    augmentation = ['Jitter', 'Scaling', 'MagWarp', 'TimeWarp', 'Permutation', 'RandSampling']

    for i, (f, label) in enumerate(zip(label_files, labels)):
        print('    {}/{}...'.format(i + 1, num_files))
        file = os.path.basename(f)
        name, ext = os.path.splitext(file)
        data, header_data = load_challenge_data(file, input_directory_label, input_directory_data)
        for j in range(data.shape[0]):
            data_j = torch.from_numpy(data[j])
            data_aug = torch.zeros((6, *data_j.shape))

            jitter = Jitter()
            scaling = Scaling()
            magWarp = MagWarp()
            timeWarp = TimeWarp()
            permutation = Permutation()
            randSampling = RandSampling()

            data_aug[0] = jitter(data_j)
            data_aug[1] = scaling(data_j)
            data_aug[2] = magWarp(data_j)
            data_aug[3] = timeWarp(data_j)
            data_aug[4] = permutation(data_j)
            data_aug[5] = randSampling(data_j)

            for k in range(6):
                plot_aug(data_j, data_aug[k], header_data, label, name, j, augmentation[k], save_path)






