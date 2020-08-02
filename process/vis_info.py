import os
import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from process.util import load_challenge_data, load_labels, load_label_files

def plot(data, header_data, label, name, save_path):
    fig, axs = plt.subplots(12, 1, sharey=True, figsize=(50, 50))

    mm = MinMaxScaler()
    data = data.swapaxes(0, 1)
    data_scaled = mm.fit_transform(data)
    data_scaled = data_scaled.swapaxes(0, 1)
    for i in range(12):
        # axs[i].set_autoscale_on(True)
        axs[i].plot(data_scaled[i,:])
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

    plt.savefig(os.path.join(save_path_label, '%s.png' %(name)))
    plt.close()

def count(label, class_count):

    label = list(label)
    labels = label[0]
    if len(label) > 1:
        for i in range(len(label)-1):
            labels += ' %s' %(label[i+1])

    if labels not in class_count:
        class_count[labels] = 1
    else:
        class_count[labels] += 1

if __name__ == '__main__':

    # Define the weights, the SNOMED CT code for the normal class, and equivalent SNOMED CT codes.
    weights_file = 'weights.csv'
    normal_class = '426783006'
    equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]

    input_directory  = '/home/weiyuhua/Data/challenge2020'
    save_path = '/home/weiyuhua/Data/challenge2020_plots'

    # Find the label files.
    print('Finding label and output files...')
    label_files = load_label_files(input_directory)

    # Load the labels and classes.
    print('Loading labels and outputs...')
    label_classes, labels_onehot, labels = load_labels(label_files, normal_class, equivalent_classes)

    num_files = len(label_files)
    print("num_files:", num_files)

    # Load data and plot
    class_count = {}

    for i, (f, label) in enumerate(zip(label_files, labels)):
        print('    {}/{}...'.format(i + 1, num_files))
        file = os.path.basename(f)
        name, ext = os.path.splitext(file)
        data, header_data = load_challenge_data(file, input_directory, input_directory)
        # plot(data, header_data, label, name, save_path)
        count(label, class_count)

    print("Done.")
    save_path_cc = '/home/weiyuhua/Data/challenge2020_info'
    savemat(os.path.join(save_path_cc, 'class_count.mat'), {'val': class_count})

    class_count_df = pd.DataFrame.from_dict(class_count, orient='index')
    class_count_df.to_csv(os.path.join(save_path_cc, 'class_count.csv'))




