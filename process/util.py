import os
import numpy as np
from scipy.io import loadmat, savemat
import neurokit2 as nk
import matplotlib.pyplot as plt

# Find Challenge files.
def load_label_files(label_directory):
    label_files = list()
    for f in sorted(os.listdir(label_directory)):
        F = os.path.join(label_directory, f) # Full path for label file
        if os.path.isfile(F) and F.lower().endswith('.hea') and not f.lower().startswith('.'):
            # root, ext = os.path.splitext(f)
            label_files.append(F)
    if label_files:
        return label_files
    else:
        raise IOError('No label or output files found.')

# Load labels from header/label files.
def load_labels(label_files, normal_class, equivalent_classes_collection):
    # The labels_onehot should have the following form:
    #
    # Dx: label_1, label_2, label_3
    #
    num_recordings = len(label_files)

    # Load diagnoses.
    tmp_labels = list()
    for i in range(num_recordings):
        with open(label_files[i], 'r') as f:
            for l in f:
                if l.startswith('#Dx'):
                    dxs = set(arr.strip() for arr in l.split(': ')[1].split(','))
                    tmp_labels.append(dxs)

    # Identify classes.
    classes = set.union(*map(set, tmp_labels))
    if normal_class not in classes:
        classes.add(normal_class)
        print('- The normal class {} is not one of the label classes, so it has been automatically added, but please check that you chose the correct normal class.'.format(normal_class))
    classes = sorted(classes)
    num_classes = len(classes)

    # Use one-hot encoding for labels.
    labels_onehot = np.zeros((num_recordings, num_classes), dtype=np.bool)
    for i in range(num_recordings):
        dxs = tmp_labels[i]
        for dx in dxs:
            j = classes.index(dx)
            labels_onehot[i, j] = 1

    # For each set of equivalent class, use only one class as the representative class for the set and discard the other classes in the set.
    # The label for the representative class is positive if any of the labels_onehot in the set is positive.
    remove_classes = list()
    remove_indices = list()
    for equivalent_classes in equivalent_classes_collection:
        equivalent_classes = [x for x in equivalent_classes if x in classes]
        if len(equivalent_classes)>1:
            representative_class = equivalent_classes[0]
            other_classes = equivalent_classes[1:]
            equivalent_indices = [classes.index(x) for x in equivalent_classes]
            representative_index = equivalent_indices[0]
            other_indices = equivalent_indices[1:]

            labels_onehot[:, representative_index] = np.any(labels_onehot[:, equivalent_indices], axis=1)
            remove_classes += other_classes
            remove_indices += other_indices

    for x in remove_classes:
        classes.remove(x)
    labels_onehot = np.delete(labels_onehot, remove_indices, axis=1)

    # If the labels_onehot are negative for all classes, then change the label for the normal class to positive.
    normal_index = classes.index(normal_class)
    for i in range(num_recordings):
        num_positive_classes = np.sum(labels_onehot[i, :])
        if num_positive_classes==0:
            labels_onehot[i, normal_index] = 1

    labels = list()
    for i in range(num_recordings):
        class_list = []
        for j in range(len(classes)):
            if labels_onehot[i][j] == True:
                class_list.append(classes[j])
        class_set = set()
        class_set.update(class_list)
        labels.append(class_set)

    return classes, labels_onehot, labels

# Load Data from input directory
def load_challenge_data(label_filename, input_directory_label, input_directory_data):

    filename = label_filename.replace('.hea','.mat')
    input_header_file = os.path.join(input_directory_label, label_filename)

    x = loadmat(os.path.join(input_directory_data, filename))
    data = np.asarray(x['val'], dtype=np.float64)

    with open(input_header_file,'r') as f:
        header_data=f.readlines()
    return data, header_data

# Load weights.
def load_weights(weight_file, classes):
    # Load the weight matrix.
    rows, cols, values = load_table(weight_file)
    assert(rows == cols)
    num_rows = len(rows)

    # Assign the entries of the weight matrix with rows and columns corresponding to the classes.
    num_classes = len(classes)
    weights = np.zeros((num_classes, num_classes), dtype=np.float64)
    for i, a in enumerate(rows):
        if a in classes:
            k = classes.index(a)
            for j, b in enumerate(rows):
                if b in classes:
                    l = classes.index(b)
                    weights[k, l] = values[i, j]

    return weights

# Load_table
def load_table(table_file):
    # The table should have the following form:
    #
    # ,    a,   b,   c
    # a, 1.2, 2.3, 3.4
    # b, 4.5, 5.6, 6.7
    # c, 7.8, 8.9, 9.0
    #
    table = list()
    with open(table_file, 'r') as f:
        for i, l in enumerate(f):
            arrs = [arr.strip() for arr in l.split(',')]
            table.append(arrs)

    # Define the numbers of rows and columns and check for errors.
    num_rows = len(table)-1
    if num_rows<1:
        raise Exception('The table {} is empty.'.format(table_file))

    num_cols = set(len(table[i])-1 for i in range(num_rows))
    if len(num_cols)!=1:
        raise Exception('The table {} has rows with different lengths.'.format(table_file))
    num_cols = min(num_cols)
    if num_cols<1:
        raise Exception('The table {} is empty.'.format(table_file))

    # Find the row and column labels.
    rows = [table[0][j+1] for j in range(num_rows)]
    cols = [table[i+1][0] for i in range(num_cols)]

    # Find the entries of the table.
    values = np.zeros((num_rows, num_cols))
    for i in range(num_rows):
        for j in range(num_cols):
            value = table[i+1][j+1]
            if is_number(value):
                values[i, j] = float(value)
            else:
                values[i, j] = float('nan')

    return rows, cols, values

def is_number(x):
    try:
        float(x)
        return True
    except ValueError:
        return False

def getRpeakSet(ecg, sampling_rate):
    ecg_II = ecg[1]
    processed_ecg = nk.ecg_process(ecg_II, sampling_rate)
    rpeaks = processed_ecg[1]['ECG_R_Peaks']
    return rpeaks

def plot_ecg_filled(ecg, ecg_filled, save_file):
    fig, axs = plt.subplots(12, 1, sharey=True, figsize=(25, 25))

    for i in range(12):
        axs[i].plot(ecg[i], color='blue')
        axs[i].plot(ecg_filled[i], color='red')
        axs[i].autoscale(enable=True, axis='both', tight=True)

    plt.savefig(save_file)
    # plt.show()
    plt.close()

def ecg_filling(ecg, sampling_rate, length, save_file):
    rpeaks = getRpeakSet(ecg, sampling_rate)
    ecg_filled = np.zeros((ecg.shape[0], length))
    sta = rpeaks[-1]
    ecg_filled[:, :sta] = ecg[:, :sta]
    seg = ecg[:, rpeaks[0]:rpeaks[-1]]
    len = seg.shape[1]
    while True:
        if (sta + len) >= length:
            ecg_filled[:, sta: length] = seg[:, : length-sta]
            break
        else:
            ecg_filled[:, sta: sta+len] = seg[:, :]
            sta = sta + len

    plot_ecg_filled(ecg, ecg_filled, save_file)
    return ecg_filled

# Plot Ecg
# channels = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

#
# ecg = loadmat('/DATASET/challenge2020/All_data_resampled_to_300HZ/E09943.mat')['val']
# ecg_filled = ecg_filling(ecg, sampling_rate=300, length=3000, save_file='./test.png')
# plot_ecg_filled(ecg, ecg_filled)

# nk error
ecg = loadmat('/DATASET/challenge2020/All_data_resampled_to_300HZ/E09943.mat')['val']
ecg_filled = np.concatenate((ecg, ecg[:, :3000-ecg.shape[1]]),axis=1)
plot_ecg_filled(ecg, ecg_filled, save_file='../info/E09943.png')



