import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Define the weights, the SNOMED CT code for the normal class, and equivalent SNOMED CT codes.
weights_file = 'weights.csv'
normal_class = '426783006'
equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]

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
def load_challenge_data(label_filename):

    filename = label_filename.replace('.hea','.mat')
    input_header_file = os.path.join(label_filename)

    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)

    with open(input_header_file,'r') as f:
        header_data=f.readlines()
    return data, header_data

# Plot Ecg
channels = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

def plot(data, header_data, label, label_file, save_path):
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

    file = os.path.basename(label_file)
    name, ext = os.path.splitext(file)

    save_path_label = os.path.join(save_path, save_path_label)

    if not os.path.exists(save_path_label):
        os.mkdir(save_path_label)

    plt.savefig(os.path.join(save_path_label, '%s.png' %(name)))
    plt.close()

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
    for i, (f, label) in enumerate(zip(label_files, labels)):
        print('    {}/{}...'.format(i + 1, num_files))
        tmp_label_file = os.path.join(input_directory, f)
        data, header_data = load_challenge_data(tmp_label_file)
        plot(data, header_data, label, tmp_label_file, save_path)

