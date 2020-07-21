import numpy as np
import os
from scipy.io import loadmat, savemat
from process.utils import load_challenge_data, load_labels, load_label_files
import time

def IRLbl(labels):
    # imbalance ratio per label
    # Args:
    #	 labels is a 2d numpy array, each row is one instance, each column is one class; the array contains (0, 1) only
    N, C = labels.shape
    pos_nums_per_label = np.sum(labels, axis=0)
    max_pos_nums = np.max(pos_nums_per_label)
    return max_pos_nums / pos_nums_per_label

def MeanIR(labels):
    IRLbl_VALUE = IRLbl(labels)
    return np.mean(IRLbl_VALUE)

def ML_ROS(all_labels, indices=None, num_samples=None, Preset_MeanIR_value=2.,
                 max_clone_percentage=50, sample_size=32):
    # the index of samples: 0, 1, ....
    # if indices is not provided,
    # all elements in the dataset will be considered
    indices = list(range(len(all_labels))) \
        if indices is None else indices

    # if num_samples is not provided,
    # draw `len(indices)` samples in each iteration
    num_samples = len(indices) \
        if num_samples is None else num_samples

    MeanIR_value = MeanIR(all_labels) if Preset_MeanIR_value == 0 else Preset_MeanIR_value
    IRLbl_value = IRLbl(all_labels)
    # N is the number of samples, C is the number of labels
    N, C = all_labels.shape
    # the samples index of every class
    indices_per_class = {}
    minority_classes = []
    # accroding to psedu code, maxSamplesToClone is the upper limit of the number of samples can be copied from original dataset
    maxSamplesToClone = N / 100 * max_clone_percentage
    print('Max Clone Limit:', maxSamplesToClone)
    for i in range(C):
        ids = all_labels[:, i] == 1
        # How many samples are there for each label
        indices_per_class[i] = [ii for ii, x in enumerate(ids) if x]
        if IRLbl_value[i] > MeanIR_value:
            minority_classes.append(i)

    new_all_labels = all_labels
    oversampled_ids = []
    minorNum = len(minority_classes)
    print(minorNum, 'minor classes.')

    for idx, i in enumerate(minority_classes):
        tid = time.time()
        while True:
            pick_id = list(np.random.choice(indices_per_class[i], sample_size))
            indices_per_class[i].extend(pick_id)
            # recalculate the IRLbl_value
            # The original label matrix (New_ all_ Labels) and randomly selected label matrix (all_ labels[pick_ ID) and recalculate the irlbl
            new_all_labels = np.concatenate([new_all_labels, all_labels[pick_id]], axis=0)
            oversampled_ids.extend(pick_id)

            newIrlbl = IRLbl(new_all_labels)
            if newIrlbl[i] <= MeanIR_value:
                print('\nMeanIR satisfied.', newIrlbl[i])
                break
            if len(oversampled_ids) >= maxSamplesToClone:
                print('\nExceed max clone.', len(oversampled_ids))
                break
            # if IRLbl(new_all_labels)[i] <= MeanIR_value or len(oversampled_ids) >= maxSamplesToClone:
            #     break
            print("\roversample length:{}".format(len(oversampled_ids)), end='')
        print('Processed the %d/%d minor class:' % (idx+1, minorNum), i, time.time()-tid, 's')
        if len(oversampled_ids) >= maxSamplesToClone:
            print('Exceed max clone. Exit', len(oversampled_ids))
            break
    return new_all_labels, oversampled_ids

def slide_and_cut(X, file_name, oversampleIds, preset_n_segment, save_path):
    n_sample = len(X)
    window_size = 3000
    short_signals = []
    short_signals_file = []
    for i in range(n_sample):
        if i in oversampleIds:
            i_count = oversampleIds.count(i)
            oversampleIds.remove(i)
        else:
            i_count = 1
        n_segment = preset_n_segment * i_count

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

    input_directory_data = '/DATASET/challenge2020/All_data_resampled_to_500HZ_and_filtered'
    input_directory_label = '/DATASET/challenge2020/All_data'
    save_path = '/DATASET/challenge2020/All_data_resampled_to_500HZ_and_filtered_slided_n_segment=1_meanIR=100'

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

    # Slide and cut using sliding window with ML_ROS oversampling
    newLables, oversampleIds = ML_ROS(labels_onehot, indices=None, num_samples=None, Preset_MeanIR_value=100,
                                      max_clone_percentage=50, sample_size=32)

    # Load data
    for i, (f, label) in enumerate(zip(label_files, labels)):
        print('    {}/{}...'.format(i + 1, num_files))
        file = os.path.basename(f)
        name, ext = os.path.splitext(file)
        data, header_data = load_challenge_data(file, input_directory_label, input_directory_data)
        dataset.append(data)
        file_name.append(name)

    slide_and_cut(dataset, file_name, oversampleIds, preset_n_segment=1, save_path=save_path)
    print("Done")


