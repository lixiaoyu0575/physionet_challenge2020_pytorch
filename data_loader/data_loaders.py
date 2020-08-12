import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
import torch
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, Dataset
from base import BaseDataLoader, BaseDataLoader2, BaseDataLoader3
from utils.dataset import ECGDataset
from sklearn.preprocessing import MinMaxScaler
from utils.dataset import load_label_files, load_labels, load_weights
from scipy.signal import savgol_filter, medfilt, wiener
from data_loader.preprocessing import cheby_lowpass_filter, butter_lowpass_filter, plot
from data_loader.util import load_challenge_data, get_classes, CustomTensorDataset, CustomTensorListDataset, custom_collate_fn
import augmentation.transformers as module_transformers
import random
import time
import pickle
import scipy

from utils.util import smooth_labels

# official data
class ChallengeDataLoader0(BaseDataLoader2):
    """
    challenge2020 data loading
    """
    def __init__(self, label_dir, data_dir, split_index, batch_size, shuffle=True, num_workers=2, training=True, training_size=None, augmentations=None):
        start = time.time()
        self.label_dir = label_dir
        self.data_dir = data_dir
        print('Loading data...')

        weights_file = 'evaluation/weights.csv'
        normal_class = '426783006'
        equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]

        # Find the label files.
        print('Finding label...')
        label_files = load_label_files(label_dir)
        time_record_1 = time.time()
        print("Finding label cost {}s".format(time_record_1-start))

        # Load the labels and classes.
        print('Loading labels...')
        classes, labels_onehot, labels = load_labels(label_files, normal_class, equivalent_classes)
        time_record_2 = time.time()
        print("Loading label cost {}s".format(time_record_2-time_record_1))

        # Load the weights for the Challenge metric.
        print('Loading weights...')
        weights = load_weights(weights_file, classes)
        self.weights = weights
        time_record_3 = time.time()
        print("Loading weights cost {}s".format(time_record_3-time_record_2))

        # Classes that are scored with the Challenge metric.
        indices = np.any(weights, axis=0)  # Find indices of classes in weight matrix.
        # from scipy.io import savemat
        # savemat('evaluation/scored_classes_indices.mat', {'val': indices})

        # Load short signals and remove from labels
        # short_signals = loadmat(os.path.join(data_dir, 'short_signals.mat'))['val']
        # short_signals_ids = list(short_signals.reshape((short_signals.shape[1], )))

        split_idx = loadmat(split_index)
        train_index, val_index, test_index = split_idx['train_index'], split_idx['val_index'], split_idx['test_index']
        train_index = train_index.reshape((train_index.shape[1], ))
        if training_size is not None:
            train_index = train_index[0:training_size]
        val_index = val_index.reshape((val_index.shape[1], ))
        test_index = test_index.reshape((test_index.shape[1], ))

        num_files = len(label_files)
        recordings = list()
        labels_onehot_new = list()
        labels_new = list()
        file_names = list()

        bb = []
        dd = []

        for i in range(num_files):
            # if i in short_signals_ids:
            #     recording = np.zeros((1, 12, 3000))
            #
            # else:
            recording, header, name = load_challenge_data(label_files[i], data_dir)
            recording[np.isnan(recording)] = 0
            recordings.append(recording)
            file_names.append(name)

            rr = np.array(recording)
            if np.isnan(rr).any():
                print(i)
                bb.append(i)
                dd.append(rr)

            labels_onehot_new.append(labels_onehot[i])
            labels_new.append(labels[i])

        time_record_4 = time.time()
        print("Loading data cost {}s".format(time_record_4-time_record_3))

        for i in range(len(recordings)):
            if np.isnan(recordings[i]).any():
                print(i)

        # slided data
        recordings_all = list()
        labels_onehot_all = list()
        labels_all = list()

        for i in range(len(recordings)):
            for j in range(recordings[i].shape[0]):
                recordings_all.append(recordings[i][j])
                labels_onehot_all.append(labels_onehot_new[i])
                labels_all.append(labels_new[i])

        recordings_all = np.array(recordings_all)
        labels_onehot_all = np.array(labels_onehot_all)
        # np.random.shuffle(labels_onehot_all)
        # labels_onehot = np.array(labels_onehot, dtype='float64')
        # labels_onehot = smooth_labels(labels_onehot)
        print(np.isnan(recordings_all).any())

        num = recordings_all.shape[0]
        c = []
        a = []
        for i in range(num):
            if np.isnan(recordings_all[i]).any():
                print(' {}/{}'.format(i, num))
                c.append(i)
                a.append(recordings_all[i])
        print(c)
        print(a)

        # Get number of samples for each category
        self.count = np.sum(labels_onehot_all, axis=0)
        self.indices = indices
        # recordings_all = scipy.signal.detrend(recordings_all)
        X = torch.from_numpy(recordings_all).float()
        # Y = torch.from_numpy(labels_onehot)
        Y = torch.from_numpy(labels_onehot_all).float()

        #add augmentation

        if augmentations:
            transformers = list()

            for key, value in augmentations.items():
                module_args = dict(value['args'])
                transformers.append(getattr(module_transformers, key)(**module_args))

            train_transform = transforms.Compose(transformers)
            self.dataset = CustomTensorDataset(X, Y, transform=train_transform)
        else:
            self.dataset = TensorDataset(X, Y)

        end = time.time()
        print('time to get and process data: {}'.format(end-start))
        super().__init__(self.dataset, batch_size, shuffle, train_index, val_index, test_index, num_workers)

        self.valid_data_loader.file_names = file_names
        self.test_data_loader.file_names = file_names

# official data (filtered\balanced\slided_and_cut)
class ChallengeDataLoader1(BaseDataLoader2):
    """
    challenge2020 data loading
    """
    def __init__(self, label_dir, data_dir, batch_size, shuffle=True, validation_split=0.0, test_split=0.0, num_workers=1, training=True):
        self.label_dir = label_dir
        self.data_dir = data_dir
        print('Loading data...')

        weights_file = 'evaluation/weights.csv'
        normal_class = '426783006'
        equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]

        # Find the label files.
        print('Finding label...')
        label_files = load_label_files(label_dir)

        # Load the labels and classes.
        print('Loading labels...')
        classes, labels_onehot, labels = load_labels(label_files, normal_class, equivalent_classes)

        # Load the weights for the Challenge metric.
        print('Loading weights...')
        weights = load_weights(weights_file, classes)

        # Classes that are scored with the Challenge metric.
        indices = np.any(weights, axis=0)  # Find indices of classes in weight matrix.
        # from scipy.io import savemat
        # savemat('evaluation/scored_classes_indices.mat', {'val': indices})

        # Load short signals and remove from labels
        # short_signals = loadmat(os.path.join(data_dir, 'short_signals.mat'))['val']
        # short_signals_ids = list(short_signals.reshape((short_signals.shape[1], )))

        num_files = len(label_files)
        recordings = list()
        labels_onehot_new = list()
        labels_new = list()

        bb = []
        dd = []

        for i in range(num_files):
            # if i in short_signals_ids:
            #     continue
            recording, header = load_challenge_data(label_files[i], data_dir)
            recording[np.isnan(recording)] = 0
            recordings.append(recording)

            rr = np.array(recording)
            if np.isnan(rr).any():
                print(i)
                bb.append(i)
                dd.append(rr)

            labels_onehot_new.append(labels_onehot[i])
            labels_new.append(labels[i])

        # shuffle
        recordings_shuffled, labels_onehot_shuffled, labels_shuffled = self.shuffle(recordings, labels_onehot_new, labels_new)

        for i in range(len(recordings_shuffled)):
            if np.isnan(recordings_shuffled[i]).any():
                print(i)

        # slided data
        recordings_all = list()
        labels_onehot_all = list()
        labels_all = list()

        for i in range(len(recordings_shuffled)):
            for j in range(recordings_shuffled[i].shape[0]):
                recordings_all.append(recordings_shuffled[i][j])
                labels_onehot_all.append(labels_onehot_shuffled[i])
                labels_all.append(labels_shuffled[i])

        recordings_all = np.array(recordings_all)
        labels_onehot_all = np.array(labels_onehot_all)

        print(np.isnan(recordings_all).any())

        num = recordings_all.shape[0]
        c = []
        a = []
        for i in range(num):
            if np.isnan(recordings_all[i]).any():
                print(' {}/{}'.format(i, num))
                c.append(i)
                a.append(recordings_all[i])
        print(c)
        print(a)

        # Get number of samples for each category
        self.count = np.sum(labels_onehot_all, axis=0)
        self.indices = indices

        X = torch.from_numpy(recordings_all).float()
        # Y = torch.from_numpy(labels_onehot)
        Y = torch.from_numpy(labels_onehot_all.astype(int))

        self.dataset = TensorDataset(X, Y)

        train_idx, valid_idx, test_idx = self.split_val_test(len(Y), validation_split, test_split)

        super().__init__(self.dataset, batch_size, shuffle, train_idx, valid_idx, test_idx, num_workers)

    def shuffle(self, recordings, labels_onehot, labels):
        randnum = random.randint(0, 100)
        random.seed(randnum)
        random.shuffle(recordings)
        random.seed(randnum)
        random.shuffle(labels_onehot)
        random.seed(randnum)
        random.shuffle(labels)

        return recordings, labels_onehot, labels

    def split_val_test(self, n_sample, validation_split, test_split):

        idx_full = np.arange(n_sample)
        valid_idx = idx_full[-int(n_sample*(validation_split+test_split)): -int(n_sample*test_split)]
        test_idx = idx_full[-int(n_sample*test_split):]
        train_idx = idx_full[:-int(n_sample*(validation_split+test_split))]

        return train_idx, valid_idx, test_idx

# DataLoader for variable length data
class ChallengeDataLoader7(BaseDataLoader2):
    """
    challenge2020 data loading
    """
    def __init__(self, label_dir, data_dir, split_index, batch_size, shuffle=True, num_workers=2, training=True):
        self.label_dir = label_dir
        self.data_dir = data_dir
        print('Loading data...')

        weights_file = 'evaluation/weights.csv'
        normal_class = '426783006'
        equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]

        # Find the label files.
        print('Finding label...')
        label_files = load_label_files(label_dir)

        # Load the labels and classes.
        print('Loading labels...')
        classes, labels_onehot, labels = load_labels(label_files, normal_class, equivalent_classes)

        # Load the weights for the Challenge metric.
        print('Loading weights...')
        weights = load_weights(weights_file, classes)

        # Classes that are scored with the Challenge metric.
        indices = np.any(weights, axis=0)  # Find indices of classes in weight matrix.
        # from scipy.io import savemat
        # savemat('evaluation/scored_classes_indices.mat', {'val': indices})

        # Load short signals and remove from labels
        # short_signals = loadmat(os.path.join(data_dir, 'short_signals.mat'))['val']
        # short_signals_ids = list(short_signals.reshape((short_signals.shape[1], )))

        split_idx = loadmat(split_index)
        train_index, val_index, test_index = split_idx['train_index'], split_idx['val_index'], split_idx['test_index']
        train_index = train_index.reshape((train_index.shape[1], ))
        val_index = val_index.reshape((val_index.shape[1], ))
        test_index = test_index.reshape((test_index.shape[1], ))

        num_files = len(label_files)
        recordings = list()
        labels_onehot_new = list()
        labels_new = list()
        file_names = list()

        bb = []
        dd = []

        for i in range(num_files):
            # if i in short_signals_ids:
            #     recording = np.zeros((1, 12, 3000))
            #
            # else:
            recording, header, name = load_challenge_data(label_files[i], data_dir)
            if len(recording.shape) > 2:
                print("******************************************************************************8")
            recording[np.isnan(recording)] = 0
            recordings.append(recording)
            file_names.append(name)

            rr = np.array(recording)
            if np.isnan(rr).any():
                print(i)
                bb.append(i)
                dd.append(rr)

            labels_onehot_new.append(labels_onehot[i])
            labels_new.append(labels[i])
            # print(i)

        for i in range(len(recordings)):
            if np.isnan(recordings[i]).any():
                print(i)

        # recordings = np.array(recordings)
        # labels_onehot_new = np.array(labels_onehot_new)

        # recordings_preprocessed, labels_onehot_new = self.preprocessing(recordings, labels_onehot_new)
        # recordings_augmented, labels_onehot_new = self.augmentation(recordings_preprocessed, labels_onehot_new)

        # print(np.isnan(recordings_augmented).any())
        #
        # num = recordings_augmented.shape[0]
        # c = []
        # a = []
        # for i in range(num):
        #     if np.isnan(recordings_augmented[i]).any():
        #         print(' {}/{}'.format(i, num))
        #         c.append(i)
        #         a.append(recordings_augmented[i])
        # print(c)
        # print(a)

        # Get number of samples for each category

        self.indices = indices

        X = []
        Y = []

        for i in range(len(labels_onehot_new)):
            X.append(torch.from_numpy(recordings[i]).float())
            Y.append(torch.from_numpy(labels_onehot[i].astype(int)))

        self.dataset = CustomTensorListDataset(X, Y)

        super().__init__(self.dataset, batch_size, shuffle, train_index, val_index, test_index, num_workers, collate_fn=custom_collate_fn, pin_memory=False)

        self.valid_data_loader.file_names = file_names
        self.test_data_loader.file_names = file_names

# DataLoader
class ChallengeDataLoader2(BaseDataLoader3):
    """
    challenge2020 data loading
    """
    def __init__(self, label_dir, data_dir, split_index, batch_size, shuffle=True, num_workers=2, training=True, training_size=None, augmentations=None):
        start = time.time()
        self.label_dir = label_dir
        self.data_dir = data_dir
        print('Loading data...')

        weights_file = 'evaluation/weights.csv'
        normal_class = '426783006'
        equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]

        # Find the label files.
        print('Finding label...')
        label_files = load_label_files(label_dir)
        time_record_1 = time.time()
        print("Finding label cost {}s".format(time_record_1 - start))

        # Load the labels and classes.
        print('Loading labels...')
        classes, labels_onehot, labels = load_labels(label_files, normal_class, equivalent_classes)
        time_record_2 = time.time()
        print("Loading label cost {}s".format(time_record_2 - time_record_1))

        # Load the weights for the Challenge metric.
        print('Loading weights...')
        weights = load_weights(weights_file, classes)
        self.weights = weights
        time_record_3 = time.time()
        print("Loading weights cost {}s".format(time_record_3 - time_record_2))

        # Classes that are scored with the Challenge metric.
        indices = np.any(weights, axis=0)  # Find indices of classes in weight matrix.

        split_idx = loadmat(split_index)
        train_index, val_index, test_index = split_idx['train_index'], split_idx['val_index'], split_idx['test_index']
        train_index = train_index.reshape((train_index.shape[1],))
        if training_size is not None:
            train_index = train_index[0:training_size]
        val_index = val_index.reshape((val_index.shape[1],))
        test_index = test_index.reshape((test_index.shape[1],))

        num_files = len(label_files)
        train_recordings = list()
        train_labels_onehot = list()

        val_recordings = list()
        val_labels_onehot = list()

        test_recordings = list()
        test_labels_onehot = list()

        file_names = list()

        for i in range(num_files):
            recording, header, name = load_challenge_data(label_files[i], data_dir)
            recording[np.isnan(recording)] = 0

            file_names.append(name)
            if i in train_index:
                for j in range(recording.shape[0]):
                    train_recordings.append(recording[j])
                    train_labels_onehot.append(labels_onehot[i])
            elif i in val_index:
                for j in range(recording.shape[0]):
                    val_recordings.append(recording[j])
                    val_labels_onehot.append(labels_onehot[i])
            else:
                for j in range(recording.shape[0]):
                    test_recordings.append(recording[j])
                    test_labels_onehot.append(labels_onehot[i])

        time_record_4 = time.time()
        print("Loading data cost {}s".format(time_record_4 - time_record_3))

        print(np.isnan(train_recordings).any())
        print(np.isnan(val_recordings).any())
        print(np.isnan(test_recordings).any())

        # Get number of samples for each category
        self.indices = indices

        train_recordings = np.array(train_recordings)
        train_labels_onehot = np.array(train_labels_onehot)

        val_recordings = np.array(val_recordings)
        val_labels_onehot = np.array(val_labels_onehot)

        test_recordings = np.array(test_recordings)
        test_labels_onehot = np.array(test_labels_onehot)

        X_train = torch.from_numpy(train_recordings).float()
        Y_train = torch.from_numpy(train_labels_onehot).float()

        X_val = torch.from_numpy(val_recordings).float()
        Y_val = torch.from_numpy(val_labels_onehot).float()

        X_test = torch.from_numpy(test_recordings).float()
        Y_test = torch.from_numpy(test_labels_onehot).float()

        #############################################################
        if augmentations:
            transformers = list()

            for key, value in augmentations.items():
                module_args = dict(value['args'])
                transformers.append(getattr(module_transformers, key)(**module_args))

            train_transform = transforms.Compose(transformers)
            self.train_dataset = CustomTensorDataset(X_train, Y_train, transform=train_transform)
        else:
            self.train_dataset = TensorDataset(X_train, Y_train)
        #############################################################

        self.val_dataset = TensorDataset(X_val, Y_val)
        self.test_dataset = TensorDataset(X_test, Y_test)

        end = time.time()
        print('time to get and process data: {}'.format(end - start))
        super().__init__(self.train_dataset, self.val_dataset, self.test_dataset, batch_size, shuffle, num_workers)

        self.valid_data_loader.file_names = file_names
        self.valid_data_loader.idx = val_index
        self.test_data_loader.file_names = file_names
        self.test_data_loader.idx = test_index
