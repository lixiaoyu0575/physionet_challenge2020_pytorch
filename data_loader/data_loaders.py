import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
import torch
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, Dataset
from base import BaseDataLoader, BaseDataLoader2
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

from utils.util import smooth_labels

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class ChallengeDataLoader(BaseDataLoader):
    """
    challenge2020 data loading
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, test_split=0.0, num_workers=1, training=True):
        self.data_dir = data_dir
        self.dataset = ECGDataset(self.data_dir)
        # trsfm = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.1307,), (0.3081,))
        # ])
        # self.data_dir = data_dir
        # self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, test_split, num_workers)

# official data
class ChallengeDataLoader0(BaseDataLoader2):
    """
    challenge2020 data loading
    """
    def __init__(self, label_dir, data_dir, split_index, batch_size, shuffle=True, num_workers=2, training=True, training_size=None):
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
        time_record_3 = time.time()
        print("Loading label cost {}s".format(time_record_3-time_record_2))

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

        recordings_preprocessed, labels_onehot = self.preprocessing(recordings_all, labels_onehot_all)
        recordings_augmented, labels_onehot = self.augmentation(recordings_preprocessed, labels_onehot_all)

        # labels_onehot = np.array(labels_onehot, dtype='float64')
        # labels_onehot = smooth_labels(labels_onehot)
        print(np.isnan(recordings_augmented).any())

        num = recordings_augmented.shape[0]
        c = []
        a = []
        for i in range(num):
            if np.isnan(recordings_augmented[i]).any():
                print(' {}/{}'.format(i, num))
                c.append(i)
                a.append(recordings_augmented[i])
        print(c)
        print(a)

        # Get number of samples for each category
        self.count = np.sum(labels_onehot, axis=0)
        self.indices = indices

        X = torch.from_numpy(recordings_augmented).float()
        # Y = torch.from_numpy(labels_onehot)
        Y = torch.from_numpy(labels_onehot.astype(int))
        self.dataset = TensorDataset(X, Y)
        end = time.time()
        print('time to get and process data: {}'.format(end-start))
        super().__init__(self.dataset, batch_size, shuffle, train_index, val_index, test_index, num_workers)

        self.valid_data_loader.file_names = file_names
        self.test_data_loader.file_names = file_names

    def preprocessing(self, recordings, labels):

        # mm = MinMaxScaler()
        # recordings = recordings.swapaxes(1, 2)
        # for i in range(len(recordings)):
        #     recordings[i] = mm.fit_transform(recordings[i])
        # recordings_scaled = recordings.swapaxes(1, 2)
        #
        # recordings_preprocessed = recordings_scaled
        return recordings, labels

    def augmentation(self, recordings, labels):

        recordings_augmented = recordings
        return recordings_augmented, labels

    def shuffle(self, recordings, labels_onehot, labels):
        randnum = random.randint(0, 100)
        random.seed(randnum)
        random.shuffle(recordings)
        random.seed(randnum)
        random.shuffle(labels_onehot)
        random.seed(randnum)
        random.shuffle(labels)

        return recordings, labels_onehot, labels

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

        recordings_preprocessed, labels_onehot = self.preprocessing(recordings_all, labels_onehot_all)
        recordings_augmented, labels_onehot = self.augmentation(recordings_preprocessed, labels_onehot_all)

        print(np.isnan(recordings_augmented).any())

        num = recordings_augmented.shape[0]
        c = []
        a = []
        for i in range(num):
            if np.isnan(recordings_augmented[i]).any():
                print(' {}/{}'.format(i, num))
                c.append(i)
                a.append(recordings_augmented[i])
        print(c)
        print(a)

        # Get number of samples for each category
        self.count = np.sum(labels_onehot, axis=0)
        self.indices = indices

        X = torch.from_numpy(recordings_augmented).float()
        # Y = torch.from_numpy(labels_onehot)
        Y = torch.from_numpy(labels_onehot.astype(int))

        self.dataset = TensorDataset(X, Y)

        train_idx, valid_idx, test_idx = self.split_val_test(len(Y), validation_split, test_split)

        super().__init__(self.dataset, batch_size, shuffle, train_idx, valid_idx, test_idx, num_workers)

    def preprocessing(self, recordings, labels):

        # mm = MinMaxScaler()
        # recordings = recordings.swapaxes(1, 2)
        # for i in range(len(recordings)):
        #     recordings[i] = mm.fit_transform(recordings[i])
        # recordings_scaled = recordings.swapaxes(1, 2)
        #
        # recordings_preprocessed = recordings_scaled
        return recordings, labels

    def augmentation(self, recordings, labels):

        recordings_augmented = recordings
        return recordings_augmented, labels

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

# official data
class ChallengeDataLoader2(BaseDataLoader):
    """
    challenge2020 data loading
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, test_split=0.0, num_workers=1, training=True):
        self.data_dir = data_dir
        print('Loading data...')

        normal_class = '426783006'
        equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]

        # Find the label files.
        print('Finding label...')
        label_files = load_label_files(data_dir)

        # Load the labels and classes.
        print('Loading labels...')
        classes, labels_onehot, labels = load_labels(label_files, normal_class, equivalent_classes)

        num_files = len(label_files)
        recordings = list()

        for i in range(num_files):
            recording, header = load_challenge_data(label_files[i], data_dir)

            ##########################
            mask = np.zeros((12, 18000))
            if recording.shape[1] <= 18000:
                mask[:, :recording.shape[1]] = recording
            else:
                mask[:, :] = recording[:, :18000]
            ##########################

            recordings.append(mask)

        recordings = np.array(recordings)

        recordings_preprocessed, labels_onehot = self.preprocessing(recordings, labels_onehot)
        recordings_augmented, labels_onehot = self.augmentation(recordings_preprocessed, labels_onehot)

        X = torch.from_numpy(recordings_augmented).float()
        Y = torch.from_numpy(labels_onehot)

        self.dataset = TensorDataset(X, Y)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, test_split, num_workers)

    def preprocessing(self, recordings, labels):

        mm = MinMaxScaler()
        recordings = recordings.swapaxes(1, 2)
        for i in range(len(recordings)):
            recordings[i] = mm.fit_transform(recordings[i])
        recordings_scaled = recordings.swapaxes(1, 2)

        recordings_preprocessed = recordings_scaled
        return recordings_preprocessed, labels

    def augmentation(self, recordings, labels):

        recordings_augmented = recordings
        return recordings_augmented, labels

# unofficial data
class ChallengeDataLoader3(BaseDataLoader):
    """
    challenge2020 data loading
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, test_split=0.0, num_workers=1, training=True):
        self.data_dir = data_dir
        print('Loading data...')

        # Find the label files.
        print('Finding label...')
        label_files = load_label_files(data_dir)

        # Load the labels and classes.
        classes = self.get_classes(label_files)

        num_files = len(label_files)
        recordings = list()
        labels_onehot = list()

        for i in range(num_files):
            recording, header = self.load_challenge_data(label_files[i])

            ##########################
            mask = np.zeros((12, 5000))
            if recording.shape[1] <= 5000:
                mask[:, :recording.shape[1]] = recording
            else:
                mask[:, :] = recording[:, :5000]
            ##########################

            recordings.append(mask)

            _, _, label = self.get_true_labels(label_files[i], classes)
            labels_onehot.append(label)

        recordings = np.array(recordings)
        labels_onehot = np.array(labels_onehot)

        recordings_preprocessed, labels_onehot = self.preprocessing(recordings, labels_onehot)
        recordings_augmented, labels_onehot = self.augmentation(recordings_preprocessed, labels_onehot)

        X = torch.from_numpy(recordings_augmented).float()
        Y = torch.from_numpy(labels_onehot)

        self.dataset = TensorDataset(X, Y)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, test_split, num_workers)

    # Find unique number of classes
    def get_classes(self, files):

        classes = set()
        for input_file in files:
            with open(input_file, 'r') as f:
                for lines in f:
                    if lines.startswith('#Dx'):
                        tmp = lines.split(': ')[1].split(',')
                        for c in tmp:
                            classes.add(c.strip())

        return sorted(classes)

    # Find unique true labels
    def get_true_labels(self, input_file, classes):

        classes_label = classes
        single_recording_labels = np.zeros(len(classes), dtype=int)

        with open(input_file, 'r') as f:
            first_line = f.readline()
            recording_label = first_line.split(' ')[0]
            print(recording_label)
            for lines in f:
                if lines.startswith('#Dx'):
                    tmp = lines.split(': ')[1].split(',')
                    for c in tmp:
                        idx = classes.index(c.strip())
                        single_recording_labels[idx] = 1

        return recording_label, classes_label, single_recording_labels

    # Load challenge data.
    def load_challenge_data(self, header_file):
        with open(header_file, 'r') as f:
            header = f.readlines()
        mat_file = header_file.replace('.hea', '.mat')
        x = loadmat(mat_file)
        recording = np.asarray(x['val'], dtype=np.float64)
        return recording, header

    def preprocessing(self, recordings, labels):

        mm = MinMaxScaler()
        recordings = recordings.swapaxes(1, 2)
        for i in range(len(recordings)):
            recordings[i] = mm.fit_transform(recordings[i])
        recordings = recordings.swapaxes(1, 2)

        recordings_filted = np.zeros((recordings.shape[0], recordings.shape[1], recordings.shape[2]))
        for i in range(recordings.shape[0]):
            for j in range(recordings.shape[1]):
                # butter_lowpass_filter\cheby_lowpass_filter\savgol_filter...
                recordings_filted[i, j, :] = butter_lowpass_filter(recordings[i, j, :], cutoff=30)

        # for i in range(recordings.shape[0]):
        #     plot(recordings[i], recordings_filted[i], save_file='data_loader/denoising/%d.png' %(i+1))
        #     if i == 50:
        #         exit(1)
        return recordings_filted, labels

    def augmentation(self, recordings, labels):

        recordings_augmented = recordings
        return recordings_augmented, labels

# unofficial data + data augmentation
class ChallengeDataLoader4(BaseDataLoader):
    """
    challenge2020 data loading
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, test_split=0.0, num_workers=1, augmentations=None, training=True):
        self.data_dir = data_dir
        print('Loading data...')

        # Find the label files.
        print('Finding label...')
        label_files = load_label_files(data_dir)

        # Load the labels and classes.
        classes = self.get_classes(label_files)

        num_files = len(label_files)
        recordings = list()
        labels_onehot = list()

        for i in range(num_files):
            recording, header = self.load_challenge_data(label_files[i])

            ##########################
            mask = np.zeros((12, 5000))
            if recording.shape[1] <= 5000:
                mask[:, :recording.shape[1]] = recording
            else:
                mask[:, :] = recording[:, :5000]
            ##########################

            recordings.append(mask)

            _, _, label = self.get_true_labels(label_files[i], classes)
            labels_onehot.append(label)

        recordings = np.array(recordings)
        labels_onehot = np.array(labels_onehot)

        recordings_preprocessed, labels_onehot = self.preprocessing(recordings, labels_onehot)
        recordings_augmented, labels_onehot = self.augmentation(recordings_preprocessed, labels_onehot)

        X = torch.from_numpy(recordings_augmented).float()
        Y = torch.from_numpy(labels_onehot)

        transformers = list()

        for key, value in augmentations.items():
            module_args = dict(value['args'])
            transformers.append(getattr(module_transformers, key)(**module_args))

        train_transform = transforms.Compose(transformers)

        self.dataset = CustomTensorDataset(X, Y, transform=train_transform)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, test_split, num_workers)

    # Find unique number of classes
    def get_classes(self, files):

        classes = set()
        for input_file in files:
            with open(input_file, 'r') as f:
                for lines in f:
                    if lines.startswith('#Dx'):
                        tmp = lines.split(': ')[1].split(',')
                        for c in tmp:
                            classes.add(c.strip())

        return sorted(classes)

    # Find unique true labels
    def get_true_labels(self, input_file, classes):

        classes_label = classes
        single_recording_labels = np.zeros(len(classes), dtype=int)

        with open(input_file, 'r') as f:
            first_line = f.readline()
            recording_label = first_line.split(' ')[0]
            print(recording_label)
            for lines in f:
                if lines.startswith('#Dx'):
                    tmp = lines.split(': ')[1].split(',')
                    for c in tmp:
                        idx = classes.index(c.strip())
                        single_recording_labels[idx] = 1

        return recording_label, classes_label, single_recording_labels

    # Load challenge data.
    def load_challenge_data(self, header_file):
        with open(header_file, 'r') as f:
            header = f.readlines()
        mat_file = header_file.replace('.hea', '.mat')
        x = loadmat(mat_file)
        recording = np.asarray(x['val'], dtype=np.float64)
        return recording, header

    def preprocessing(self, recordings, labels):

        mm = MinMaxScaler()
        recordings = recordings.swapaxes(1, 2)
        for i in range(len(recordings)):
            recordings[i] = mm.fit_transform(recordings[i])
        recordings = recordings.swapaxes(1, 2)

        recordings_filted = np.zeros((recordings.shape[0], recordings.shape[1], recordings.shape[2]))
        for i in range(recordings.shape[0]):
            for j in range(recordings.shape[1]):
                # butter_lowpass_filter\cheby_lowpass_filter\savgol_filter...
                recordings_filted[i, j, :] = butter_lowpass_filter(recordings[i, j, :], cutoff=30)

        # for i in range(recordings.shape[0]):
        #     plot(recordings[i], recordings_filted[i], save_file='data_loader/denoising/%d.png' %(i+1))
        #     if i == 50:
        #         exit(1)
        return recordings_filted, labels

    def augmentation(self, recordings, labels):

        recordings_augmented = recordings
        return recordings_augmented, labels

# feature data of official data
class ChallengeDataLoader5(BaseDataLoader):
    """
    challenge2020 data loading
    """
    def __init__(self, label_dir, data_dir, batch_size, shuffle=True, validation_split=0.0, test_split=0.0, num_workers=1, training=True):
        self.label_dir =label_dir
        self.data_dir = data_dir

        print('Loading data...')

        normal_class = '426783006'
        equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]

        # Find the label files.
        print('Finding label...')
        feature_df = self.load_challenge_feature_data(data_dir)

        file_names = feature_df.ix[:, 0].values
        label_files = self.get_label_files(label_dir, file_names)

        num_files = len(label_files)

        # Load the labels and classes.
        print('Loading labels...')
        classes, labels_onehot, labels = load_labels(label_files, normal_class, equivalent_classes)

        # Load feature data
        # Convert 'Sex' to int
        def convert(sex):
            if sex == 'Male':
                return 1
            else:
                return 0

        feature_df.ix[:, 2] = feature_df.ix[:, 2].apply(convert)

        recordings_feature = feature_df.ix[:, 1:]

        recordings_feature = np.array(recordings_feature)

        nan_array = np.isnan(recordings_feature)
        print("nan value: ")
        print(np.isnan(recordings_feature).any())

        nan_idx = []
        for i in range(num_files):
            if np.isnan(recordings_feature[i, :]).any():
                nan_idx.append(i)

        recordings_feature = np.delete(recordings_feature, nan_idx, axis=0)
        labels_onehot = np.delete(labels_onehot, nan_idx, axis=0)

        recordings_feature = recordings_feature.reshape((recordings_feature.shape[0], 1, recordings_feature.shape[1]))

        print("remove nan value: ")
        print(np.isnan(recordings_feature).any())

        recordings_feature_preprocessed, labels_onehot = self.preprocessing(recordings_feature, labels_onehot)
        recordings_feature_augmented, labels_onehot = self.augmentation(recordings_feature_preprocessed, labels_onehot)

        X = torch.from_numpy(recordings_feature_augmented).float()
        Y = torch.from_numpy(labels_onehot)

        self.dataset = TensorDataset(X, Y)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, test_split, num_workers, normalization=True)

    # Load challenge data.
    def load_challenge_feature_data(self, data_dir):
        feature_df =  pd.read_csv(data_dir, header=None)
        return feature_df

    # get label_files
    def get_label_files(self, label_dir, file_names):
        label_files = list()
        for f in sorted(file_names):
            F = os.path.join(label_dir, f)
            F = F + '.hea'
            if os.path.isfile(F):
                label_files.append(F)
        if label_files:
            return label_files
        else:
            raise IOError('No label or output files found.')

    def preprocessing(self, recordings, labels):

        return recordings, labels

    def augmentation(self, recordings, labels):

        recordings_augmented = recordings
        return recordings_augmented, labels

# feature data of unofficial data
class ChallengeDataLoader6(BaseDataLoader):
    """
    challenge2020 data loading
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, test_split=0.0, num_workers=1, training=True):
        self.data_dir = data_dir
        print('Loading data...')

        # Find the label files.
        print('Finding label...')
        label_files = load_label_files(data_dir)

        # Load the labels and classes.
        classes = self.get_classes(label_files)

        num_files = len(label_files)
        recordings_feature = self.load_challenge_feature_data(data_dir)
        labels_onehot = list()

        nan_array = np.isnan(recordings_feature)
        print("nan value: ")
        print(np.isnan(recordings_feature).any())

        for i in range(num_files):
            _, _, label = self.get_true_labels(label_files[i], classes)
            labels_onehot.append(label)

        labels_onehot = np.array(labels_onehot)

        nan_idx = []
        for i in range(num_files):
            if np.isnan(recordings_feature[i, :]).any():
                nan_idx.append(i)

        recordings_feature = np.delete(recordings_feature, nan_idx, axis=0)
        labels_onehot = np.delete(labels_onehot, nan_idx, axis=0)

        recordings_feature = recordings_feature.reshape((recordings_feature.shape[0], 1, recordings_feature.shape[1]))

        print("remove nan value: ")
        print(np.isnan(recordings_feature).any())

        recordings_feature_preprocessed, labels_onehot = self.preprocessing(recordings_feature, labels_onehot)
        recordings_feature_augmented, labels_onehot = self.augmentation(recordings_feature_preprocessed, labels_onehot)

        X = torch.from_numpy(recordings_feature_augmented).float()
        Y = torch.from_numpy(labels_onehot)

        self.dataset = TensorDataset(X, Y)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, test_split, num_workers, normalization=True)

    # Find unique number of classes
    def get_classes(self, files):

        classes = set()
        for input_file in files:
            with open(input_file, 'r') as f:
                for lines in f:
                    if lines.startswith('#Dx'):
                        tmp = lines.split(': ')[1].split(',')
                        for c in tmp:
                            classes.add(c.strip())

        return sorted(classes)

    # Find unique true labels
    def get_true_labels(self, input_file, classes):

        classes_label = classes
        single_recording_labels = np.zeros(len(classes), dtype=int)

        with open(input_file, 'r') as f:
            first_line = f.readline()
            recording_label = first_line.split(' ')[0]
            print(recording_label)
            for lines in f:
                if lines.startswith('#Dx'):
                    tmp = lines.split(': ')[1].split(',')
                    for c in tmp:
                        idx = classes.index(c.strip())
                        single_recording_labels[idx] = 1

        return recording_label, classes_label, single_recording_labels

    # Load challenge data.
    def load_challenge_feature_data(self, data_dir):
        file_name = 'OUT_FEATURE.csv'
        feature_data =  np.array(pd.read_csv(os.path.join(data_dir, file_name), header=None))[1:]
        # feature_data = feature_data.reshape((feature_data.shape[0], 1, feature_data.shape[1]))
        return feature_data

    def preprocessing(self, recordings, labels):

        return recordings, labels

    def augmentation(self, recordings, labels):

        recordings_augmented = recordings
        return recordings_augmented, labels

# TCN
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

    def preprocessing(self, recordings, labels):

        # mm = MinMaxScaler()
        # recordings = recordings.swapaxes(1, 2)
        # for i in range(len(recordings)):
        #     recordings[i] = mm.fit_transform(recordings[i])
        # recordings_scaled = recordings.swapaxes(1, 2)
        #
        # recordings_preprocessed = recordings_scaled
        return recordings, labels

    def augmentation(self, recordings, labels):

        recordings_augmented = recordings
        return recordings_augmented, labels

    def shuffle(self, recordings, labels_onehot, labels):
        randnum = random.randint(0, 100)
        random.seed(randnum)
        random.shuffle(recordings)
        random.seed(randnum)
        random.shuffle(labels_onehot)
        random.seed(randnum)
        random.shuffle(labels)

        return recordings, labels_onehot, labels