import os
import numpy as np
from scipy.io import loadmat
import torch
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset
from base import BaseDataLoader
from utils.dataset import ECGDataset
from sklearn.preprocessing import MinMaxScaler
from utils.dataset import load_label_files, load_labels

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
            recording, header = self.load_challenge_data(label_files[i])

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

    # Find unique classes.
    def get_classes(self, input_directory, filenames):
        classes = set()
        for filename in filenames:
            with open(filename, 'r') as f:
                for l in f:
                    if l.startswith('#Dx'):
                        tmp = l.split(': ')[1].split(',')
                        for c in tmp:
                            classes.add(c.strip())
        return sorted(classes)

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
        recordings_scaled = recordings.swapaxes(1, 2)

        recordings_preprocessed = recordings_scaled
        return recordings_preprocessed, labels

    def augmentation(self, recordings, labels):

        recordings_augmented = recordings
        return recordings_augmented, labels