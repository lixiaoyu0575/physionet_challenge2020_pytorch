import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataset import Subset
from torch.utils.data import TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, validation_split, test_split, num_workers, collate_fn=default_collate, pin_memory=True, normalization=False):
        self.validation_split = validation_split
        self.test_split = test_split
        self.shuffle = shuffle
        self.dataset =dataset
        self.batch_idx = 0
        self.n_samples = len(self.dataset)

        self.sampler, self.valid_sampler, self.test_subset = self._split_sampler(self.validation_split, self.test_split, normalization)

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers,
            'pin_memory': pin_memory
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

        self.valid_data_loader = self.split_validation()

        self.test_data_loader_init_kwargs = {
            'batch_size': batch_size,
            'shuffle': False,
            'collate_fn': collate_fn,
            'num_workers': num_workers,
            'pin_memory': pin_memory
        }

        self.test_data_loader = self.split_test()

    def _split_sampler(self, validation_split, test_split, normalization):
        if validation_split == 0.0 and test_split == 0.0:
            return None, None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(1)
        np.random.shuffle(idx_full)

        if isinstance(validation_split, int):
            assert validation_split > 0
            assert validation_split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = validation_split
        else:
            len_valid = int(self.n_samples * validation_split)

        if isinstance(test_split, int):
            assert test_split > 0
            assert test_split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_test = test_split
        else:
            len_test = int(self.n_samples * test_split)

        valid_idx = idx_full[0:len_valid]
        test_idx = idx_full[len_valid:len_valid+len_test]
        train_idx = np.delete(idx_full, np.arange(0, len_valid + len_test))

        if normalization:
            self.data_normalization(train_idx, valid_idx, test_idx)

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        # test_sampler = SubsetRandomSampler(test_idx)
        test_subset = Subset(self.dataset, test_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)
        self.valid_n_samples = len(valid_idx)
        self.test_n_samples = len(test_idx)

        return train_sampler, valid_sampler, test_subset

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            valid_data_loader = DataLoader(sampler=self.valid_sampler, **self.init_kwargs)
            valid_data_loader.n_samples = self.valid_n_samples
            return valid_data_loader

    def split_test(self):
        if self.split_test is None:
            return None
        else:
            test_data_loader = DataLoader(self.test_subset, **self.test_data_loader_init_kwargs)
            test_data_loader.n_samples = self.test_n_samples
            return test_data_loader

    # for feature data
    def data_normalization(self, train_idx, val_idx, test_idx):

        X = self.dataset[:][0]
        Y = self.dataset[:][1]

        ## max_min
        # mm = MinMaxScaler()
        # X = mm.fit_transform(X[:, 0, :])
        # X = X.reshape((X.shape[0], 1, X.shape[1]))

        # StandardScaler
        std = StandardScaler()
        X = std.fit_transform(X[:, 0, :])
        X = X.reshape((X.shape[0], 1, X.shape[1]))

        print("after std, nan:")
        print(np.isnan(X).any())

        X = torch.from_numpy(X).float()
        self.dataset = TensorDataset(X, Y)

class BaseDataLoader2(DataLoader):
    """
    Base class for all data loaders
    """

    def __init__(self, dataset, batch_size, shuffle, train_idx, valid_idx, test_idx, num_workers,
                 collate_fn=default_collate, pin_memory=True, normalization=False):
        self.train_idx = train_idx
        self.valid_idx = valid_idx
        self.test_idx = test_idx
        self.shuffle = shuffle
        self.dataset = dataset
        self.batch_idx = 0
        self.n_samples = len(self.dataset)

        self.sampler, self.valid_sampler, self.test_subset = self._split_sampler(self.train_idx, self.valid_idx,
                                                                                  self.test_idx)

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers,
            'pin_memory': pin_memory
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

        self.valid_data_loader = self.split_validation()

        self.test_data_loader_init_kwargs = {
            'batch_size': batch_size,
            'shuffle': False,
            'collate_fn': collate_fn,
            'num_workers': num_workers,
            'pin_memory': pin_memory
        }

        self.test_data_loader = self.split_test()

    def _split_sampler(self, train_idx, valid_idx, test_idx):
        if valid_idx is None and test_idx is None:
            return None, None, None

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        # test_sampler = SubsetRandomSampler(test_idx)
        test_subset = Subset(self.dataset, test_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)
        self.valid_n_samples = len(valid_idx)
        self.test_n_samples = len(test_idx)

        return train_sampler, valid_sampler, test_subset

    def split_validation(self):
        if self.valid_idx is None:
            return None
        else:
            valid_data_loader = DataLoader(sampler=self.valid_sampler, **self.init_kwargs)
            valid_data_loader.n_samples = self.valid_n_samples
            valid_data_loader.idx = self.valid_idx
            return valid_data_loader

    def split_test(self):
        if self.test_idx is None:
            return None
        else:
            test_data_loader = DataLoader(self.test_subset, **self.test_data_loader_init_kwargs)
            test_data_loader.n_samples = self.test_n_samples
            test_data_loader.idx = self.test_idx
            return test_data_loader

class BaseDataLoader3(DataLoader):
    """
    Base class for all data loaders
    """

    def __init__(self, train_dataset, val_dataset, test_dataset,  batch_size, shuffle, num_workers,
                 collate_fn=default_collate, pin_memory=True):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.batch_idx = 0
        self.shuffle = shuffle

        self.init_kwargs = {
            'dataset': self.train_dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers,
            'pin_memory': pin_memory
        }
        super().__init__(**self.init_kwargs)

        self.valid_data_loader_init_kwargs = {
            'dataset': self.val_dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers,
            'pin_memory': pin_memory
        }

        self.valid_data_loader = DataLoader(**self.valid_data_loader_init_kwargs)

        self.test_data_loader_init_kwargs = {
            'dataset': self.test_dataset,
            'batch_size': batch_size,
            'shuffle': False,
            'collate_fn': collate_fn,
            'num_workers': num_workers,
            'pin_memory': pin_memory
        }

        self.test_data_loader = DataLoader(**self.test_data_loader_init_kwargs)

        self.n_samples = len(self.train_dataset)
        self.valid_data_loader.n_samples = len(self.val_dataset)
        self.test_data_loader.n_samples = len(self.test_dataset)