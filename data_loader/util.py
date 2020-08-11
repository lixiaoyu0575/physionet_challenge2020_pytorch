from scipy.io import loadmat
import numpy as np
import os
import torch
from torch.utils.data import Dataset, TensorDataset
from torchvision import transforms

# Find unique classes.
def get_classes(input_directory, filenames):
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
def load_challenge_data(label_file, data_dir):
    file = os.path.basename(label_file)
    name, ext = os.path.splitext(file)
    with open(label_file, 'r') as f:
        header = f.readlines()
    mat_file = file.replace('.hea', '.mat')
    x = loadmat(os.path.join(data_dir, mat_file))
    recording = np.asarray(x['val'], dtype=np.float64)
    return recording, header, name

# Customed TensorDataset
class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, *tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]
        torch.randn(1)

        if self.transform:
            if torch.rand(1) >= 0.8:
                x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)

class CustomTensorListDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, *tensors_list, transform=None):
        self.tensors_list = tensors_list
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors_list[0][index]

        if self.transform:
            torch.randn(1)
            if torch.rand(1) >= 0.5:
                x = self.transform(x)

        y = self.tensors_list[1][index]

        return x, y

    def __len__(self):
        return len(self.tensors_list[0])

def custom_collate_fn(batch):
    data = [item[0].unsqueeze(0) for item in batch]
    target = [item[1].unsqueeze(0) for item in batch]
    return [data, target]
