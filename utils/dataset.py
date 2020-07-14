import torch
from torch.utils.data.dataset import Dataset
import os
import pickle as dill

def loaddata(data_path):
    ##TODO
    #further modification
    # data_path = '/data/weiyuhua/data/Challenge2018_500hz/preprocessed_data_new/'
    print("Loading data training set")
    with open(os.path.join(data_path, 'data_aug_train.pkl'), 'rb') as fin:
        res = dill.load(fin)
    x_train = res['trainset']
    y_train = res['traintarget']

    with open(os.path.join(data_path, 'data_aug_val.pkl'), 'rb') as fin:
        res = dill.load(fin)
    x_val = res['val_set']
    y_val = res['val_target']

    with open(os.path.join(data_path, 'data_aug_test.pkl'), 'rb') as fin:
        res = dill.load(fin)
    x_test = res['test_set']
    y_test = res['test_target']

    # x_train = x_train.swapaxes(1,2)
    # x_val = x_val.swapaxes(1,2)
    # x_test = x_test.swapaxes(1,2)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

class ECGDataset(Dataset):
    '''Challenge 2017'''
    def __init__(self, data_path):
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = loaddata(data_path)
        self.x, self.y = x_train, y_train

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = (self.x[idx], self.y[idx])

        return sample