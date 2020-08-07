import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import matplotlib
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self._keys = keys
        self._log = self.create_log()
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        # if self.writer is not None:
        #     self.writer.add_scalar(key, value)
        # self._data.total[key] += value * n
        self._data.total[key] += value
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def add(self, key, value):
        self._data.total[key] = value
        self._data.counts[key] = 1
        self._data.average[key] = value

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

    def create_log(self):
        log_dict = {}
        for key in self._keys:
            log_dict[key] = []
        return log_dict

    def update_log(self, key, value):
        self._log[key].append(value)

    def log(self):
        for key in self._keys:
            key_list = self._log[key]
            self._log[key] = np.array(key_list)
        return self._log

def plot_metric(metric_train, metric_val, metric, save_path):
    textsize = 15
    marker = 5
    plt.figure()
    fig, ax1 = plt.subplots()
    ax1.plot(metric_train, 'r')
    ax1.plot(metric_val, 'b')
    ax1.set_ylabel(metric)
    plt.xlabel('epoch')
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='k', linestyle='--')
    if metric_val is not None:
        lgd = plt.legend(['train', 'val'], markerscale=marker, prop={'size': textsize, 'weight': 'normal'})
    else:
        lgd = plt.legend(['train'], markerscale=marker, prop={'size': textsize, 'weight': 'normal'})
    ax = plt.gca()
    plt.title(metric)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(textsize)
        item.set_weight('normal')
    plt.savefig(save_path + '/' + metric + '.png', bbox_extra_artists=(lgd,), bbox_inches='tight')

def plot_lr(lr, save_path):
    textsize = 15
    marker = 5
    plt.figure()
    fig, ax1 = plt.subplots()
    ax1.plot(lr, 'r')
    ax1.set_ylabel('lr')
    plt.xlabel('epoch')
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='k', linestyle='--')
    ax = plt.gca()
    plt.title('lr')
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(textsize)
        item.set_weight('normal')
    plt.savefig(save_path + '/lr.png')

#https://github.com/pytorch/pytorch/issues/7455
def smooth_one_hot(true_labels: torch.Tensor, classes: int, smoothing=0.1):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method

    """
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    print(true_labels.size())
    label_shape = torch.Size((true_labels.size(0), classes))
    print(label_shape)
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape, device=true_labels.device)
        true_dist.fill_(smoothing / (classes - 1))
        index1, index2 = torch.nonzero(true_labels, as_tuple=True)
        index1 = index1.cpu().detach().numpy()
        index2 = index2.cpu().detach().numpy()
        label_index = []
        count = 0
        tmp = []
        for i in range(len(index1)):
            if count == index1[i]:
                tmp.append(index2[i])
            else:
                count += 1
                label_index.append(tmp)
                tmp =[]
                tmp.append(index2[i])
        label_index.append(tmp)
        # label_index = torch.tensor(label_index)
        # label_index = torch.from_numpy(np.array(label_index, dtype='float32'))
        # index2 = true_labels.data.unsqueeze(1)
        index = torch.tensor([[1, 2, -1],[0, 1, 2]])
        test = true_dist.scatter_(1, index=index, value=confidence)
    return true_dist

def smooth_labels(y, smooth_factor=0.1):
    assert len(y.shape) == 2
    assert 0 <= smooth_factor < 1
    y *= 1 - smooth_factor
    y += smooth_factor / y.shape[1]
    return y

def load_model(model, ckpth):
    checkpoint = torch.load(ckpth)
    state_dict = checkpoint['model']
    own_state = model.state_dict()
    new_state_dict = {}
    layers2del = ["head.0.weight", "head.0.bias", "head.2.weight", "head.2.bias", "fc_layer.0.weight", "fc_layer.0.bias", "fc_layer.1.weight", "fc_layer.1.bias", "fc_layer.3.weight", "fc_layer.3.bias", "fc_layer.4.weight", "fc_layer.4.bias", "fc.weight", "fc.bias"]
    for k, v in state_dict.items():
        # print(k)
        k = k.replace("encoder.module.", "")
        if k in layers2del:
            continue
        new_state_dict[k] = v
    new_state_dict["fc.weight"] = own_state["fc.weight"]
    new_state_dict["fc.bias"] = own_state["fc.bias"]
    state_dict = new_state_dict
    model.load_state_dict(state_dict)
    return model

def mixup(input, target, gamma):
    perm = torch.randperm(input.size(0))
    perm_input = input[perm]
    perm_target = target[perm]
    return input.mul_(gamma).add_(1-gamma, perm_input), target.mul_(gamma).add_(1-gamma, perm_target)

if __name__ == '__main__':
    # label = torch.zeros(128, 108)

    # label = torch.tensor([[0, 1, 1], [1, 1, 1]])
    # smth_label = smooth_one_hot(label, 3, 0.1)
    # print('done')

    label = np.ones((128, 108))
    smth_label = smooth_labels(label)
    print('done')
