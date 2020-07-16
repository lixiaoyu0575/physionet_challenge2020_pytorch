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
