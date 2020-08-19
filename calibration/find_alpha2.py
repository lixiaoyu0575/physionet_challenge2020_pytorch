import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
print(curPath)
rootPath = curPath
for i in range(1):
    rootPath = os.path.split(rootPath)[0]
print(rootPath)
sys.path.append(rootPath)
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch_model
import model.resnet as module_arch_resnet
import model.resnext as module_arch_resnext
import model.inceptiontime as module_arch_inceptiontime
import model.mc_inceptiontime as module_arch_mc_inceptiontime
import model.fcn as module_arch_fcn
import model.tcn as module_arch_tcn
import model.resnest as module_arch_resnest
import model.resnest2 as module_arch_resnest2
import model.vanilla_cnn as module_arch_vanilla_cnn
import model.xception as module_arch_xception
from hyperopt import hp, tpe, fmin, Trials
from alpha.util import *

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

# model selection
files_models = {
    "fcn": ['FCN'],
    "inceptiontime": ['InceptionTimeV1', 'InceptionTimeV2'],
    "mc_inceptiontime": ['MCInceptionTimeV2'],
    "resnet": ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
    "resnext": ['ResNeXt', 'resnext18', 'resnext34', 'resnext50', 'resnext101', 'resnext152'],
    "resnest": ['resnest50', 'resnest'],
    "resnest2": ['resnest2'],
    "model": ['CNN', 'MLP'],
    "tcn": ['TCN'],
    "vanilla_cnn": ['VanillaCNN'],
    "xception": ['Xception', 'Gception']
}

arch = {
    "type": "resnest",
    "args": {
        "layers": [2, 2, 1, 3],
        "bottleneck_width": 64,
        "stem_width": 16,
        "num_classes": 108,
        "kernel_size": 7
    }
}

label_dir = "/DATASET/challenge2020/new_data/All_data_new"
data_dir = "/DATASET/challenge2020/new_data/All_data_new_resampled_to_300HZ_and_slided_n_segment=1_windowsize=3000"
split_index = "../process/data_split/split1.mat"
model_dir = "/home/weiyuhua/physionet_challenge2020_pytorch/saved/Resnest/model_resnest_0.571/model_best.pth"

space = {
    "alpha1": hp.quniform("alpha1", 0, 1, 0.1),
    "alpha2": hp.quniform("alpha2", 0, 1, 0.1),
    "alpha3": hp.quniform("alpha3", 0, 1, 0.1),
    "alpha4": hp.quniform("alpha4", 0, 1, 0.1),
    "alpha5": hp.quniform("alpha5", 0, 1, 0.1),
    "alpha6": hp.quniform("alpha6", 0, 1, 0.1),
    "alpha7": hp.quniform("alpha7", 0, 1, 0.1),
    "alpha8": hp.quniform("alpha8", 0, 1, 0.1),
    "alpha9": hp.quniform("alpha9", 0, 1, 0.1),
    "alpha10": hp.quniform("alpha10", 0, 1, 0.1),
    "alpha11": hp.quniform("alpha11", 0, 1, 0.1),
    "alpha12": hp.quniform("alpha12", 0, 1, 0.1),
    "alpha13": hp.quniform("alpha13", 0, 1, 0.1),
    "alpha14": hp.quniform("alpha14", 0, 1, 0.1),
    "alpha15": hp.quniform("alpha15", 0, 1, 0.1),
    "alpha16": hp.quniform("alpha16", 0, 1, 0.1),
    "alpha17": hp.quniform("alpha17", 0, 1, 0.1),
    "alpha18": hp.quniform("alpha18", 0, 1, 0.1),
    "alpha19": hp.quniform("alpha19", 0, 1, 0.1),
    "alpha20": hp.quniform("alpha20", 0, 1, 0.1),
    "alpha21": hp.quniform("alpha21", 0, 1, 0.1),
    "alpha22": hp.quniform("alpha22", 0, 1, 0.1),
    "alpha23": hp.quniform("alpha23", 0, 1, 0.1),
    "alpha24": hp.quniform("alpha24", 0, 1, 0.1),
}

# Data
(train_data, train_label), (val_data, val_label), (test_data, test_label) = load_data(label_dir, data_dir, split_index)
# Dataset
val_dataset = TensorDataset(torch.from_numpy(val_data))
test_dataset = TensorDataset(torch.from_numpy(test_data))
# Dataloader
batch_size = 32
val_dataloader = DataLoader(val_dataset, batch_size)
test_dataloader = DataLoader(test_dataset, batch_size)

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for file, types in files_models.items():
    for type in types:
        if arch['type'] == type:
            model = getattr(eval("module_arch_" + file), arch['type'])(**arch['args'])
checkpoint = load_checkpoint(model_dir)
model.load_state_dict(checkpoint['state_dict'])
model.to(device)

# Inference
model.eval()
with torch.no_grad():
    outputs = torch.zeros((len(val_dataset), 108))
    start = 0
    for i, [data] in enumerate(val_dataloader):
        data = data.to(device=device, dtype=torch.float32)
        output = nn.Sigmoid()(model(data))
        end = len(data) + start
        outputs[start:end, :] = output
        start = end

def find_alpha(hp):
    alphas = [hp['alpha1'], hp['alpha2'], hp['alpha3'], hp['alpha4'], hp['alpha5'], hp['alpha6'], hp['alpha7'],
              hp['alpha8'],
              hp['alpha9'], hp['alpha10'], hp['alpha11'], hp['alpha12'], hp['alpha13'], hp['alpha14'], hp['alpha15'],
              hp['alpha16'],
              hp['alpha17'], hp['alpha18'], hp['alpha19'], hp['alpha20'], hp['alpha21'], hp['alpha22'], hp['alpha23'],
              hp['alpha24']]

    challenge_metrics = ChallengeMetric(label_dir, alphas)

    accuracy, macro_f_measure, macro_f_beta_measure, macro_g_beta_measure, challenge_metric = get_metrics(
        to_np(outputs, device), val_label, challenge_metrics=challenge_metrics)

    return -challenge_metric


if __name__ == '__main__':
    trials = Trials()
    max_evals = 500
    best = fmin(
        find_alpha,
        space,
        algo=tpe.suggest,
        trials=trials,
        max_evals=max_evals,
    )
    print("BEST:", best)
    for trial in trials:
        print(trial)

    best_alphas = list()
    for (k, v) in dict.items(best):
        best_alphas.append(v)

    # For testing
    model.eval()
    with torch.no_grad():
        outputs = torch.zeros((len(test_dataset), 108))
        start = 0
        for i, [data] in enumerate(test_dataloader):
            data = data.to(device=device, dtype=torch.float32)
            end = start + len(data)
            output = nn.Sigmoid()(model(data))
            outputs[start:end, :] = output
            start = end

    challenge_metrics = ChallengeMetric(label_dir, best_alphas)
    accuracy, macro_f_measure, macro_f_beta_measure, macro_g_beta_measure, challenge_metric = get_metrics(
        to_np(outputs, device), test_label, challenge_metrics=challenge_metrics)

    print("Testing data:")
    print("accuracy", accuracy)
    print("macro_f_measure", macro_f_measure)
    print("macro_g_beta_measure", macro_g_beta_measure)
    print("macro_f_beta_measure", macro_f_beta_measure)
    print("challenge_metric", challenge_metric)
