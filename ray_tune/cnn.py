import os
import sys
import numpy as np
import argparse
from filelock import FileLock
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler

from model.model import CNN
from util import ChallengeData, ChallengeMetric
from scipy.io import loadmat, savemat


# Change these values if you want the training to run quicker or slower.

in_channels = 12
num_classes = 108
label_dir = "/DATASET/challenge2020/All_data"
data_dir = "/DATASET/challenge2020/All_data_resampled_to_300HZ_and_slided_n_segment=1"
weights_file = '/home/weiyuhua/physionet_challenge2020_pytorch/evaluation/weights.csv'
split_index = "/home/weiyuhua/physionet_challenge2020_pytorch/process/data_split/split1.mat"
batch_size = 64

challenge_metrics = ChallengeMetric(label_dir, weights_file)
challenge_metric = challenge_metrics.challenge_metric

print(os.getcwd())
def train(model, optimizer, lr_scheduler, train_loader, criterion, device=None):
    device = device or torch.device("cpu")
    sigmoid = nn.Sigmoid()
    model.train()
    cc = 0
    Loss = 0
    total = 0
    batchs = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        # if batch_idx * len(data) > EPOCH_SIZE:
        #     return
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # print(to_np(sigmoid(output), device))
        # print(to_np(target, device))

        c = challenge_metrics.challenge_metric(to_np(sigmoid(output), device), to_np(target, device))
        cc += c
        Loss += loss
        total += target.size(0)
        batchs += 1

    lr_scheduler.step()
    return Loss / total, cc / batchs


def test(model, data_loader, device=None):
    device = device or torch.device("cpu")
    model.eval()
    sigmoid = nn.Sigmoid()
    cc = 0
    batchs = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            # if batch_idx * len(data) > TEST_SIZE:
            #     break
            data, target = data.to(device), target.to(device)
            outputs = model(data)

            # print(to_np(sigmoid(outputs),device))
            # print(to_np(target, device))

            c = challenge_metrics.challenge_metric(to_np(sigmoid(outputs),device), to_np(target, device))
            cc += c
            batchs += 1
    return cc / batchs

def to_np(tensor, device):
    if device.type == 'cuda':
        return tensor.cpu().detach().numpy()
    else:
        return tensor.detach().numpy()

def get_data_loaders():
    dataset, indices = ChallengeData(label_dir, data_dir, weights_file)
    split_idx = loadmat(split_index)
    train_index, val_index, test_index = split_idx['train_index'], split_idx['val_index'], split_idx['test_index']
    train_index = train_index.reshape((train_index.shape[1],))
    val_index = val_index.reshape((val_index.shape[1],))
    test_index = test_index.reshape((test_index.shape[1],))

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(train_index),
        )

    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(np.append(val_index, test_index)),
        )

    return train_loader, test_loader


def train_challenge2020(config):
    use_cuda = config.get("use_gpu") and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    train_loader, test_loader = get_data_loaders()

    model = CNN(in_channels, num_classes).to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, "min")

    scheduler =  torch.optim.lr_scheduler.StepLR( optimizer, step_size= 50,gamma= 0.1)

    criterion = nn.BCEWithLogitsLoss()
    while True:
        train(model, optimizer, scheduler, train_loader, criterion, device)
        cc = test(model, test_loader, device)
        tune.report(mean_challenge_metric=cc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Challenge2020 CNN")
    parser.add_argument(
        "--cuda",
        action="store_true",
        default=True,
        help="Enables GPU training")
    parser.add_argument(
        "--smoke-test", default=True, action="store_true", help="Finish quickly for testing")
    parser.add_argument(
        "--ray-address",
        help="Address of Ray cluster for seamless distributed execution.")
    args = parser.parse_args()
    if args.ray_address:
        ray.init(address=args.ray_address)
    else:
        ray.init(num_cpus=2 if args.smoke_test else None)
    sched = AsyncHyperBandScheduler(
        time_attr="training_iteration",
        metric="mean_challenge_metric",
        mode="max",
        max_t=200,
        grace_period=20
    )
    analysis = tune.run(
        train_challenge2020,
        name="Exp_Challenge2020_CNN",
        scheduler=sched,
        stop={
            "mean_challenge_metric": 0.7,
            "training_iteration": 5 if args.smoke_test else 150
        },
        resources_per_trial={
            "cpu": 2,
            "gpu": int(args.cuda)
        },
        num_samples=1 if args.smoke_test else 2,
        config={
            "lr": tune.sample_from(lambda spec: 10**(-10 * np.random.rand())),
            "weight_decay": tune.uniform(1e-6, 1e-4),
            "use_gpu": int(args.cuda),

        })

    print("Best config is:", analysis.get_best_config(metric="mean_challenge_metric"))
