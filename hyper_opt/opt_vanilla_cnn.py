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
from torch.autograd import Variable
import pickle
import traceback
import time
from datetime import datetime
from hyperopt import hp, tpe, fmin, Trials
import numpy as np

from data_loader.data_loaders import ChallengeDataLoader2
from model.metric import ChallengeMetric
import augmentation.transformers as module_transformers
from utils.lr_scheduler import GradualWarmupScheduler

import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch_model
import model.resnet as module_arch_resnet
import model.resnext as module_arch_resnext
import model.inceptiontime as module_arch_inceptiontime
import model.fcn as module_arch_fcn
import model.tcn as module_arch_tcn
import model.resnest as module_arch_resnest
import model.vanilla_cnn as module_arch_vanilla_cnn

from hyper_opt.util import init_obj, to_np, get_mnt_mode, save_checkpoint, \
    write_json, get_logger, analyze, progress, load_checkpoint

from tensorboardX import SummaryWriter

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import random
seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

in_channels = 12
num_classes = 108
label_dir = "/DATASET/challenge2020/new_data/All_data_new"
data_dir = "/DATASET/challenge2020/new_data/All_data_new_resampled_to_300HZ_and_slided_n_segment=1_windowsize=3000"
weights_file = '/home/weiyuhua/physionet_challenge2020_pytorch/evaluation/weights.csv'

# label_dir = "/home/weiyuhua/Data/All_data_new"
# data_dir = "/home/weiyuhua/Data/All_data_new_resampled_to_300HZ_and_slided_n_segment=1_windowsize=3000"
# weights_file = '/home/weiyuhua/Code/physionet_challenge2020_pytorch/evaluation/weights.csv'


save_path = './vanilla_cnn_test'
log_step = 1

space2 = {
    'arch':
        hp.choice('arch', [
            {
                'type': 'VanillaCNN',
                'args':
                    {
                        "in_channels": 12,
                        "num_classes": 108,
                        "channels": hp.choice('channels', [[32, 64, 128, 256, 512]]),
                        "kernel_size": hp.choice('kernel_size', [[5, 5, 5, 5, 5]]),
                        "pool": hp.choice('pool', ['max']),
                        "pool_kernel_size": hp.choice('pool_kernel_size', [2]),
                        "n_hid": hp.choice('n_hid', [[100, 80]]),
                        "drop_rate": hp.choice('drop_rate', [0.2])
                    }
            }
        ]),

    'data_split':
        hp.choice('data_split', ['split1']),

    'data_normalization':
        hp.choice('data_normalization', [False]),

    'only_scored':
        hp.choice('only_scores', [True]),

    'augmentation':
        {
            'method': hp.choice('method', [
                None,
                {
                    "TimeWarp": {
                        "args": {
                            "sigma": 0.2
                        }
                    },
                }
            ]),

            'prob': hp.choice('prob', [0.8])
        },

    'optimizer':
        hp.choice('optimizer', [
            {
                'type': 'Adam',
                'args':
                    {
                        'lr': hp.choice('lr', [0.001]),
                        'weight_decay': hp.choice('weight_decay', [1e-3, 1e-4]),
                        'amsgrad': True
                    }
            }
        ]),

    'loss':
        hp.choice('loss', [
            {
                'type': 'bce_with_logits_loss'
            }
        ]),

    'lr_scheduler':
        hp.choice('lr_scheduler', [
            {
                "type": "ReduceLROnPlateau",
                "args": {
                    "mode": "min",
                    "factor": hp.choice('factor', [0.1, 0.5]),
                    "patience": 6,
                    "verbose": False,
                    "threshold": 0.0001,
                    "threshold_mode": "rel",
                    "cooldown": 0,
                    "min_lr": 0,
                    "eps": 1e-08
                }

            },
            {
                "type": "StepLR",
                "args":
                    {
                        "step_size": hp.choice('step_size', [30, 40]),
                        "gamma": hp.choice('StepLR_gamma', [0.1, 0.5])
                    }
            },

            {
                "type": "GradualWarmupScheduler",
                "args": {
                    "multiplier": hp.choice('multiplier', [1, 1.5]),
                    "total_epoch": hp.choice('total_epoch', [5, 10]),
                    "after_scheduler": {
                        "type": "ReduceLROnPlateau",
                        "args": {
                            "mode": "min",
                            "factor": hp.choice('factor2', [0.1, 0.5]),
                            "patience": 6,
                            "verbose": False,
                            "threshold": 0.0001,
                            "threshold_mode": "rel",
                            "cooldown": 0,
                            "min_lr": 0,
                            "eps": 1e-08
                        }

                    }
                }
            }
        ]),

    'trainer':
        hp.choice('trainer', [
            {
                "epochs": hp.choice('epochs', [100]),
                "batch_size": hp.choice('batch_size', [64, 32, 16]),
                "monitor": hp.choice('monitor', ['min val_loss', 'max val_challenge_metric']),
                'early_stop': hp.choice('early_stop', [10, 15])
            },
        ])
}

space = {
    'arch':
        hp.choice('arch', [
            {
                'type': 'VanillaCNN',
                'args':
                    {
                        "in_channels": 12,
                        "num_classes": 108,
                        "channels": hp.choice('channels', [[32, 64, 128, 256, 512]]),
                        "kernel_size": hp.choice('kernel_size', [[5, 5, 5, 5, 5]]),
                        "pool": hp.choice('pool', ['max']),
                        "pool_kernel_size": hp.choice('pool_kernel_size', [2]),
                        "n_hid": hp.choice('n_hid', [[100, 80]]),
                        "drop_rate": hp.choice('drop_rate', [0.2])
                    }
            }
        ]),

    'data_split':
        hp.choice('data_split', ['split1']),

    'data_normalization':
        hp.choice('data_normalization', [False]),

    'only_scored':
        hp.choice('only_scores', [False]),

    'augmentation':
        {
            'method': hp.choice('method', [
                None,
        ]),

            'prob': hp.choice('prob', [0.8])
        },

    'optimizer':
        hp.choice('optimizer', [
            {
                'type': 'Adam',
                'args':
                    {
                        'lr': hp.choice('lr', [0.001]),
                        'weight_decay': hp.choice('weight_decay', [0]),
                        'amsgrad': True
                    }
            }
        ]),

    'loss':
        hp.choice('loss', [
            {
                'type': 'bce_with_logits_loss'
            }
        ]),

    'lr_scheduler':
        hp.choice('lr_scheduler', [
            {
                "type": "StepLR",
                "args":
                    {
                        "step_size": hp.choice('step_size', [50]),
                        "gamma": hp.choice('StepLR_gamma', [0.1])
                    }
            }
        ]),

    'trainer':
        hp.choice('trainer', [
            {
                "epochs": hp.choice('epochs', [100]),
                "batch_size": hp.choice('batch_size', [128]),
                "monitor": hp.choice('monitor', ['min val_loss']),
                'early_stop': hp.choice('early_stop', [10])
            },
        ])
}

# model selection
files_models = {
    "fcn": ['FCN'],
    "inceptiontime": ['InceptionTimeV1', 'InceptionTimeV2'],
    "mc_inceptiontime": ['MCInceptionTimeV2'],
    "resnet": ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
    "resnext": ['ResNeXt', 'resnext18', 'resnext34', 'resnext50', 'resnext101', 'resnext152'],
    "resnest": ['resnest50', 'resnest'],
    "model": ['CNN', 'MLP'],
    "tcn": ['TCN'],
    "vanilla_cnn": ['VanillaCNN']
}


def train(model, optimizer, train_loader, criterion, metric, indices, epoch, device=None):
    sigmoid = nn.Sigmoid()
    model.train()
    cc = 0
    Loss = 0
    total = 0
    batchs = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        batch_start = time.time()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        if not indices is None:
            loss = criterion(output[:, indices], target[:, indices])
        else:
            loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        c = metric(to_np(sigmoid(output), device), to_np(target, device))
        cc += c
        Loss += loss
        total += target.size(0)
        batchs += 1

        if batch_idx % log_step == 0:
            batch_end = time.time()
            # logger.debug('Epoch: {} {} Loss: {:.6f}, 1 batch cost time {:.2f}'.format(epoch, batch_idx, loss.item(),
            #                                                                           batch_end - batch_start))
            print('Train Epoch: {} {} Loss: {:.6f}, 1 batch cost time {:.2f}'.format(epoch,
                                                                                     progress(train_loader, batch_idx),
                                                                                     loss.item(),
                                                                                     batch_end - batch_start))

    return Loss / total, cc / batchs


def valid(model, valid_loader, criterion, metric, indices, device=None):
    sigmoid = nn.Sigmoid()
    model.eval()
    cc = 0
    Loss = 0
    total = 0
    batchs = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(valid_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            if not indices is None:
                loss = criterion(output[:, indices], target[:, indices])
            else:
                loss = criterion(output, target)
            c = metric(to_np(sigmoid(output), device), to_np(target, device))
            cc += c
            Loss += loss
            total += target.size(0)
            batchs += 1

    return Loss / total, cc / batchs


def test(model, test_loader, criterion, metric, device=None):
    model.eval()
    sigmoid = nn.Sigmoid()
    batchs = 0
    Loss = 0
    total = 0
    outputs = torch.zeros((test_loader.n_samples, model.num_classes))
    targets = torch.zeros((test_loader.n_samples, model.num_classes))

    with torch.no_grad():
        start = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            end = len(data) + start
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            outputs[start:end, :] = output
            targets[start:end, :] = target
            start = end

            Loss += loss
            batchs += 1
            total += target.size(0)

        cc = metric(to_np(sigmoid(outputs), device), to_np(targets, device))

    return Loss / total, cc


def train_challenge2020(hype_space):
    # Paths to save log, checkpoint, tensorboard logs and results
    run_id = datetime.now().strftime(r'%m%d_%H%M%S')
    base_path = save_path + '/' + run_id
    os.makedirs(base_path)
    write_json(hype_space, base_path + '/hype_space.json')

    checkpoint_dir = base_path + '/checkpoints'
    log_dir = base_path + '/log'
    tb_dir = base_path + '/tb_log'
    result_dir = base_path + '/results'

    os.makedirs(result_dir)
    os.makedirs(log_dir)
    os.makedirs(checkpoint_dir)
    os.makedirs(tb_dir)

    # Logger for train
    logger = get_logger(log_dir + '/info.log', name='train' + run_id)
    logger.info(hype_space)

    # Tensorboard
    train_writer = SummaryWriter(tb_dir + '/train')
    val_writer = SummaryWriter(tb_dir + '/valid')


    # Hyper Parameters
    split_index = "../process/data_split/" + hype_space['data_split']

    # Setup Cuda
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Data_loader
    train_loader = ChallengeDataLoader2(label_dir, data_dir, split_index,
                                          batch_size=hype_space['trainer']['batch_size'],
                                          normalization=hype_space['data_normalization'],
                                          augmentations=hype_space['augmentation']['method'],
                                          p=hype_space['augmentation']['prob'])
    valid_loader = train_loader.valid_data_loader
    test_loader = train_loader.test_data_loader

    # Build model architecture
    global model
    for file, types in files_models.items():
        for type in types:
            if hype_space["arch"]["type"] == type:
                model = init_obj(hype_space, 'arch', eval("module_arch_" + file))

    dummy_input = Variable(torch.rand(16, 12, 3000))
    train_writer.add_graph(model, (dummy_input,))

    model.to(device)

    # Get function handles of loss and metrics
    criterion = getattr(module_loss, hype_space['loss']['type'])

    # Get function handles of metrics
    challenge_metrics = ChallengeMetric(label_dir)
    metric = challenge_metrics.challenge_metric

    # Get indices of the scored labels
    if hype_space['only_scored']:
        indices = challenge_metrics.indices
    else:
        indices = None

    # Build optimizer, learning rate scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = init_obj(hype_space, 'optimizer', torch.optim, trainable_params)
    if hype_space['lr_scheduler']['type'] == 'GradualWarmupScheduler':
        params = hype_space["lr_scheduler"]["args"]
        scheduler_steplr_args = dict(params["after_scheduler"]["args"])
        scheduler_steplr = getattr(torch.optim.lr_scheduler, params["after_scheduler"]["type"])(optimizer, **scheduler_steplr_args)
        lr_scheduler = GradualWarmupScheduler(optimizer, multiplier=params["multiplier"],
                                              total_epoch=params["total_epoch"], after_scheduler=scheduler_steplr)
    else:
        lr_scheduler = init_obj(hype_space, 'lr_scheduler', torch.optim.lr_scheduler, optimizer)

    # Begin training process
    trainer = hype_space['trainer']
    epochs = trainer['epochs']

    # Full train and valid logic
    mnt_metric_name, mnt_mode, mnt_best, early_stop = get_mnt_mode(trainer)
    not_improved_count = 0

    for epoch in range(epochs):
        best = False
        train_loss, train_metric = train(model, optimizer, train_loader, criterion, metric, indices, epoch, device=device)
        val_loss, val_metric = valid(model, valid_loader, criterion, metric, indices, device=device)

        if hype_space['lr_scheduler']['type'] == 'ReduceLROnPlateau':
            # if hype_space['lr_scheduler']['args']['mode'] == 'min':
            #     lr_scheduler.step(train_loss)
            # else:
            #     lr_scheduler.step(train_metric)
            lr_scheduler.step(val_loss)
        elif hype_space['lr_scheduler']['type'] == 'GradualWarmupScheduler':
            lr_scheduler.step(epoch, val_loss)
        else:
            lr_scheduler.step()

        logger.info(
            'Epoch:[{}/{}]\t {:10s}: {:.5f}\t {:10s}: {:.5f}'.format(epoch, epochs, 'loss', train_loss, 'metric',
                                                                     train_metric))
        logger.info(
            '             \t {:10s}: {:.5f}\t {:10s}: {:.5f}'.format('val_loss', val_loss, 'val_metric', val_metric))
        logger.info('             \t learning_rate: {}'.format(optimizer.param_groups[0]['lr']))

        # check whether model performance improved or not, according to specified metric(mnt_metric)
        if mnt_mode != 'off':
            mnt_metric = val_loss if mnt_metric_name == 'val_loss' else val_metric
            improved = (mnt_mode == 'min' and mnt_metric <= mnt_best) or \
                       (mnt_mode == 'max' and mnt_metric >= mnt_best)
            if improved:
                mnt_best = mnt_metric
                not_improved_count = 0
                best = True
            else:
                not_improved_count += 1

            if not_improved_count > early_stop:
                logger.info("Validation performance didn\'t improve for {} epochs. Training stops.".format(early_stop))
                break

        if best == True:
            save_checkpoint(model, epoch, optimizer, mnt_best, hype_space, checkpoint_dir, save_best=True)
            logger.info("Saving current best: model_best.pth ...")

        # Tensorboard log
        train_writer.add_scalar('loss', train_loss, epoch)
        train_writer.add_scalar('metric', train_metric, epoch)
        train_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        val_writer.add_scalar('loss', val_loss, epoch)
        val_writer.add_scalar('metric', val_metric, epoch)

    # Logger for test
    logger = get_logger(result_dir + '/info.log', name='test' + run_id)
    logger.propagate = False

    # Load model_best checkpoint
    model = load_checkpoint(model, checkpoint_dir + '/model_best.pth', logger)

    # Testing
    test_loss, test_metric = test(model, test_loader, criterion, metric, device=device)
    logger.info('    {:10s}: {:.5f}\t {:10s}: {:.5f}'.format('loss', test_loss, 'metric', test_metric))

    challenge_metrics.return_metric_list()
    analyze(model, test_loader, criterion, challenge_metrics, logger, result_dir, device=device)

    write_json(hype_space, '{}/{}_{:.5f}.json'.format(save_path, run_id, test_metric))

    return -test_metric


def run_trials():
    """Run one TPE meta optimisation step and save its results."""
    max_evals = nb_evals = 1

    logger = get_logger(save_path + '/trials.log', name='trials')

    logger.info("Attempt to resume a past training if it exists:")

    try:
        # https://github.com/hyperopt/hyperopt/issues/267
        trials = pickle.load(open(save_path + "/results.pkl", "rb"))
        logger.info("Found saved Trials! Loading...")
        max_evals = len(trials.trials) + nb_evals
        logger.info("Rerunning from {} trials to add another one.".format(
            len(trials.trials)))
    except:
        trials = Trials()
        logger.info("Starting from scratch: new trials.")

    best = fmin(
        train_challenge2020,
        space,
        algo=tpe.suggest,
        trials=trials,
        max_evals=max_evals,
    )

    logger.info("Best: {}".format(best))
    pickle.dump(trials, open(save_path + "/results.pkl", "wb"))
    logger.info("\nOPTIMIZATION STEP COMPLETE.\n")
    logger.info("Trials:")

    for trial in trials:
        logger.info(trial)


if __name__ == "__main__":
    try:
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        run_trials()
    except Exception as err:
        err_str = str(err)
        print(err_str)
        traceback_str = str(traceback.format_exc())
        print(traceback_str)
