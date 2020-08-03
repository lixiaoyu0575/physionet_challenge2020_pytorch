import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch_model
import model.resnet as module_arch_resnet
import model.resnext as module_arch_resnext
import model.inceptiontime as module_arch_inceptiontime
import model.fcn as module_arch_fcn
import model.tcn as module_arch_tcn
import model.resnest as module_arch_resnest
from parse_config import ConfigParser
from trainer import Trainer
from evaluater import Evaluater
from model.metric import ChallengeMetric, ChallengeMetric2
from utils.dataset import load_label_files, load_labels, load_weights

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
    "resnet": ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
    "resnext": ['ResNeXt', 'resnext18', 'resnext34', 'resnext50', 'resnext101', 'resnext152'],
    "resnest": ['resnest50', 'resnest'],
    "model": ['CNN', 'MLP'],
    "tcn": ['TCN']
}

def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.valid_data_loader
    test_data_loader = data_loader.test_data_loader

    # build model architecture, then print to console
    global model
    for file, types in files_models.items():
        for type in types:
            if config["arch"]["type"] == type:
                model = config.init_obj('arch', eval("module_arch_" + file))
                logger.info(model)

    # get function handles of loss and metrics
    if config['loss']['type'] == 'FocalLoss2d':
        count = data_loader.count
        indices = data_loader.indices
        w =  np.max(count[indices]) / count
        w[indices] = 0

        only_scored_classes = config['trainer'].get('only_scored_class', False)
        if only_scored_classes:
            w = w[indices]

        weight = config['loss'].get('args', w)
        criterion = getattr(module_loss, 'FocalLoss2d')(weight=weight)
    else:
        criterion = getattr(module_loss, config['loss']['type'])

    # get function handles of metrics

    challenge_metrics = ChallengeMetric(config['data_loader']['args']['label_dir'])
    # challenge_metrics = ChallengeMetric2(num_classes=9)

    metrics = [getattr(challenge_metrics, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()

    evaluater = Evaluater(model, criterion, metrics,
                          config=config,
                          test_data_loader=test_data_loader)

    evaluater.evaluate()

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)

    import os
    print(torch.cuda.device_count())
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    print(torch.cuda.device_count())
    main(config)
