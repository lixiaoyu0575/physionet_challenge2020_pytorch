import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker, smooth_one_hot, mixup
import torch.nn.functional as F
from scipy.io import loadmat
import time

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

        if self.do_validation:
            keys_val = ['val_' + k for k in self.keys]
            for key in keys_val:
                self.log[key] = []


        self.only_scored_classes = config['trainer'].get('only_scored_class', True)
        self.lable_smooth = config['trainer'].get('label_smooth', None)
        self.mixup = config['trainer'].get('mixup', None)
        if self.only_scored_classes:
            # Only consider classes that are scored with the Challenge metric.
            indices = loadmat('evaluation/scored_classes_indices.mat')['val']
            self.indices = indices.reshape([indices.shape[1],]).astype(bool)

        self.sigmoid = nn.Sigmoid()

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        start_time = time.time()
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target) in enumerate(self.data_loader):
            batch_start = time.time()
            data, target = data.to(device=self.device, dtype=torch.float), target.to(self.device, dtype=torch.float)
            if self.mixup is not None:
                data, target = mixup(data, target, np.random.beta(1, 1))
            # if self.lable_smooth is not None:
            #     target = target.long()
            #     print('getting smooth label')
            #     target = smooth_one_hot(true_labels=target, classes=108, smoothing=self.lable_smooth)

            self.optimizer.zero_grad()
            output = self.model(data)

            if self.only_scored_classes:
                # Only consider classes that are scored with the Challenge metric.
                loss = self.criterion(output[:, self.indices], target[:, self.indices])
            else:
                loss = self.criterion(output, target)

            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())

            output_logit = self.sigmoid(output)
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(self._to_np(output_logit), self._to_np(target)))

            if batch_idx % self.log_step == 0:
                batch_end = time.time()
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}, 1 batch cost time {:.2f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item(),
                    batch_end - batch_start))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        end_time = time.time()
        print("training epoch cost {} seconds".format(end_time-start_time))
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            if self.config['lr_scheduler']['type'] == 'ReduceLROnPlateau':
                self.lr_scheduler.step(log['val_loss'])
            elif self.config['lr_scheduler']['type'] == 'GradualWarmupScheduler':
                self.lr_scheduler.step(epoch, log['val_loss'])
            else:
                self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(device=self.device, dtype=torch.float), target.to(self.device, dtype=torch.float)

                output = self.model(data)

                if self.only_scored_classes:
                    # Only consider classes that are scored with the Challenge metric.
                    loss = self.criterion(output[:, self.indices], target[:, self.indices])
                else:
                    loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())

                output_logit = self.sigmoid(output)
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(self._to_np(output_logit), self._to_np(target)))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if self.lr_scheduler is not None and self.config["lr_scheduler"]["type"] == "ReduceLROnPlateau":
                self.lr_scheduler.step(self.valid_metrics.result()["challenge_metric"])

        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _to_np(self, tensor):
        if self.device.type == 'cuda':
            return tensor.cpu().detach().numpy()
        else:
            return tensor.detach().numpy()

# For Variable length data
class Trainer2(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

        if self.do_validation:
            keys_val = ['val_' + k for k in self.keys]
            for key in keys_val:
                self.log[key] = []

        self.only_scored_classes = config['trainer'].get('only_scored_class', False)
        if self.only_scored_classes:
            # Only consider classes that are scored with the Challenge metric.
            indices = loadmat('evaluation/scored_classes_indices.mat')['val']
            self.indices = indices.reshape([indices.shape[1], ]).astype(bool)

        self.sigmoid = nn.Sigmoid()

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()

        for batch_idx, (data, target) in enumerate(self.data_loader):

            loss = torch.tensor(0).to(self.device, dtype=torch.float)
            outputs = torch.zeros(len(target), target[0].shape[1])
            targets = torch.zeros(len(target), target[0].shape[1])

            self.optimizer.zero_grad()

            for i in range(len(data)):
                data[i], target[i] = data[i].to(device=self.device, dtype=torch.float), target[i].to(self.device, dtype=torch.float)

                output = self.model(data[i])

                if self.only_scored_classes:
                    # Only consider classes that are scored with the Challenge metric.
                    loss_i = self.criterion(output[:, self.indices], target[i][:, self.indices])
                else:
                    loss_i = self.criterion(output, target[i])
                loss += loss_i
                outputs[i:i+1, :] = output
                targets[i:i+1, :] = target[i]

            loss /= len(data)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())

            output_logit = self.sigmoid(outputs)
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(self._to_np(output_logit), self._to_np(targets)))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            if self.config['lr_scheduler']['type'] == 'ReduceLROnPlateau':
                self.lr_scheduler.step(log['val_loss'])
            elif self.config['lr_scheduler']['type'] == 'GradualWarmupScheduler':
                self.lr_scheduler.step(epoch, log['val_loss'])
            else:
                self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):

                loss = torch.tensor(0).to(self.device, dtype=torch.float)
                outputs = torch.zeros(len(target), target[0].shape[1])
                targets = torch.zeros(len(target), target[0].shape[1])

                for i in range(len(data)):
                    data[i], target[i] = data[i].to(device=self.device, dtype=torch.float), target[i].to(self.device, dtype=torch.float)

                    output = self.model(data[i])

                    if self.only_scored_classes:
                        # Only consider classes that are scored with the Challenge metric.
                        loss_i = self.criterion(output[:, self.indices], target[i][:, self.indices])
                    else:
                        loss_i = self.criterion(output, target[i])
                    loss += loss_i
                    outputs[i:i + 1, :] = output
                    targets[i:i + 1, :] = target[i]

                loss /= len(data)
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())

                output_logit = self.sigmoid(outputs)
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(self._to_np(output_logit), self._to_np(targets)))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _to_np(self, tensor):
        if self.device.type == 'cuda':
            return tensor.cpu().detach().numpy()
        else:
            return tensor.detach().numpy()

def get_pred(output, alpha=0.5):
    output = F.sigmoid(output)
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            if output[i, j] >= alpha:
                output[i, j] = 1
            else:
                output[i, j] = 0
    return output
