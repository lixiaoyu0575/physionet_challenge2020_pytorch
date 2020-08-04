import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseEvaluater
from utils import *
import torch.nn.functional as F

class Evaluater(BaseEvaluater):
    """
    Evaluater class
    """
    def __init__(self, model, criterion, metric_ftns, config, test_data_loader):
        super().__init__(model, criterion, metric_ftns, config)
        self.config = config
        self.test_data_loader = test_data_loader
        self.test_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.num_classes = self.config['arch']['args']['num_classes']

    def evaluate_batchwise(self):
        """
        Evaluate after training procedure finished (batch-wised)
        :return: A log that contains information about validation
        """
        self.model.eval()

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_data_loader):
                data, target = data.to(self.device, dtype=torch.float), target.to(self.device, dtype=torch.float)

                output = self.model(data)

                loss = self.criterion(output, target)
                self.test_metrics.update('loss', loss.item())

                for met in self.metric_ftns:
                    self.test_metrics.update(met.__name__, met(self._to_np(output), self._to_np(target)))

        result = self.test_metrics.result()

        # print logged informations to the screen
        for key, value in result.items():
            self.logger.info('    {:15s}: {}'.format(str(key), value))

    def evaluate(self):
        """
        Evaluate after training procedure finished
        :return: A log that contains information about validation
        """

        self.model.eval()

        outputs = torch.zeros((self.test_data_loader.n_samples, self.num_classes))
        targets = torch.zeros((self.test_data_loader.n_samples, self.num_classes))

        with torch.no_grad():
            start = 0
            for batch_idx, (data, target) in enumerate(self.test_data_loader):
                end = len(data) + start
                data, target = data.to(self.device, dtype=torch.float), target.to(self.device, dtype=torch.float)

                output = self.model(data)

                outputs[start:end, :] = output
                targets[start:end, :] = target
                start = end

                loss = self.criterion(output, target)
                self.test_metrics.update('loss', loss.item())

        for met in self.metric_ftns:
            self.test_metrics.update(met.__name__, met(self._to_np(outputs), self._to_np(targets)))

        result = self.test_metrics.result()

        # print logged informations to the screen
        for key, value in result.items():
            self.logger.info('    {:15s}: {}'.format(str(key), value))

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