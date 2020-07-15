import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseEvaluater
from utils import *


class Evaluater(BaseEvaluater):
    """
    Evaluater class
    """
    def __init__(self, model, criterion, metric_ftns, config, test_data_loader):
        super().__init__(model, criterion, metric_ftns, config)
        self.config = config
        self.test_data_loader = test_data_loader
        self.test_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])

    def evaluate(self):
        """
        Evaluate after training procedure finished

        :return: A log that contains information about validation
        """
        self.model.eval()

        Outputs = torch.zeros(self.test_data_loader.n_samples, self.model.num_classes).to(self.device)
        targets = torch.zeros(self.test_data_loader.n_samples)

        with torch.no_grad():
            start = 0
            for batch_idx, (data, target) in enumerate(self.test_data_loader):
                end = len(data) + start
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                Outputs[start:end, :] = output
                targets[start:end] = target
                start = end

                loss = self.criterion(output, target)
                self.test_metrics.update('loss', loss.item())

                for met in self.metric_ftns:
                    self.test_metrics.update(met.__name__, met(output, target, type="Normal"))

        result = self.test_metrics.result()

        # print logged informations to the screen
        for key, value in result.items():
            self.logger.info('    {:15s}: {}'.format(str(key), value))



