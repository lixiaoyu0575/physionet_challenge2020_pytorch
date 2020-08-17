import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseEvaluater
from utils import *
import torch.nn.functional as F
import pandas as pd
import torch.nn as nn

class Evaluater(BaseEvaluater):
    """
    Evaluater class
    """
    def __init__(self, model, criterion, metric_ftns, config, test_data_loader, rule_based_ftns=None, checkpoint_dir=None, result_dir=None):
        super().__init__(model, criterion, metric_ftns, config, checkpoint_dir, result_dir)
        self.config = config
        self.test_data_loader = test_data_loader
        self.test_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.rule_based_ftnsm = rule_based_ftns
        self.num_classes = self.config['arch']['args']['num_classes']
        self.sigmoid = nn.Sigmoid()

    def evaluate_batchwise(self):
        """
        Evaluate after training procedure finished (batch-wised)
        """
        self.model.eval()

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_data_loader):
                data, target = data.to(self.device, dtype=torch.float), target.to(self.device, dtype=torch.float)

                output = self.model(data)

                loss = self.criterion(output, target)
                self.test_metrics.update('loss', loss.item())

                output_logit = self.sigmoid(output)
                for met in self.metric_ftns:
                    self.test_metrics.update(met.__name__, met(self._to_np(output_logit), self._to_np(target)))

        result = self.test_metrics.result()

        # print logged informations to the screen
        for key, value in result.items():
            self.logger.info('    {:15s}: {}'.format(str(key), value))

    def evaluate(self):
        """
        Evaluate after training procedure finished
        """

        self.model.eval()

        outputs = torch.zeros((self.test_data_loader.n_samples, self.num_classes))
        targets = torch.zeros((self.test_data_loader.n_samples, self.num_classes))
        inputs = torch.zeros((self.test_data_loader.n_samples, 12, 3000))

        with torch.no_grad():
            start = 0
            for batch_idx, (data, target) in enumerate(self.test_data_loader):
                end = len(data) + start
                data, target = data.to(self.device, dtype=torch.float), target.to(self.device, dtype=torch.float)

                output = self.model(data)

                outputs[start:end, :] = output
                targets[start:end, :] = target
                inputs[start:end, :] = data
                start = end

            loss = self.criterion(outputs, targets)
            self.test_metrics.update('loss', loss.item())

        output_logits = self.sigmoid(outputs)
        output_logits = self._to_np(output_logits)
        targets = self._to_np(targets)

        # rule-based
        if self.rule_based_ftnsm:
            for i, fn_rb in enumerate(self.rule_based_ftnsm):
                output_logits[:, self.test_data_loader.index_rb[i]] = fn_rb(self._to_np(inputs))
    
        for met in self.metric_ftns:
            self.test_metrics.update(met.__name__, met(output_logits, targets))

        result = self.test_metrics.result()

        # print logged informations to the screen
        for key, value in result.items():
            self.logger.info('    {:15s}: {}'.format(str(key), value))

    def analyze(self, challenge_metrics):
        """
        Analyze after training procedure finished
        """
        self.model.eval()

        outputs = torch.zeros((self.test_data_loader.n_samples, self.num_classes))
        targets = torch.zeros((self.test_data_loader.n_samples, self.num_classes))
        inputs = torch.zeros((self.test_data_loader.n_samples, 12, 3000))
        
        with torch.no_grad():
            start = 0
            for batch_idx, (data, target) in enumerate(self.test_data_loader):
                end = len(data) + start
                data, target = data.to(self.device, dtype=torch.float), target.to(self.device, dtype=torch.float)

                output = self.model(data)

                outputs[start:end, :] = output
                targets[start:end, :] = target
                inputs[start:end, :] = data
                start = end

                loss = self.criterion(output, target)
                self.test_metrics.update('loss', loss.item())

        classes = challenge_metrics.classes
        test_idx = self.test_data_loader.idx
        file_names = self.test_data_loader.file_names

        dataset_idx_list = [[] for i in range(6)]
        for i in range(len(test_idx)):
            if file_names[test_idx[i]].startswith('A'):      # CPSC 2018 data, 6,877 recordings
                dataset_idx_list[0].append(i)
            elif file_names[test_idx[i]].startswith('Q'):     # Unused CPSC2018 data, 3,453 recordings
                dataset_idx_list[1].append(i)
            elif file_names[test_idx[i]].startswith('S'):     # PTB Diagnostic ECG Database, 516 recordings
                dataset_idx_list[2].append(i)
            elif file_names[test_idx[i]].startswith('HR'):     # PTB-XL electrocardiography Database, 21,837 recordings
                dataset_idx_list[3].append(i)
            elif file_names[test_idx[i]].startswith('E'):     # Georgia 12-Lead ECG Challenge Database, 10,344 recordings
                dataset_idx_list[4].append(i)
            else:
                dataset_idx_list[5].append(i)       # St Petersburg INCART 12-lead Arrhythmia Database, 74 recordings

        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)

        # All Testing Data
        self.logger.info("**********************************************************************************")
        self.logger.info("All Testing Data")

        outputs_logits = self.sigmoid(outputs)
        outputs_logits = self._to_np(outputs_logits)
        targets = self._to_np(targets)

        # rule-based
        if self.rule_based_ftnsm:
            for i, fn_rb in enumerate(self.rule_based_ftnsm):
                outputs_logits[:, self.test_data_loader.index_rb[i]] = fn_rb(self._to_np(inputs))

        accuracy = challenge_metrics.accuracy(outputs_logits, targets)
        macro_f_measure, f_measure  = challenge_metrics.f_measure(outputs_logits, targets)
        macro_f_beta_measure, macro_g_beta_measure, f_beta_measure, g_beta_measure = challenge_metrics.beta_measures(outputs_logits, targets)
        macro_auroc, macro_auprc, auroc, auprc = challenge_metrics.auc(outputs_logits, targets)
        challenge_metric = challenge_metrics.challenge_metric(outputs_logits, targets)

        self.logger.info("challenge_metric:{}".format(challenge_metric))
        self.logger.info("accuracy:{}".format(accuracy))
        self.logger.info("macro_f_measure:{}".format(macro_f_measure))
        self.logger.info("macro_g_beta_measure:{}".format(macro_g_beta_measure))
        self.logger.info("macro_auroc:{}".format(macro_auroc))
        self.logger.info("macro_auprc:{}".format(macro_auprc))

        metrics = np.vstack((f_measure, f_beta_measure, g_beta_measure, auroc, auprc))
        metrics_df = pd.DataFrame(data=metrics, columns=classes, index=['f_measure', 'f_beta_measure', 'g_beta_measure', 'auroc', 'auprc'])
        self.logger.info(metrics_df)
        metrics_df.to_csv(str(self.result_dir)+'/All.csv')

        Dataset = ['CPSC 2018 data', 'Unused CPSC2018 data', 'PTB Diagnostic ECG Database', 'PTB-XL electrocardiography Database',
                   'Georgia 12-Lead ECG Challenge Database', 'St Petersburg INCART 12-lead Arrhythmia Database']
        
        for i in range(len(Dataset)):
            outputs_i_logit = outputs_logits[dataset_idx_list[i]]
            targets_i = targets[dataset_idx_list[i]]

            accuracy = challenge_metrics.accuracy(outputs_i_logit, targets_i)
            macro_f_measure, f_measure = challenge_metrics.f_measure(outputs_i_logit, targets_i)
            macro_f_beta_measure, macro_g_beta_measure, f_beta_measure, g_beta_measure = challenge_metrics.beta_measures(
                outputs_i_logit, targets_i)
            macro_auroc, macro_auprc, auroc, auprc = challenge_metrics.auc(outputs_i_logit, targets_i)
            challenge_metric = challenge_metrics.challenge_metric(outputs_i_logit, targets_i)

            self.logger.info("**********************************************************************************")
            self.logger.info(Dataset[i])
            self.logger.info("challenge_metric:{}".format(challenge_metric))
            self.logger.info("accuracy:{}".format(accuracy))
            self.logger.info("macro_f_measure:{}".format(macro_f_measure))
            self.logger.info("macro_g_beta_measure:{}".format(macro_g_beta_measure))
            self.logger.info("macro_auroc:{}".format(macro_auroc))
            self.logger.info("macro_auprc:{}".format(macro_auprc))

            metrics = np.vstack((f_measure, f_beta_measure, g_beta_measure, auroc, auprc))
            metrics_df = pd.DataFrame(data=metrics, columns=classes,
                                      index=['f_measure', 'f_beta_measure', 'g_beta_measure', 'auroc', 'auprc'])
            self.logger.info(metrics_df)
            metrics_df.to_csv(str(self.result_dir) + '/' + Dataset[i] + '.csv')

    def _to_np(self, tensor):
        if self.device.type == 'cuda':
            return tensor.cpu().detach().numpy()
        else:
            return tensor.detach().numpy()

# For Variable length data
class Evaluater2(BaseEvaluater):
    """
    Evaluater class
    """

    def __init__(self, model, criterion, metric_ftns, config, test_data_loader, checkpoint_dir=None, result_dir=None):
        super().__init__(model, criterion, metric_ftns, config, checkpoint_dir, result_dir)
        self.config = config
        self.test_data_loader = test_data_loader
        self.test_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.num_classes = self.config['arch']['args']['num_classes']
        self.sigmoid = nn.Sigmoid()

    def evaluate_batchwise(self):
        """
        Evaluate after training procedure finished (batch-wised)
        """
        self.model.eval()

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_data_loader):

                loss = torch.tensor(0).to(self.device, dtype=torch.float)
                outputs = torch.zeros(len(target), self.num_classes)
                targets = torch.zeros(len(target), self.num_classes)

                for i in  range(len(data)):
                    data[i], target[i] = data[i].to(self.device, dtype=torch.float), target[i].to(self.device, dtype=torch.float)
                    output = self.model(data[i])
                    loss_i = self.criterion(output, target[i])
                    loss += loss_i
                    outputs[i:i + 1, :] = output
                    targets[i:i + 1, :] = target[i]

                loss /= len(data)
                self.test_metrics.update('loss', loss.item())

                outputs_logit = self.sigmoid(outputs)
                for met in self.metric_ftns:
                    self.test_metrics.update(met.__name__, met(self._to_np(outputs_logit), self._to_np(targets)))

        result = self.test_metrics.result()

        # print logged informations to the screen
        for key, value in result.items():
            self.logger.info('    {:15s}: {}'.format(str(key), value))

    def evaluate(self):
        """
        Evaluate after training procedure finished
        """

        self.model.eval()

        Outputs = torch.zeros((self.test_data_loader.n_samples, self.num_classes))
        Targets = torch.zeros((self.test_data_loader.n_samples, self.num_classes))

        with torch.no_grad():
            start = 0
            for batch_idx, (data, target) in enumerate(self.test_data_loader):
                end = len(data) + start

                outputs = torch.zeros(len(target), self.num_classes)
                targets = torch.zeros(len(target), self.num_classes)

                for i in range(len(data)):
                    data[i], target[i] = data[i].to(self.device, dtype=torch.float), target[i].to(self.device, dtype=torch.float)
                    output = self.model(data[i])

                    outputs[i:i+1, :] = output
                    targets[i:i+1, :] = target[i]

                Outputs[start:end, :] = outputs
                Targets[start:end, :] = targets
                start = end

            loss = self.criterion(Outputs, Targets)
            self.test_metrics.update('loss', loss.item())

        Outputs_logit = self.sigmoid(Outputs)
        for met in self.metric_ftns:
            self.test_metrics.update(met.__name__, met(self._to_np(Outputs_logit), self._to_np(Targets)))

        result = self.test_metrics.result()

        # print logged informations to the screen
        for key, value in result.items():
            self.logger.info('    {:15s}: {}'.format(str(key), value))

    def analyze(self, challenge_metrics):
        """
        Analyze after training procedure finished
        """
        self.model.eval()

        Outputs = torch.zeros((self.test_data_loader.n_samples, self.num_classes))
        Targets = torch.zeros((self.test_data_loader.n_samples, self.num_classes))

        with torch.no_grad():
            start = 0
            for batch_idx, (data, target) in enumerate(self.test_data_loader):
                end = len(data) + start

                outputs = torch.zeros(len(target), self.num_classes)
                targets = torch.zeros(len(target), self.num_classes)

                for i in range(len(data)):
                    data[i], target[i] = data[i].to(self.device, dtype=torch.float), target[i].to(self.device,dtype=torch.float)
                    output = self.model(data[i])

                    outputs[i:i + 1, :] = output
                    targets[i:i + 1, :] = target[i]

                Outputs[start:end, :] = outputs
                Targets[start:end, :] = targets
                start = end

            loss = self.criterion(Outputs, Targets)
            self.test_metrics.update('loss', loss.item())

        indices = challenge_metrics.indices
        classes = challenge_metrics.classes
        test_idx = self.test_data_loader.idx
        file_names = self.test_data_loader.file_names

        dataset_idx_list = [[] for i in range(6)]
        for i in range(len(test_idx)):
            if file_names[test_idx[i]].startswith('A'):  # CPSC 2018 data, 6,877 recordings
                dataset_idx_list[0].append(i)
            elif file_names[test_idx[i]].startswith('Q'):  # Unused CPSC2018 data, 3,453 recordings
                dataset_idx_list[1].append(i)
            elif file_names[test_idx[i]].startswith('S'):  # PTB Diagnostic ECG Database, 516 recordings
                dataset_idx_list[2].append(i)
            elif file_names[test_idx[i]].startswith('HR'):  # PTB-XL electrocardiography Database, 21,837 recordings
                dataset_idx_list[3].append(i)
            elif file_names[test_idx[i]].startswith('E'):  # Georgia 12-Lead ECG Challenge Database, 10,344 recordings
                dataset_idx_list[4].append(i)
            else:
                dataset_idx_list[5].append(i)  # St Petersburg INCART 12-lead Arrhythmia Database, 74 recordings

        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)

        # All Testing Data
        self.logger.info("**********************************************************************************")
        self.logger.info("All Testing Data")

        Outputs_logit = self.sigmoid(Outputs)
        accuracy = challenge_metrics.accuracy(self._to_np(Outputs_logit), self._to_np(Targets))
        macro_f_measure, f_measure = challenge_metrics.f_measure(self._to_np(Outputs_logit), self._to_np(Targets))
        macro_f_beta_measure, macro_g_beta_measure, f_beta_measure, g_beta_measure = challenge_metrics.beta_measures(
            self._to_np(Outputs_logit), self._to_np(Targets))
        macro_auroc, macro_auprc, auroc, auprc = challenge_metrics.auc(self._to_np(Outputs_logit), self._to_np(Targets))
        challenge_metric = challenge_metrics.challenge_metric(self._to_np(Outputs_logit), self._to_np(Targets))

        self.logger.info("challenge_metric:{}".format(challenge_metric))
        self.logger.info("accuracy:{}".format(accuracy))
        self.logger.info("macro_f_measure:{}".format(macro_f_measure))
        self.logger.info("macro_g_beta_measure:{}".format(macro_g_beta_measure))
        self.logger.info("macro_auroc:{}".format(macro_auroc))
        self.logger.info("macro_auprc:{}".format(macro_auprc))

        metrics = np.vstack((f_measure, f_beta_measure, g_beta_measure, auroc, auprc))
        metrics_df = pd.DataFrame(data=metrics, columns=classes,
                                  index=['f_measure', 'f_beta_measure', 'g_beta_measure', 'auroc', 'auprc'])
        self.logger.info(metrics_df)
        metrics_df.to_csv(str(self.result_dir) + '/All.csv')

        Dataset = ['CPSC 2018 data', 'Unused CPSC2018 data', 'PTB Diagnostic ECG Database',
                   'PTB-XL electrocardiography Database',
                   'Georgia 12-Lead ECG Challenge Database', 'St Petersburg INCART 12-lead Arrhythmia Database']

        for i in range(len(Dataset)):
            outputs_i_logit = self._to_np(Outputs_logit)[dataset_idx_list[i]]
            targets_i = self._to_np(Targets)[dataset_idx_list[i]]

            accuracy = challenge_metrics.accuracy(outputs_i_logit, targets_i)
            macro_f_measure, f_measure = challenge_metrics.f_measure(outputs_i_logit, targets_i)
            macro_f_beta_measure, macro_g_beta_measure, f_beta_measure, g_beta_measure = challenge_metrics.beta_measures(
                outputs_i_logit, targets_i)
            macro_auroc, macro_auprc, auroc, auprc = challenge_metrics.auc(outputs_i_logit, targets_i)
            challenge_metric = challenge_metrics.challenge_metric(outputs_i_logit, targets_i)

            self.logger.info("**********************************************************************************")
            self.logger.info(Dataset[i])
            self.logger.info("challenge_metric:{}".format(challenge_metric))
            self.logger.info("accuracy:{}".format(accuracy))
            self.logger.info("macro_f_measure:{}".format(macro_f_measure))
            self.logger.info("macro_g_beta_measure:{}".format(macro_g_beta_measure))
            self.logger.info("macro_auroc:{}".format(macro_auroc))
            self.logger.info("macro_auprc:{}".format(macro_auprc))

            metrics = np.vstack((f_measure, f_beta_measure, g_beta_measure, auroc, auprc))
            metrics_df = pd.DataFrame(data=metrics, columns=classes,
                                      index=['f_measure', 'f_beta_measure', 'g_beta_measure', 'auroc', 'auprc'])
            self.logger.info(metrics_df)
            metrics_df.to_csv(str(self.result_dir) + '/' + Dataset[i] + '.csv')

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