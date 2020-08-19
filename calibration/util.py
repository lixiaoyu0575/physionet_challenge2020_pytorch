import numpy as np
from scipy.io import loadmat
from utils.dataset import load_label_files, load_labels, load_weights
from data_loader.util import load_challenge_data
import torch
import torch.nn.functional as F


class ChallengeMetric():

    def __init__(self, input_directory, alphas):

        # challengeMetric initialization
        weights_file = '../evaluation/weights.csv'
        normal_class = '426783006'
        equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]

        # Find the label files.
        print('Finding label...')
        label_files = load_label_files(input_directory)

        # Load the labels and classes.
        print('Loading labels...')
        classes, labels_onehot, labels = load_labels(label_files, normal_class, equivalent_classes)

        num_files = len(label_files)
        print("num_files:", num_files)

        # Load the weights for the Challenge metric.
        print('Loading weights...')
        weights = load_weights(weights_file, classes)

        # Only consider classes that are scored with the Challenge metric.
        indices = np.any(weights, axis=0)  # Find indices of classes in weight matrix.
        classes = [x for i, x in enumerate(classes) if indices[i]]
        weights = weights[np.ix_(indices, indices)]

        self.weights = weights
        self.indices = indices
        self.classes = classes
        self.normal_class = normal_class

        self.alphas = alphas

    # Compute recording-wise accuracy.
    def accuracy(self, outputs, labels):
        outputs = outputs[:, self.indices]
        labels = labels[:, self.indices]
        outputs = self.get_pred(outputs)

        num_recordings, num_classes = np.shape(labels)

        num_correct_recordings = 0
        for i in range(num_recordings):
            if np.all(labels[i, :] == outputs[i, :]):
                num_correct_recordings += 1

        return float(num_correct_recordings) / float(num_recordings)

    # Compute confusion matrices.
    def confusion_matrices(self, outputs, labels, normalize=False):
        # Compute a binary confusion matrix for each class k:
        #
        #     [TN_k FN_k]
        #     [FP_k TP_k]
        #
        # If the normalize variable is set to true, then normalize the contributions
        # to the confusion matrix by the number of labels per recording.
        num_recordings, num_classes = np.shape(labels)

        if not normalize:
            A = np.zeros((num_classes, 2, 2))
            for i in range(num_recordings):
                for j in range(num_classes):
                    if labels[i, j] == 1 and outputs[i, j] == 1:  # TP
                        A[j, 1, 1] += 1
                    elif labels[i, j] == 0 and outputs[i, j] == 1:  # FP
                        A[j, 1, 0] += 1
                    elif labels[i, j] == 1 and outputs[i, j] == 0:  # FN
                        A[j, 0, 1] += 1
                    elif labels[i, j] == 0 and outputs[i, j] == 0:  # TN
                        A[j, 0, 0] += 1
                    else:  # This condition should not happen.
                        raise ValueError('Error in computing the confusion matrix.')
        else:
            A = np.zeros((num_classes, 2, 2))
            for i in range(num_recordings):
                normalization = float(max(np.sum(labels[i, :]), 1))
                for j in range(num_classes):
                    if labels[i, j] == 1 and outputs[i, j] == 1:  # TP
                        A[j, 1, 1] += 1.0 / normalization
                    elif labels[i, j] == 0 and outputs[i, j] == 1:  # FP
                        A[j, 1, 0] += 1.0 / normalization
                    elif labels[i, j] == 1 and outputs[i, j] == 0:  # FN
                        A[j, 0, 1] += 1.0 / normalization
                    elif labels[i, j] == 0 and outputs[i, j] == 0:  # TN
                        A[j, 0, 0] += 1.0 / normalization
                    else:  # This condition should not happen.
                        raise ValueError('Error in computing the confusion matrix.')

        return A

    # Compute macro F-measure.
    def f_measure(self, outputs, labels):
        outputs = outputs[:, self.indices]
        labels = labels[:, self.indices]
        outputs = self.get_pred(outputs)
        num_recordings, num_classes = np.shape(labels)

        A = self.confusion_matrices(outputs, labels)

        f_measure = np.zeros(num_classes)
        for k in range(num_classes):
            tp, fp, fn, tn = A[k, 1, 1], A[k, 1, 0], A[k, 0, 1], A[k, 0, 0]
            if 2 * tp + fp + fn:
                f_measure[k] = float(2 * tp) / float(2 * tp + fp + fn)
            else:
                f_measure[k] = float('nan')

        macro_f_measure = np.nanmean(f_measure)

        return macro_f_measure

    def beta_measures(self, outputs, labels, beta=2):
        outputs = outputs[:, self.indices]
        labels = labels[:, self.indices]
        outputs = self.get_pred(outputs)
        num_recordings, num_classes = np.shape(labels)

        A = self.confusion_matrices(outputs, labels, normalize=True)

        f_beta_measure = np.zeros(num_classes)
        g_beta_measure = np.zeros(num_classes)
        for k in range(num_classes):
            tp, fp, fn, tn = A[k, 1, 1], A[k, 1, 0], A[k, 0, 1], A[k, 0, 0]
            if (1 + beta ** 2) * tp + fp + beta ** 2 * fn:
                f_beta_measure[k] = float((1 + beta ** 2) * tp) / float((1 + beta ** 2) * tp + fp + beta ** 2 * fn)
            else:
                f_beta_measure[k] = float('nan')
            if tp + fp + beta * fn:
                g_beta_measure[k] = float(tp) / float(tp + fp + beta * fn)
            else:
                g_beta_measure[k] = float('nan')

        macro_f_beta_measure = np.nanmean(f_beta_measure)
        macro_g_beta_measure = np.nanmean(g_beta_measure)

        return macro_f_beta_measure, macro_g_beta_measure

    # Compute modified confusion matrix for multi-class, multi-label tasks.
    def modified_confusion_matrix(self, outputs, labels):
        # Compute a binary multi-class, multi-label confusion matrix, where the rows
        # are the labels and the columns are the outputs.
        num_recordings, num_classes = np.shape(labels)

        A = np.zeros((num_classes, num_classes))

        # Iterate over all of the recordings.
        for i in range(num_recordings):
            # Calculate the number of positive labels and/or outputs.
            normalization = float(max(np.sum(np.any((labels[i, :], outputs[i, :]), axis=0)), 1))
            # Iterate over all of the classes.
            for j in range(num_classes):
                # Assign full and/or partial credit for each positive class.
                if labels[i, j]:
                    for k in range(num_classes):
                        if outputs[i, k]:
                            A[j, k] += 1.0 / normalization

        return A

    # Compute the evaluation metric for the Challenge.
    def challenge_metric(self, outputs, labels):
        outputs = outputs[:, self.indices]
        labels = labels[:, self.indices]
        outputs = self.get_pred(outputs)

        num_recordings, num_classes = np.shape(labels)
        normal_index = self.classes.index(self.normal_class)

        # Compute the observed score.
        A = self.modified_confusion_matrix(outputs, labels)
        observed_score = np.nansum(self.weights * A)

        # Compute the score for the model that always chooses the correct label(s).
        correct_outputs = labels
        A = self.modified_confusion_matrix(labels, correct_outputs)
        correct_score = np.nansum(self.weights * A)

        # Compute the score for the model that always chooses the normal class.
        inactive_outputs = np.zeros((num_recordings, num_classes), dtype=np.bool)
        inactive_outputs[:, normal_index] = 1
        A = self.modified_confusion_matrix(labels, inactive_outputs)
        inactive_score = np.nansum(self.weights * A)

        if correct_score != inactive_score:
            normalized_score = float(observed_score - inactive_score) / float(correct_score - inactive_score)
        else:
            normalized_score = float('nan')

        return normalized_score

    def get_pred(self, output):
        num_recordings, num_classes = output.shape
        labels = np.zeros((num_recordings, num_classes))
        for i in range(num_recordings):
            for j in range(num_classes):
                if output[i, j] >= self.alphas[j]:
                    labels[i, j] = 1
                else:
                    labels[i, j] = 0
        return labels

def load_data(label_dir, data_dir, split_index):
    print('Loading data...')
    weights_file = '../evaluation/weights.csv'
    normal_class = '426783006'
    equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]

    # Find the label files.
    label_files = load_label_files(label_dir)
    # Load the labels and classes.
    classes, labels_onehot, labels = load_labels(label_files, normal_class, equivalent_classes)
    # Load the weights for the Challenge metric.
    weights = load_weights(weights_file, classes)
    # Classes that are scored with the Challenge metric.
    indices = np.any(weights, axis=0)  # Find indices of classes in weight matrix.
    classes_scored = [x for i, x in enumerate(classes) if indices[i]]

    split_idx = loadmat(split_index)
    train_index, val_index, test_index = split_idx['train_index'], split_idx['val_index'], split_idx['test_index']
    train_index = train_index.reshape((train_index.shape[1],))
    val_index = val_index.reshape((val_index.shape[1],))
    test_index = test_index.reshape((test_index.shape[1],))

    train_data = list()
    val_data = list()
    test_data = list()
    train_label = list()
    val_label = list()
    test_label = list()

    num_files = len(label_files)
    for i in range(num_files):
        recording, header, name = load_challenge_data(label_files[i], data_dir)
        recording[np.isnan(recording)] = 0

        if i in train_index:
            for j in range(recording.shape[0]):
                train_data.append(recording[j])
                train_label.append(labels_onehot[i])
        elif i in val_index:
            for j in range(recording.shape[0]):
                val_data.append(recording[j])
                val_label.append(labels_onehot[i])
        else:
            for j in range(recording.shape[0]):
                test_data.append(recording[j])
                test_label.append(labels_onehot[i])
    train_data = np.array(train_data)
    val_data = np.array(val_data)
    test_data = np.array(test_data)
    train_label = np.array(train_label)
    val_label = np.array(val_label)
    test_label = np.array(test_label)

    return (train_data, train_label), (val_data, val_label), (test_data, test_label)

def load_checkpoint(resume_path):
    """
    Resume from saved checkpoints

    :param resume_path: Checkpoint path to be resumed
    """
    checkpoint = torch.load(resume_path)
    epoch = checkpoint['epoch']
    mnt_best = checkpoint['monitor_best']
    # load architecture params from checkpoint.
    return checkpoint

def get_metrics(outputs, targets, challenge_metrics):
    accuracy = challenge_metrics.accuracy(outputs, targets)
    macro_f_measure = challenge_metrics.f_measure(outputs, targets)
    macro_f_beta_measure, macro_g_beta_measure = challenge_metrics.beta_measures(outputs, targets)
    challenge_metric = challenge_metrics.challenge_metric(outputs, targets)
    return accuracy, macro_f_measure, macro_f_beta_measure, macro_g_beta_measure, challenge_metric

def to_np(tensor, device):
    if device == 'cuda':
        return tensor.cpu().detach().numpy()
    else:
        return tensor.detach().numpy()
