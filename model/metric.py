import torch
import numpy as np
from utils.dataset import load_label_files, load_labels, load_weights

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

# Challenge2020 official evaluation
class ChallengeMetric():

    def __init__(self, input_directory):

        # challengeMetric initialization
        weights_file = '/home/weiyuhua/Data/challenge2020/evaluation/weights.csv'
        # weights_file = 'F:\challenge2020\evaluation\weights.csv'
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

        self.weights = weights
        self.classes = classes
        self.normal_class = normal_class

    # Compute recording-wise accuracy.
    def accuracy(self, outputs, labels):
        num_recordings, num_classes = np.shape(labels)

        num_correct_recordings = 0
        for i in range(num_recordings):
            if np.all(labels[i, :]==outputs[i, :]):
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
                    if labels[i, j]==1 and outputs[i, j]==1: # TP
                        A[j, 1, 1] += 1
                    elif labels[i, j]==0 and outputs[i, j]==1: # FP
                        A[j, 1, 0] += 1
                    elif labels[i, j]==1 and outputs[i, j]==0: # FN
                        A[j, 0, 1] += 1
                    elif labels[i, j]==0 and outputs[i, j]==0: # TN
                        A[j, 0, 0] += 1
                    else: # This condition should not happen.
                        raise ValueError('Error in computing the confusion matrix.')
        else:
            A = np.zeros((num_classes, 2, 2))
            for i in range(num_recordings):
                normalization = float(max(np.sum(labels[i, :]), 1))
                for j in range(num_classes):
                    if labels[i, j]==1 and outputs[i, j]==1: # TP
                        A[j, 1, 1] += 1.0/normalization
                    elif labels[i, j]==0 and outputs[i, j]==1: # FP
                        A[j, 1, 0] += 1.0/normalization
                    elif labels[i, j]==1 and outputs[i, j]==0: # FN
                        A[j, 0, 1] += 1.0/normalization
                    elif labels[i, j]==0 and outputs[i, j]==0: # TN
                        A[j, 0, 0] += 1.0/normalization
                    else: # This condition should not happen.
                        raise ValueError('Error in computing the confusion matrix.')

        return A

    # Compute macro F-measure.
    def f_measure(self, outputs, labels):
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

    # Compute F-beta and G-beta measures from the unofficial phase of the Challenge.
    def macro_f_beta_measure(self, outputs, labels, beta=2):
        return self.beta_measures(outputs, labels, beta)[0]

    def macro_g_beta_measure(self, outputs, labels, beta=2):
        return self.beta_measures(outputs, labels, beta)[1]

    def beta_measures(self, outputs, labels, beta=2):
        num_recordings, num_classes = np.shape(labels)

        A = self.confusion_matrices(outputs, labels, normalize=True)

        f_beta_measure = np.zeros(num_classes)
        g_beta_measure = np.zeros(num_classes)
        for k in range(num_classes):
            tp, fp, fn, tn = A[k, 1, 1], A[k, 1, 0], A[k, 0, 1], A[k, 0, 0]
            if (1+beta**2)*tp + fp + beta**2*fn:
                f_beta_measure[k] = float((1+beta**2)*tp) / float((1+beta**2)*tp + fp + beta**2*fn)
            else:
                f_beta_measure[k] = float('nan')
            if tp + fp + beta*fn:
                g_beta_measure[k] = float(tp) / float(tp + fp + beta*fn)
            else:
                g_beta_measure[k] = float('nan')

        macro_f_beta_measure = np.nanmean(f_beta_measure)
        macro_g_beta_measure = np.nanmean(g_beta_measure)

        return macro_f_beta_measure, macro_g_beta_measure

    # Compute macro AUROC and macro AUPRC.
    def macro_auroc(self, outputs, labels):
        return self.auc(outputs, labels)[0]

    def macro_auprc(self, outputs, labels):
        return self.auc(outputs, labels)[1]

    def auc(self, outputs, labels):
        num_recordings, num_classes = np.shape(labels)

        # Compute and summarize the confusion matrices for each class across at distinct output values.
        auroc = np.zeros(num_classes)
        auprc = np.zeros(num_classes)

        for k in range(num_classes):
            # We only need to compute TPs, FPs, FNs, and TNs at distinct output values.
            thresholds = np.unique(outputs[:, k])
            thresholds = np.append(thresholds, thresholds[-1]+1)
            thresholds = thresholds[::-1]
            num_thresholds = len(thresholds)

            # Initialize the TPs, FPs, FNs, and TNs.
            tp = np.zeros(num_thresholds)
            fp = np.zeros(num_thresholds)
            fn = np.zeros(num_thresholds)
            tn = np.zeros(num_thresholds)
            fn[0] = np.sum(labels[:, k]==1)
            tn[0] = np.sum(labels[:, k]==0)

            # Find the indices that result in sorted output values.
            idx = np.argsort(outputs[:, k])[::-1]

            # Compute the TPs, FPs, FNs, and TNs for class k across thresholds.
            i = 0
            for j in range(1, num_thresholds):
                # Initialize TPs, FPs, FNs, and TNs using values at previous threshold.
                tp[j] = tp[j-1]
                fp[j] = fp[j-1]
                fn[j] = fn[j-1]
                tn[j] = tn[j-1]

                # Update the TPs, FPs, FNs, and TNs at i-th output value.
                while i < num_recordings and outputs[idx[i], k] >= thresholds[j]:
                    if labels[idx[i], k]:
                        tp[j] += 1
                        fn[j] -= 1
                    else:
                        fp[j] += 1
                        tn[j] -= 1
                    i += 1

            # Summarize the TPs, FPs, FNs, and TNs for class k.
            tpr = np.zeros(num_thresholds)
            tnr = np.zeros(num_thresholds)
            ppv = np.zeros(num_thresholds)
            npv = np.zeros(num_thresholds)

            for j in range(num_thresholds):
                if tp[j] + fn[j]:
                    tpr[j] = float(tp[j]) / float(tp[j] + fn[j])
                else:
                    tpr[j] = float('nan')
                if fp[j] + tn[j]:
                    tnr[j] = float(tn[j]) / float(fp[j] + tn[j])
                else:
                    tnr[j] = float('nan')
                if tp[j] + fp[j]:
                    ppv[j] = float(tp[j]) / float(tp[j] + fp[j])
                else:
                    ppv[j] = float('nan')

            # Compute AUROC as the area under a piecewise linear function with TPR/
            # sensitivity (x-axis) and TNR/specificity (y-axis) and AUPRC as the area
            # under a piecewise constant with TPR/recall (x-axis) and PPV/precision
            # (y-axis) for class k.
            for j in range(num_thresholds-1):
                auroc[k] += 0.5 * (tpr[j+1] - tpr[j]) * (tnr[j+1] + tnr[j])
                auprc[k] += (tpr[j+1] - tpr[j]) * ppv[j+1]

        # Compute macro AUROC and macro AUPRC across classes.
        macro_auroc = np.nanmean(auroc)
        macro_auprc = np.nanmean(auprc)

        return macro_auroc, macro_auprc

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
                            A[j, k] += 1.0/normalization

        return A

    # Compute the evaluation metric for the Challenge.
    def challenge_metric(self, outputs, labels):

        classes = self.classes
        normal_class = self.normal_class
        weights = self.weights

        num_recordings, num_classes = np.shape(labels)
        normal_index = classes.index(normal_class)

        # Compute the observed score.
        A = self.modified_confusion_matrix(outputs, labels)
        observed_score = np.nansum(weights * A)

        # Compute the score for the model that always chooses the correct label(s).
        correct_outputs = labels
        A = self.modified_confusion_matrix(labels, correct_outputs)
        correct_score = np.nansum(weights * A)

        # Compute the score for the model that always chooses the normal class.
        inactive_outputs = np.zeros((num_recordings, num_classes), dtype=np.bool)
        inactive_outputs[:, normal_index] = 1
        A = self.modified_confusion_matrix(labels, inactive_outputs)
        inactive_score = np.nansum(weights * A)

        if correct_score != inactive_score:
            normalized_score = float(observed_score - inactive_score) / float(correct_score - inactive_score)
        else:
            normalized_score = float('nan')

        return normalized_score