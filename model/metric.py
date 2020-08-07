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
        weights_file = 'evaluation/weights.csv'
        normal_class = '426783006'
        equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]

        # Find the label files.
        print('Finding label...')
        label_files = load_label_files(input_directory)

        # Load the labels and classes.
        print('Loading labels...')
        classes, _, _ = load_labels(label_files, normal_class, equivalent_classes)

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
        self._return_metric_list = False

    def return_metric_list(self):
        self._return_metric_list = True

    # Compute recording-wise accuracy.
    def accuracy(self, outputs, labels):
        outputs = self.get_pred(outputs)
        outputs = outputs[:, self.indices]
        labels = labels[:, self.indices]

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
                    if labels[i, j] >= 0.5 and outputs[i, j] >= 0.5:  # TP
                        A[j, 1, 1] += 1
                    elif labels[i, j] < 0.5 and outputs[i, j] >= 0.5:  # FP
                        A[j, 1, 0] += 1
                    elif labels[i, j] >= 0.5 and outputs[i, j] < 0.5:  # FN
                        A[j, 0, 1] += 1
                    elif labels[i, j] < 0.5 and outputs[i, j] < 0.5:  # TN
                        A[j, 0, 0] += 1
                    else:  # This condition should not happen.
                        raise ValueError('Error in computing the confusion matrix.')
        else:
            A = np.zeros((num_classes, 2, 2))
            for i in range(num_recordings):
                normalization = float(max(np.sum(labels[i, :]), 1))
                for j in range(num_classes):
                    if labels[i, j] >= 0.5 and outputs[i, j] >= 0.5:  # TP
                        A[j, 1, 1] += 1.0 / normalization
                    elif labels[i, j] < 0.5 and outputs[i, j] >= 0.5:  # FP
                        A[j, 1, 0] += 1.0 / normalization
                    elif labels[i, j] >= 0.5 and outputs[i, j] < 0.5:  # FN
                        A[j, 0, 1] += 1.0 / normalization
                    elif labels[i, j] < 0.5 and outputs[i, j] < 0.5:  # TN
                        A[j, 0, 0] += 1.0 / normalization
                    else:  # This condition should not happen.
                        raise ValueError('Error in computing the confusion matrix.')

        return A

    # Compute macro F-measure.
    def f_measure(self, outputs, labels):
        outputs = self.get_pred(outputs)
        outputs = outputs[:, self.indices]
        labels = labels[:, self.indices]
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

        if self._return_metric_list:
            return macro_f_measure, f_measure
        else:
            return macro_f_measure

    # Compute F-beta and G-beta measures from the unofficial phase of the Challenge.
    def macro_f_beta_measure(self, outputs, labels, beta=2):
        macro_f_beta_measure, macro_g_beta_measure, f_beta_measure, g_beta_measure = self.beta_measures(outputs, labels, beta)
        if self._return_metric_list:
            return macro_f_beta_measure, f_beta_measure
        else:
            return macro_f_beta_measure

    def macro_g_beta_measure(self, outputs, labels, beta=2):
        macro_f_beta_measure, macro_g_beta_measure, f_beta_measure, g_beta_measure = self.beta_measures(outputs, labels,
                                                                                                        beta)
        if self._return_metric_list:
            return macro_g_beta_measure, g_beta_measure
        else:
            return macro_g_beta_measure

    def beta_measures(self, outputs, labels, beta=2):
        outputs = self.get_pred(outputs)
        outputs = outputs[:, self.indices]
        labels = labels[:, self.indices]
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

        return macro_f_beta_measure, macro_g_beta_measure, f_beta_measure, g_beta_measure

    # Compute macro AUROC and macro AUPRC.
    def macro_auroc(self, outputs, labels):
        macro_auroc, macro_auprc, auroc, auprc = self.auc(outputs, labels)
        if self._return_metric_list:
            return macro_auroc, auroc
        else:
            return macro_auroc

    def macro_auprc(self, outputs, labels):
        macro_auroc, macro_auprc, auroc, auprc = self.auc(outputs, labels)
        if self._return_metric_list:
            return macro_auprc, auprc
        else:
            return macro_auprc

    def auc(self, outputs, labels):
        outputs = outputs[:, self.indices]
        labels = labels[:, self.indices]
        num_recordings, num_classes = np.shape(labels)

        # Compute and summarize the confusion matrices for each class across at distinct output values.
        auroc = np.zeros(num_classes)
        auprc = np.zeros(num_classes)

        for k in range(num_classes):
            # We only need to compute TPs, FPs, FNs, and TNs at distinct output values.
            thresholds = np.unique(outputs[:, k])
            thresholds = np.append(thresholds, thresholds[-1] + 1)
            thresholds = thresholds[::-1]
            num_thresholds = len(thresholds)

            # Initialize the TPs, FPs, FNs, and TNs.
            tp = np.zeros(num_thresholds)
            fp = np.zeros(num_thresholds)
            fn = np.zeros(num_thresholds)
            tn = np.zeros(num_thresholds)
            fn[0] = np.sum(labels[:, k] >= 0.5)
            tn[0] = np.sum(labels[:, k] < 0.5)

            # Find the indices that result in sorted output values.
            idx = np.argsort(outputs[:, k])[::-1]

            # Compute the TPs, FPs, FNs, and TNs for class k across thresholds.
            i = 0
            for j in range(1, num_thresholds):
                # Initialize TPs, FPs, FNs, and TNs using values at previous threshold.
                tp[j] = tp[j - 1]
                fp[j] = fp[j - 1]
                fn[j] = fn[j - 1]
                tn[j] = tn[j - 1]

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
            for j in range(num_thresholds - 1):
                auroc[k] += 0.5 * (tpr[j + 1] - tpr[j]) * (tnr[j + 1] + tnr[j])
                auprc[k] += (tpr[j + 1] - tpr[j]) * ppv[j + 1]

        # Compute macro AUROC and macro AUPRC across classes.
        macro_auroc = np.nanmean(auroc)
        macro_auprc = np.nanmean(auprc)

        return macro_auroc, macro_auprc, auroc, auprc

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
                if labels[i, j] > 0.5:
                    for k in range(num_classes):
                        if outputs[i, k] > 0.5:
                            A[j, k] += 1.0/normalization
        return A

    # Compute the evaluation metric for the Challenge.
    def challenge_metric(self, outputs, labels):

        outputs = self.get_pred(outputs)
        outputs = outputs[:, self.indices]
        labels = labels[:, self.indices]

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

    def get_pred(self, outputs, alpha=0.5):
        for i in range(outputs.shape[0]):
            for j in range(outputs.shape[1]):
                if outputs[i, j] >= alpha:
                    outputs[i, j] = 1
                else:
                    outputs[i, j] = 0
        return outputs

# Challenge2020 official evaluation (unofficial stage)
class ChallengeMetric2():

    def __init__(self, num_classes):
        self.num_classes = num_classes
    # The compute_beta_score function computes the Fbeta-measure given an specific beta value
    # and the G value define at the begining of the file.
    #
    # Inputs:
    #   'labels' are the true classes of the recording
    #
    #   'output' are the output classes of your model
    #
    #   'beta' is the weight
    #
    # Outputs:
    #
    # fbeta_measure, Fbeta measure given an specific beta
    # Gbeta_measure, Generalization of the Jaccard measure with a beta weigth
    #
    def accuracy(self, outputs, labels):
        return self.beta_score(outputs, labels)[0]

    def f_measure(self, outputs, labels):
        return self.beta_score(outputs, labels)[1]

    def f_beta(self, outputs, labels, beta=2):
        return self.beta_score(outputs, labels, beta)[2]

    def g_beta(self, outputs, labels, beta=2):
        return self.beta_score(outputs, labels, beta)[3]

    def beta_score(self, outputs, labels, beta=2, check_errors=True):

        outputs = self.get_pred(outputs)
        # Check inputs for errors.
        if check_errors:
            if len(outputs) != len(labels):
                raise Exception('Numbers of outputs and labels must be the same.')

        self.num_classes = labels.shape[1]
        # Populate contingency table.
        num_recordings = len(labels)

        fbeta_l = np.zeros(self.num_classes)
        gbeta_l = np.zeros(self.num_classes)
        fmeasure_l = np.zeros(self.num_classes)
        accuracy_l = np.zeros(self.num_classes)

        f_beta = 0
        g_beta = 0
        f_measure = 0
        accuracy = 0

        # Weight function
        C_l = np.ones(self.num_classes)

        for j in range(self.num_classes):
            tp = 0
            fp = 0
            fn = 0
            tn = 0

            for i in range(num_recordings):

                num_labels = np.sum(labels[i])

                if labels[i][j] and outputs[i][j]:
                    tp += 1 / num_labels
                elif not labels[i][j] and outputs[i][j]:
                    fp += 1 / num_labels
                elif labels[i][j] and not outputs[i][j]:
                    fn += 1 / num_labels
                elif not labels[i][j] and not outputs[i][j]:
                    tn += 1 / num_labels

            # Summarize contingency table.
            if ((1 + beta ** 2) * tp + (fn * beta ** 2) + fp):
                fbeta_l[j] = float((1 + beta ** 2) * tp) / float(((1 + beta ** 2) * tp) + (fn * beta ** 2) + fp)
            else:
                fbeta_l[j] = 1.0

            if (tp + fp + beta * fn):
                gbeta_l[j] = float(tp) / float(tp + fp + beta * fn)
            else:
                gbeta_l[j] = 1.0

            if tp + fp + fn + tn:
                accuracy_l[j] = float(tp + tn) / float(tp + fp + fn + tn)
            else:
                accuracy_l[j] = 1.0

            if 2 * tp + fp + fn:
                fmeasure_l[j] = float(2 * tp) / float(2 * tp + fp + fn)
            else:
                fmeasure_l[j] = 1.0

        for i in range(self.num_classes):
            f_beta += fbeta_l[i] * C_l[i]
            g_beta += gbeta_l[i] * C_l[i]
            f_measure += fmeasure_l[i] * C_l[i]
            accuracy += accuracy_l[i] * C_l[i]

        f_beta = float(f_beta) / float(self.num_classes)
        g_beta = float(g_beta) / float(self.num_classes)
        f_measure = float(f_measure) / float(self.num_classes)
        accuracy = float(accuracy) / float(self.num_classes)

        return accuracy, f_measure, f_beta, g_beta

    # The compute_auc function computes AUROC and AUPRC as well as other summary
    # statistics (TP, FP, FN, TN, TPR, TNR, PPV, NPV, etc.) that can be exposed
    # from this function.
    #
    # Inputs:
    #   'labels' are the true classes of the recording
    #
    #   'output' are the output classes of your model
    #
    #   'beta' is the weight
    #
    #
    # Outputs:
    #   'auroc' is a scalar that gives the AUROC of the algorithm using its
    #   output probabilities, where specificity is interpolated for intermediate
    #   sensitivity values.
    #
    #   'auprc' is a scalar that gives the AUPRC of the algorithm using its
    #   output probabilities, where precision is a piecewise constant function of
    #   recall.
    #
    def auroc(self, probabilities, labels):
        return self.auc(probabilities, labels)[0]

    def auprc(self, probabilities, labels):
        return self.auc(probabilities, labels)[1]

    def auc(self, probabilities, labels, check_errors=True):

        # Check inputs for errors.
        if check_errors:
            if len(labels) != len(probabilities):
                raise Exception('Numbers of outputs and labels must be the same.')

        find_NaNs = np.isnan(probabilities)
        probabilities[find_NaNs] = 0

        self.num_classes = labels.shape[1]

        auroc_l = np.zeros(self.num_classes)
        auprc_l = np.zeros(self.num_classes)

        auroc = 0
        auprc = 0

        # Weight function - this will change
        C_l = np.ones(self.num_classes)

        # Populate contingency table.
        num_recordings = len(labels)

        for k in range(self.num_classes):

            # Find probabilities thresholds.
            thresholds = np.unique(probabilities[:, k])[::-1]
            if thresholds[0] != 1:
                thresholds = np.insert(thresholds, 0, 1)
            if thresholds[-1] == 0:
                thresholds = thresholds[:-1]

            m = len(thresholds)

            # Populate contingency table across probabilities thresholds.
            tp = np.zeros(m)
            fp = np.zeros(m)
            fn = np.zeros(m)
            tn = np.zeros(m)

            # Find indices that sort the predicted probabilities from largest to
            # smallest.
            idx = np.argsort(probabilities[:, k])[::-1]

            i = 0
            for j in range(m):
                # Initialize contingency table for j-th probabilities threshold.
                if j == 0:
                    tp[j] = 0
                    fp[j] = 0
                    fn[j] = np.sum(labels[:, k])
                    tn[j] = num_recordings - fn[j]
                else:
                    tp[j] = tp[j - 1]
                    fp[j] = fp[j - 1]
                    fn[j] = fn[j - 1]
                    tn[j] = tn[j - 1]
                # Update contingency table for i-th largest predicted probability.
                while i < num_recordings and probabilities[idx[i], k] >= thresholds[j]:
                    if labels[idx[i], k]:
                        tp[j] += 1
                        fn[j] -= 1
                    else:
                        fp[j] += 1
                        tn[j] -= 1
                    i += 1

            # Summarize contingency table.
            tpr = np.zeros(m)
            tnr = np.zeros(m)
            ppv = np.zeros(m)
            npv = np.zeros(m)

            for j in range(m):
                if tp[j] + fn[j]:
                    tpr[j] = float(tp[j]) / float(tp[j] + fn[j])
                else:
                    tpr[j] = 1
                if fp[j] + tn[j]:
                    tnr[j] = float(tn[j]) / float(fp[j] + tn[j])
                else:
                    tnr[j] = 1
                if tp[j] + fp[j]:
                    ppv[j] = float(tp[j]) / float(tp[j] + fp[j])
                else:
                    ppv[j] = 1
                if fn[j] + tn[j]:
                    npv[j] = float(tn[j]) / float(fn[j] + tn[j])
                else:
                    npv[j] = 1

            # Compute AUROC as the area under a piecewise linear function with TPR /
            # sensitivity (x-axis) and TNR / specificity (y-axis) and AUPRC as the area
            # under a piecewise constant with TPR / recall (x-axis) and PPV / precision
            # (y-axis).

            for j in range(m - 1):
                auroc_l[k] += 0.5 * (tpr[j + 1] - tpr[j]) * (tnr[j + 1] + tnr[j])
                auprc_l[k] += (tpr[j + 1] - tpr[j]) * ppv[j + 1]

        for i in range(self.num_classes):
            auroc += auroc_l[i] * C_l[i]
            auprc += auprc_l[i] * C_l[i]

        auroc = float(auroc) / float(self.num_classes)
        auprc = float(auprc) / float(self.num_classes)

        return auroc, auprc

    def get_pred(self, outputs, alpha=0.5):
        for i in range(outputs.shape[0]):
            for j in range(outputs.shape[1]):
                if outputs[i, j] >= alpha:
                    outputs[i, j] = 1
                else:
                    outputs[i, j] = 0
        return outputs


if __name__ == '__main__':
    target = torch.tensor([[0, 0, 1, 1], [1, 0, 1, 1]])
    pred = torch.tensor([[0.01, 0.3, 0.9, 0.1], [0.6, 0.1, 0.5, 0.8]])
    acc = accuracy(pred, target)
    print('test')
