import os
import json
import numpy as np
from numpy import inf
from scipy.io import loadmat
import torch
from torch.utils.data import TensorDataset
import pandas as pd
from pathlib import Path
import logging

import random

# challenge 2020

def ChallengeData(label_dir, data_dir, weights_file):
    print('Loading data...')

    weights_file = 'evaluation/weights.csv'
    normal_class = '426783006'
    equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]

    # Find the label files.
    print('Finding label...')
    label_files = load_label_files(label_dir)

    # Load the labels and classes.
    print('Loading labels...')
    classes, labels_onehot, labels = load_labels(label_files, normal_class, equivalent_classes)

    # Load the weights for the Challenge metric.
    print('Loading weights...')
    weights = load_weights(weights_file, classes)

    # Classes that are scored with the Challenge metric.
    indices = np.any(weights, axis=0)  # Find indices of classes in weight matrix.
    # from scipy.io import savemat
    # savemat('evaluation/scored_classes_indices.mat', {'val': indices})

    # Load short signals and remove from labels
    # short_signals = loadmat(os.path.join(data_dir, 'short_signals.mat'))['val']
    # short_signals_ids = list(short_signals.reshape((short_signals.shape[1], )))

    num_files = len(label_files)
    recordings = list()
    labels_onehot_new = list()
    labels_new = list()
    file_names = list()

    bb = []
    dd = []

    for i in range(num_files):
        # if i in short_signals_ids:
        #     recording = np.zeros((1, 12, 3000))
        #
        # else:
        recording, header, name = load_challenge_data(label_files[i], data_dir)
        recording[np.isnan(recording)] = 0
        recordings.append(recording)
        file_names.append(name)

        rr = np.array(recording)
        if np.isnan(rr).any():
            print(i)
            bb.append(i)
            dd.append(rr)

        labels_onehot_new.append(labels_onehot[i])
        labels_new.append(labels[i])

    for i in range(len(recordings)):
        if np.isnan(recordings[i]).any():
            print(i)

    # slided data
    recordings_all = list()
    labels_onehot_all = list()
    labels_all = list()

    for i in range(len(recordings)):
        for j in range(recordings[i].shape[0]):
            recordings_all.append(recordings[i][j])
            labels_onehot_all.append(labels_onehot_new[i])
            labels_all.append(labels_new[i])

    recordings_all = np.array(recordings_all)
    labels_onehot_all = np.array(labels_onehot_all)

    recordings_preprocessed, labels_onehot = preprocessing(recordings_all, labels_onehot_all)
    recordings_augmented, labels_onehot = augmentation(recordings_preprocessed, labels_onehot_all)

    # labels_onehot = np.array(labels_onehot, dtype='float64')
    # labels_onehot = smooth_labels(labels_onehot)
    print(np.isnan(recordings_augmented).any())

    num = recordings_augmented.shape[0]
    c = []
    a = []
    for i in range(num):
        if np.isnan(recordings_augmented[i]).any():
            print(' {}/{}'.format(i, num))
            c.append(i)
            a.append(recordings_augmented[i])
    print(c)
    print(a)

    # Get number of samples for each category
    count = np.sum(labels_onehot, axis=0)
    indices = indices

    X = torch.from_numpy(recordings_augmented).float()
    Y = torch.from_numpy(labels_onehot.astype(int))
    dataset = TensorDataset(X, Y)

    return dataset, indices

def preprocessing(recordings, labels):

    return recordings, labels

def augmentation(recordings, labels):

    recordings_augmented = recordings
    return recordings_augmented, labels

def is_number(x):
    try:
        float(x)
        return True
    except ValueError:
        return False

# Find Challenge files.
def load_label_files(label_directory):
    label_files = list()
    for f in sorted(os.listdir(label_directory)):
        F = os.path.join(label_directory, f) # Full path for label file
        if os.path.isfile(F) and F.lower().endswith('.hea') and not f.lower().startswith('.'):
            # root, ext = os.path.splitext(f)
            label_files.append(F)
    if label_files:
        return label_files
    else:
        raise IOError('No label or output files found.')

# Load labels from header/label files.
def load_labels(label_files, normal_class, equivalent_classes_collection):
    # The labels_onehot should have the following form:
    #
    # Dx: label_1, label_2, label_3
    #
    num_recordings = len(label_files)

    # Load diagnoses.
    tmp_labels = list()
    for i in range(num_recordings):
        with open(label_files[i], 'r') as f:
            for l in f:
                if l.startswith('#Dx'):
                    dxs = set(arr.strip() for arr in l.split(': ')[1].split(','))
                    tmp_labels.append(dxs)

    # Identify classes.
    classes = set.union(*map(set, tmp_labels))
    if normal_class not in classes:
        classes.add(normal_class)
        print('- The normal class {} is not one of the label classes, so it has been automatically added, but please check that you chose the correct normal class.'.format(normal_class))
    classes = sorted(classes)
    num_classes = len(classes)

    # Use one-hot encoding for labels.
    labels_onehot = np.zeros((num_recordings, num_classes), dtype=np.bool)
    for i in range(num_recordings):
        dxs = tmp_labels[i]
        for dx in dxs:
            j = classes.index(dx)
            labels_onehot[i, j] = 1

    # For each set of equivalent class, use only one class as the representative class for the set and discard the other classes in the set.
    # The label for the representative class is positive if any of the labels_onehot in the set is positive.
    remove_classes = list()
    remove_indices = list()
    for equivalent_classes in equivalent_classes_collection:
        equivalent_classes = [x for x in equivalent_classes if x in classes]
        if len(equivalent_classes)>1:
            representative_class = equivalent_classes[0]
            other_classes = equivalent_classes[1:]
            equivalent_indices = [classes.index(x) for x in equivalent_classes]
            representative_index = equivalent_indices[0]
            other_indices = equivalent_indices[1:]

            labels_onehot[:, representative_index] = np.any(labels_onehot[:, equivalent_indices], axis=1)
            remove_classes += other_classes
            remove_indices += other_indices

    for x in remove_classes:
        classes.remove(x)
    labels_onehot = np.delete(labels_onehot, remove_indices, axis=1)

    # If the labels_onehot are negative for all classes, then change the label for the normal class to positive.
    normal_index = classes.index(normal_class)
    for i in range(num_recordings):
        num_positive_classes = np.sum(labels_onehot[i, :])
        if num_positive_classes==0:
            labels_onehot[i, normal_index] = 1

    labels = list()
    for i in range(num_recordings):
        class_list = []
        for j in range(len(classes)):
            if labels_onehot[i][j] == True:
                class_list.append(classes[j])
        class_set = set()
        class_set.update(class_list)
        labels.append(class_set)

    return classes, labels_onehot, labels

# Load challenge data.
def load_challenge_data(label_file, data_dir):
    file = os.path.basename(label_file)
    name, ext = os.path.splitext(file)
    with open(label_file, 'r') as f:
        header = f.readlines()
    mat_file = file.replace('.hea', '.mat')
    x = loadmat(os.path.join(data_dir, mat_file))
    recording = np.asarray(x['val'], dtype=np.float64)
    return recording, header, name

# Load weights.
def load_weights(weight_file, classes):
    # Load the weight matrix.
    rows, cols, values = load_table(weight_file)
    assert(rows == cols)
    num_rows = len(rows)

    # Assign the entries of the weight matrix with rows and columns corresponding to the classes.
    num_classes = len(classes)
    weights = np.zeros((num_classes, num_classes), dtype=np.float64)
    for i, a in enumerate(rows):
        if a in classes:
            k = classes.index(a)
            for j, b in enumerate(rows):
                if b in classes:
                    l = classes.index(b)
                    weights[k, l] = values[i, j]

    return weights

# Load_table
def load_table(table_file):
    # The table should have the following form:
    #
    # ,    a,   b,   c
    # a, 1.2, 2.3, 3.4
    # b, 4.5, 5.6, 6.7
    # c, 7.8, 8.9, 9.0
    #
    table = list()
    print(os.getcwd())
    with open(table_file, 'r') as f:
        for i, l in enumerate(f):
            arrs = [arr.strip() for arr in l.split(',')]
            table.append(arrs)

    # Define the numbers of rows and columns and check for errors.
    num_rows = len(table)-1
    if num_rows<1:
        raise Exception('The table {} is empty.'.format(table_file))

    num_cols = set(len(table[i])-1 for i in range(num_rows))
    if len(num_cols)!=1:
        raise Exception('The table {} has rows with different lengths.'.format(table_file))
    num_cols = min(num_cols)
    if num_cols<1:
        raise Exception('The table {} is empty.'.format(table_file))

    # Find the row and column labels.
    rows = [table[0][j+1] for j in range(num_rows)]
    cols = [table[i+1][0] for i in range(num_cols)]

    # Find the entries of the table.
    values = np.zeros((num_rows, num_cols))
    for i in range(num_rows):
        for j in range(num_cols):
            value = table[i+1][j+1]
            if is_number(value):
                values[i, j] = float(value)
            else:
                values[i, j] = float('nan')

    return rows, cols, values

def init_obj(hype_space, name, module, *args, **kwargs):
    """
    Finds a function handle with the name given as 'type' in config, and returns the
    instance initialized with corresponding arguments given.
    """
    module_name = hype_space[name]['type']
    module_args = dict(hype_space[name]['args'])
    assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
    module_args.update(kwargs)
    return getattr(module, module_name)(*args, **module_args)

def to_np(tensor, device):
    if device.type == 'cuda':
        return tensor.cpu().detach().numpy()
    else:
        return tensor.detach().numpy()

def get_mnt_mode(trainer):
    monitor = trainer.get('monitor', 'off')
    if monitor == 'off':
        mnt_mode = 'off'
        mnt_best = 0
        early_stop = 0
        mnt_metric_name = None
    else:
        mnt_mode, mnt_metric_name = monitor.split()
        assert mnt_mode in ['min', 'max']
        mnt_best = inf if mnt_mode == 'min' else -inf
        early_stop = trainer.get('early_stop', inf)

    return mnt_metric_name, mnt_mode, mnt_best, early_stop

def save_checkpoint(model, epoch, optimizer, mnt_best, config, checkpoint_dir, save_best=False):
    arch = type(model).__name__
    state = {
        'arch': arch,
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'monitor_best': mnt_best,
        'config': config
    }
    # filename = checkpoint_dir + '/checkpoint-epoch{}.pth'.format(epoch)
    # torch.save(state, filename)
    # print("Saving checkpoint: {} ...".format(filename))
    if save_best:
        best_path = checkpoint_dir + '/model_best.pth'
        torch.save(state, best_path)
        print("Saving current best: model_best.pth ...")

def analyze(model, test_data_loader, criterion, challenge_metrics, logger, result_dir, device=None):
    """
    Analyze after training procedure finished
    """
    model.eval()

    outputs = torch.zeros((test_data_loader.n_samples, model.num_classes))
    targets = torch.zeros((test_data_loader.n_samples, model.num_classes))

    with torch.no_grad():
        start = 0
        for batch_idx, (data, target) in enumerate(test_data_loader):
            end = len(data) + start
            data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.float)

            output = model(data)

            outputs[start:end, :] = output
            targets[start:end, :] = target
            start = end

            loss = criterion(output, target)

    indices = challenge_metrics.indices
    classes = challenge_metrics.classes
    test_idx = test_data_loader.idx
    file_names = test_data_loader.file_names

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
    logger.info("**********************************************************************************")
    logger.info("All Testing Data")

    outputs_logit = torch.nn.Sigmoid()(outputs)
    accuracy = challenge_metrics.accuracy(to_np(outputs_logit, device), to_np(targets, device))
    macro_f_measure, f_measure  = challenge_metrics.f_measure(to_np(outputs_logit, device), to_np(targets, device))
    macro_f_beta_measure, macro_g_beta_measure, f_beta_measure, g_beta_measure = challenge_metrics.beta_measures(to_np(outputs_logit, device), to_np(targets, device))
    macro_auroc, macro_auprc, auroc, auprc = challenge_metrics.auc(to_np(outputs_logit, device), to_np(targets, device))
    challenge_metric = challenge_metrics.challenge_metric(to_np(outputs_logit, device), to_np(targets, device))

    logger.info("challenge_metric:{}".format(challenge_metric))
    logger.info("accuracy:{}".format(accuracy))
    logger.info("macro_f_measure:{}".format(macro_f_measure))
    logger.info("macro_g_beta_measure:{}".format(macro_g_beta_measure))
    logger.info("macro_auroc:{}".format(macro_auroc))
    logger.info("macro_auprc:{}".format(macro_auprc))

    metrics = np.vstack((f_measure, f_beta_measure, g_beta_measure, auroc, auprc))
    metrics_df = pd.DataFrame(data=metrics, columns=classes, index=['f_measure', 'f_beta_measure', 'g_beta_measure', 'auroc', 'auprc'])
    logger.info(metrics_df)
    metrics_df.to_csv(result_dir+'/All.csv')

    Dataset = ['CPSC 2018 data', 'Unused CPSC2018 data', 'PTB Diagnostic ECG Database', 'PTB-XL electrocardiography Database',
               'Georgia 12-Lead ECG Challenge Database', 'St Petersburg INCART 12-lead Arrhythmia Database']

    for i in range(len(Dataset)):
        outputs_i_logit = to_np(outputs_logit, device)[dataset_idx_list[i]]
        targets_i = to_np(targets, device)[dataset_idx_list[i]]

        accuracy = challenge_metrics.accuracy(outputs_i_logit, targets_i)
        macro_f_measure, f_measure = challenge_metrics.f_measure(outputs_i_logit, targets_i)
        macro_f_beta_measure, macro_g_beta_measure, f_beta_measure, g_beta_measure = challenge_metrics.beta_measures(
            outputs_i_logit, targets_i)
        macro_auroc, macro_auprc, auroc, auprc = challenge_metrics.auc(outputs_i_logit, targets_i)
        challenge_metric = challenge_metrics.challenge_metric(outputs_i_logit, targets_i)

        logger.info("**********************************************************************************")
        logger.info(Dataset[i])
        logger.info("challenge_metric:{}".format(challenge_metric))
        logger.info("accuracy:{}".format(accuracy))
        logger.info("macro_f_measure:{}".format(macro_f_measure))
        logger.info("macro_g_beta_measure:{}".format(macro_g_beta_measure))
        logger.info("macro_auroc:{}".format(macro_auroc))
        logger.info("macro_auprc:{}".format(macro_auprc))

        metrics = np.vstack((f_measure, f_beta_measure, g_beta_measure, auroc, auprc))
        metrics_df = pd.DataFrame(data=metrics, columns=classes,
                                  index=['f_measure', 'f_beta_measure', 'g_beta_measure', 'auroc', 'auprc'])
        logger.info(metrics_df)
        metrics_df.to_csv(result_dir + '/' + Dataset[i] + '.csv')

def analyze2(model, test_data_loader, criterion, challenge_metrics, logger, result_dir, device=None):
    """
    Analyze after training procedure finished
    """
    model.eval()

    Outputs = torch.zeros((test_data_loader.n_samples, model.num_classes))
    Targets = torch.zeros((test_data_loader.n_samples, model.num_classes))

    with torch.no_grad():
        start = 0
        for batch_idx, (data, target) in enumerate(test_data_loader):
            end = len(data) + start

            outputs = torch.zeros(len(target), model.num_classes)
            targets = torch.zeros(len(target), model.num_classes)

            for i in range(len(data)):
                data[i], target[i] = data[i].to(device), target[i].to(device)
                output = model(data[i])

                outputs[i:i + 1, :] = output
                targets[i:i + 1, :] = target[i]

            Outputs[start:end, :] = outputs
            Targets[start:end, :] = targets
            start = end

    indices = challenge_metrics.indices
    classes = challenge_metrics.classes
    test_idx = test_data_loader.idx
    file_names = test_data_loader.file_names

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
    logger.info("**********************************************************************************")
    logger.info("All Testing Data")

    Outputs_logit = torch.nn.Sigmoid()(Outputs)
    accuracy = challenge_metrics.accuracy(to_np(Outputs_logit, device), to_np(Targets, device))
    macro_f_measure, f_measure  = challenge_metrics.f_measure(to_np(Outputs_logit, device), to_np(Targets, device))
    macro_f_beta_measure, macro_g_beta_measure, f_beta_measure, g_beta_measure = challenge_metrics.beta_measures(to_np(Outputs_logit, device), to_np(Targets, device))
    macro_auroc, macro_auprc, auroc, auprc = challenge_metrics.auc(to_np(Outputs_logit, device), to_np(Targets, device))
    challenge_metric = challenge_metrics.challenge_metric(to_np(Outputs_logit, device), to_np(Targets, device))

    logger.info("challenge_metric:{}".format(challenge_metric))
    logger.info("accuracy:{}".format(accuracy))
    logger.info("macro_f_measure:{}".format(macro_f_measure))
    logger.info("macro_g_beta_measure:{}".format(macro_g_beta_measure))
    logger.info("macro_auroc:{}".format(macro_auroc))
    logger.info("macro_auprc:{}".format(macro_auprc))

    metrics = np.vstack((f_measure, f_beta_measure, g_beta_measure, auroc, auprc))
    metrics_df = pd.DataFrame(data=metrics, columns=classes, index=['f_measure', 'f_beta_measure', 'g_beta_measure', 'auroc', 'auprc'])
    logger.info(metrics_df)
    metrics_df.to_csv(result_dir+'/All.csv')

    Dataset = ['CPSC 2018 data', 'Unused CPSC2018 data', 'PTB Diagnostic ECG Database', 'PTB-XL electrocardiography Database',
               'Georgia 12-Lead ECG Challenge Database', 'St Petersburg INCART 12-lead Arrhythmia Database']

    for i in range(len(Dataset)):
        outputs_i_logit = to_np(Outputs_logit, device)[dataset_idx_list[i]]
        targets_i = to_np(Targets, device)[dataset_idx_list[i]]

        accuracy = challenge_metrics.accuracy(outputs_i_logit, targets_i)
        macro_f_measure, f_measure = challenge_metrics.f_measure(outputs_i_logit, targets_i)
        macro_f_beta_measure, macro_g_beta_measure, f_beta_measure, g_beta_measure = challenge_metrics.beta_measures(
            outputs_i_logit, targets_i)
        macro_auroc, macro_auprc, auroc, auprc = challenge_metrics.auc(outputs_i_logit, targets_i)
        challenge_metric = challenge_metrics.challenge_metric(outputs_i_logit, targets_i)

        logger.info("**********************************************************************************")
        logger.info(Dataset[i])
        logger.info("challenge_metric:{}".format(challenge_metric))
        logger.info("accuracy:{}".format(accuracy))
        logger.info("macro_f_measure:{}".format(macro_f_measure))
        logger.info("macro_g_beta_measure:{}".format(macro_g_beta_measure))
        logger.info("macro_auroc:{}".format(macro_auroc))
        logger.info("macro_auprc:{}".format(macro_auprc))

        metrics = np.vstack((f_measure, f_beta_measure, g_beta_measure, auroc, auprc))
        metrics_df = pd.DataFrame(data=metrics, columns=classes,
                                  index=['f_measure', 'f_beta_measure', 'g_beta_measure', 'auroc', 'auprc'])
        logger.info(metrics_df)
        metrics_df.to_csv(result_dir + '/' + Dataset[i] + '.csv')

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def progress(data_loader, batch_idx):
    base = '[{}/{} ({:.0f}%)]'
    if hasattr(data_loader, 'n_samples'):
        current = batch_idx * data_loader.batch_size
        total = data_loader.n_samples
    else:
        current = batch_idx
        total = len(data_loader)
    return base.format(current, total, 100.0 * current / total)

def load_checkpoint(model, resume_path, logger):
    """
    Resume from saved checkpoints

    :param resume_path: Checkpoint path to be resumed
    """
    logger.info("Loading checkpoint: {} ...".format(resume_path))
    checkpoint = torch.load(resume_path)
    epoch = checkpoint['epoch']
    mnt_best = checkpoint['monitor_best']

    # load architecture params from checkpoint.
    model.load_state_dict(checkpoint['state_dict'])

    logger.info("Checkpoint loaded from epoch {}".format(epoch))

    return model