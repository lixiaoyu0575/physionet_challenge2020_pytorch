# -*- coding: utf-8 -*-
# @Author  : Light--

"""
Description：use ml-ros to balance celeba train set
Similar repo: https://github.com/wtomin/Multitask-Emotion-Recognition-with-Incomplete-Labels/blob/0cedf4ca8e117fe5f2950d58af67c1e4cc831f95/create_annotation_file/DISFA/create_annotation_Mixed_AU.py
"""
import numpy as np
import time
from process.utils import load_label_files, load_labels, load_challenge_data

def IRLbl(labels):
    # imbalance ratio per label
    # Args:
    #	 labels is a 2d numpy array, each row is one instance, each column is one class; the array contains (0, 1) only
    N, C = labels.shape
    pos_nums_per_label = np.sum(labels, axis=0)
    max_pos_nums = np.max(pos_nums_per_label)
    return max_pos_nums / pos_nums_per_label

def MeanIR(labels):
    IRLbl_VALUE = IRLbl(labels)
    return np.mean(IRLbl_VALUE)

def ML_ROS(all_labels, indices=None, num_samples=None, Preset_MeanIR_value=2.,
                 max_clone_percentage=50, sample_size=32):
    # the index of samples: 0, 1, ....
    # if indices is not provided,
    # all elements in the dataset will be considered
    indices = list(range(len(all_labels))) \
        if indices is None else indices

    # if num_samples is not provided,
    # draw `len(indices)` samples in each iteration
    num_samples = len(indices) \
        if num_samples is None else num_samples

    MeanIR_value = MeanIR(all_labels) if Preset_MeanIR_value == 0 else Preset_MeanIR_value
    IRLbl_value = IRLbl(all_labels)
    # N is the number of samples, C is the number of labels
    N, C = all_labels.shape
    # the samples index of every class
    indices_per_class = {}
    minority_classes = []
    # accroding to psedu code, maxSamplesToClone is the upper limit of the number of samples can be copied from original dataset
    maxSamplesToClone = N / 100 * max_clone_percentage
    print('Max Clone Limit:', maxSamplesToClone)
    for i in range(C):
        ids = all_labels[:, i] == 1
        # How many samples are there for each label
        indices_per_class[i] = [ii for ii, x in enumerate(ids) if x]
        if IRLbl_value[i] > MeanIR_value:
            minority_classes.append(i)

    new_all_labels = all_labels
    oversampled_ids = []
    minorNum = len(minority_classes)
    print(minorNum, 'minor classes.')

    for idx, i in enumerate(minority_classes):
        tid = time.time()
        while True:
            pick_id = list(np.random.choice(indices_per_class[i], sample_size))
            indices_per_class[i].extend(pick_id)
            # recalculate the IRLbl_value
            # The original label matrix (New_ all_ Labels) and randomly selected label matrix (all_ labels[pick_ ID) and recalculate the irlbl
            new_all_labels = np.concatenate([new_all_labels, all_labels[pick_id]], axis=0)
            oversampled_ids.extend(pick_id)

            newIrlbl = IRLbl(new_all_labels)
            if newIrlbl[i] <= MeanIR_value:
                print('\nMeanIR satisfied.', newIrlbl[i])
                break
            if len(oversampled_ids) >= maxSamplesToClone:
                print('\nExceed max clone.', len(oversampled_ids))
                break
            # if IRLbl(new_all_labels)[i] <= MeanIR_value or len(oversampled_ids) >= maxSamplesToClone:
            #     break
            print("\roversample length:{}".format(len(oversampled_ids)), end='')
        print('Processed the %d/%d minor class:' % (idx+1, minorNum), i, time.time()-tid, 's')
        if len(oversampled_ids) >= maxSamplesToClone:
            print('Exceed max clone. Exit', len(oversampled_ids))
            break
    return new_all_labels, oversampled_ids

# get all sample labels from celeba txt annotation file
def get_all_labels(root, annFile, ):
    import os
    targets = []
    images = []

    for line in open(os.path.join(root, annFile), 'r'):
        sample = line.split()
        # print(sample)
        if len(sample) != 41:
            raise (RuntimeError("# Annotated face attributes of CelebA dataset should not be different from 40"))
        images.append(sample[0])
        # target = [int(i) for i in sample[1:]] # -1 label will cause error
        target = [0 if int(i) == -1 else int(i) for i in sample[1:]]
        targets.append(target)
    targets = np.array(targets)
    print('All labels in the dataset:', targets.shape)
    return targets

# Copy samples that need to be oversampled, according to new txt annotation file generated after ml-ros processed
def copy_samples(root, annFile, newAnnFile, copyIds):
    import os
    from shutil import copyfile
    import sys

    # adding exception handling
    srcFile = os.path.join(root, annFile)
    dstFile = os.path.join(root, newAnnFile)
    # copy original file to new file
    try:
        copyfile(srcFile, dstFile)
    except IOError as e:
        print("Unable to copy file. %s" % e)
        exit(1)
    except:
        print("Unexpected error:", sys.exc_info())
        exit(1)

    # copy samples to new file
    copied = 0
    txt = np.loadtxt(srcFile, dtype=str)
    copyNum = len(copyIds)
    txt = txt.tolist()

    for lineId in copyIds:
        newAnnFile = open(dstFile, 'a')
        line = txt[lineId]
        line = ' '.join(line) + '\n'
        newAnnFile.write(line)
        copied += 1
        print('\rCopied %d/%d' % (copied, copyNum), end='')
    print('\nUpsampling Done. ', dstFile)

# use ml-ros to process celeba dataset, reduce the class imbalance
def mlros_celeba():
    celebaRoot = r'/data2/xxx/tmp_data/celeba_mlros/annot'
    annFile = 'train_40_att_list.txt'
    allLabels = get_all_labels(celebaRoot, annFile,)
    print('Origianl:', len(allLabels))
    t1 = time.time()
    newLables, oversampleIds = ML_ROS(allLabels, )
    print('New:', len(newLables), time.time()-t1, 's')
    # generate new train.txt
    newAnnFile = 'mlros_' + annFile
    copy_samples(celebaRoot, annFile, newAnnFile, oversampleIds)

def mlros_challenge2020():

    # Define the weights, the SNOMED CT code for the normal class, and equivalent SNOMED CT codes.
    weights_file = 'weights.csv'
    normal_class = '426783006'
    equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]

    input_directory = '/home/weiyuhua/Data/challenge2020'
    save_path = '/home/weiyuhua/Data/challenge2020_plots'

    # Find the label files.
    print('Finding label and output files...')
    label_files = load_label_files(input_directory)

    # Load the labels and classes.
    print('Loading labels and outputs...')
    label_classes, labels_onehot, labels = load_labels(label_files, normal_class, equivalent_classes)

    print('Origianl:', len(labels_onehot))
    t1 = time.time()
    newLables, oversampleIds = ML_ROS(labels_onehot, Preset_MeanIR_value=100)
    print('New:', len(newLables), time.time()-t1, 's')
    # # generate new train.txt
    # newAnnFile = 'mlros_' + annFile
    # copy_samples(celebaRoot, annFile, newAnnFile, oversampleIds)

if __name__ == '__main__':
    mlros_challenge2020()

