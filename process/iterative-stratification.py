from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import numpy as np
from process.util import load_challenge_data, load_labels, load_label_files
import matplotlib.pyplot as plt
from scipy.io import savemat
import os

save_path = './data_split'

def datasets_distribution(labels_int, indexs):
   num_of_bins = 108
   fig, axs = plt.subplots(len(indexs), 1, sharey=True, figsize=(50, 50))
   for i in range(len(indexs)):
      subdataset = list()
      for j in indexs[i]:
         for k in labels_int[j]:
            subdataset.append(k)
      subdataset = np.array(subdataset)
      axs[i].hist(subdataset, bins=num_of_bins)

   plt.savefig(os.path.join(save_path, 'split1.png'))
   plt.show()

def datasets_distribution2(labels_int, indexs):
   freqs = []
   num_of_bins = 108
   bins = range(1, num_of_bins + 2)
   fig, axs = plt.subplots(len(indexs), 1, sharey=True, figsize=(100, 100))
   for i in range(len(indexs)):
      subdataset = list()
      for j in indexs[i]:
         for k in labels_int[j]:
            subdataset.append(k)
      subdataset = np.array(subdataset)
      freq, bin, _ = axs[i].hist(subdataset, bins=bins, align='left')
      freqs.append(freq)

   return freqs

# Define the weights, the SNOMED CT code for the normal class, and equivalent SNOMED CT codes.
weights_file = 'weights.csv'
normal_class = '426783006'
equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]

input_directory_label = '/DATASET/challenge2020/All_data'

# Find the label files.
print('Finding label and output files...')
label_files = load_label_files(input_directory_label)

# Load the labels and classes.
print('Loading labels and outputs...')
label_classes, labels_onehot, labels = load_labels(label_files, normal_class, equivalent_classes)

temp = [[] for _ in range(len(labels_onehot))]
indexes, values = np.where(np.array(labels_onehot).astype(int) == 1)
for k, v in zip(indexes, values):
   temp[k].append(v)
labels_int = temp

X = np.zeros(len(labels_onehot))
y = labels_onehot

msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)

for train_index, test_index in msss.split(X, y):
   print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]

   msss_val = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=1/9, random_state=0)

   for index_1, index_2 in msss_val.split(X_train, y_train):
      print("TRAIN:", index_1, "Valid:", index_2)
      X_train, X_val = X_train[index_1], X_train[index_2]
      y_train, y_val = y_train[index_1], y_train[index_2]

      val_index = train_index[index_2]
      train_index = train_index[index_1]

      datasets_distribution(labels_int, [train_index, val_index, test_index])

      freqs = datasets_distribution2(labels_int, [train_index, val_index, test_index])

      print(freqs[0]/freqs[1])
      print(freqs[0]/freqs[2])

      print("TRAIN:", train_index)
      print("Valid:", val_index)
      print("Test:", test_index)

      savemat(os.path.join(save_path, 'split1.mat'), {'train_index':train_index, 'val_index': val_index, 'test_index': test_index})



