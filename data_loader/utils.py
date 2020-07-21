from scipy.io import loadmat
import numpy as np
import os

# Find unique classes.
def get_classes(input_directory, filenames):
    classes = set()
    for filename in filenames:
        with open(filename, 'r') as f:
            for l in f:
                if l.startswith('#Dx'):
                    tmp = l.split(': ')[1].split(',')
                    for c in tmp:
                        classes.add(c.strip())
    return sorted(classes)


# Load challenge data.
def load_challenge_data(label_file, data_dir):
    file = os.path.basename(label_file)
    with open(label_file, 'r') as f:
        header = f.readlines()
    mat_file = file.replace('.hea', '.mat')
    x = loadmat(os.path.join(data_dir, mat_file))
    recording = np.asarray(x['val'], dtype=np.float64)
    return recording, header
