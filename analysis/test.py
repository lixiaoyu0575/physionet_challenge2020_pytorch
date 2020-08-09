import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
losses = np.load('losses.npy')
indices = loadmat('../evaluation/scored_classes_indices.mat')['val']
indices = indices.reshape([indices.shape[1],]).astype(bool)
losses = losses[:, indices]
# losses = losses.reshape((len(losses)*24))

# #for samples
# sample_losses = np.sum(losses, axis=1)
# plt.hist(sample_losses, bins=200, color='blue')
# plt.show()

#for class
class_loss = np.sum(losses.T, axis=1)
plt.hist(class_loss, bins=200, color='blue')
plt.show()

# for i in range(108):
#     losses_per_sample = np.sum(losses[:, i:i+1], axis=1)
#     # plt.hist(losses[0], bins=200, color='blue')
#     # plt.show()
#     plt.hist(losses_per_sample, bins=200, color='blue')
# #     plt.show()
# plt.hist(losses, bins=200, color='blue')
# plt.show()

print('done')