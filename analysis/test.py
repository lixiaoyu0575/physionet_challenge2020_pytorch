import numpy as np
import matplotlib.pyplot as plt
losses = np.load('test_losses.npy')
for i in range(108):
    losses_per_sample = np.sum(losses[:, i:i+1], axis=1)
    # plt.hist(losses[0], bins=200, color='blue')
    # plt.show()
    plt.hist(losses_per_sample, bins=200, color='blue')
    plt.show()

print('done')