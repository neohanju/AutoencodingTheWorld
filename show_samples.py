# show samples with their ground truth mask
# how to use:
#  1) import this module
#  2) call 'get_sample' function with sample index (not a filename, index of sorted file path)
#

import os
import glob
import numpy as np
import PathManager as pm
from matplotlib import pyplot as plt

# get folder paths
train_augmented_path = os.path.join(pm.datasetroot, 'train_augmented')
gt_augmented_path = os.path.join(pm.datasetroot, 'Ground_Truth_augmented')

# get sample paths
aug_sample_paths = glob.glob(os.path.join(train_augmented_path, '*.npy'))
aug_sample_paths.sort()
aug_gt_paths = glob.glob(os.path.join(gt_augmented_path, '*.npy'))
aug_gt_paths.sort()


def get_sample(sample_number):
    print('load ' + aug_sample_paths[sample_number])
    sample = np.load(aug_sample_paths[sample_number])
    gt = np.load(aug_gt_paths[sample_number])
    fig = plt.figure(sample_number)
    ax1 = fig.add_subplot(121)
    ax1.imshow(sample)
    ax2 = fig.add_subplot(122)
    ax2.imshow(gt)
    plt.show()

#()()
#('')HAANJU.YOO
