import os
import sys
import subprocess as sp
from PIL import Image
import numpy as np
import glob
# =============================================================================
# SET DATASET LIST
# =============================================================================
def make_dir(path):
    # if there is no directory, make a directory.
    if not os.path.exists(path):
        os.makedirs(path)
    return

def get_file_paths(path, file_type):
    # return file list of the given type in the given path.
    file_paths = []
    if not os.path.exists(path):
        return file_paths
    for extension in file_type:
        file_paths += glob.glob(path + '/*.' + extension)
    file_paths.sort()
    return file_paths


# =============================================================================
# DATA PREPARATION
# =============================================================================
FFMPEG_BIN = "ffmpeg"
DATASET_BASE_PATH = "/mnt/fastdataset/Datasets"
#DATASET_BASE_PATH = os.environ['YCL_DATA_ROOT']
dataset = 'avenue'
dataset_type = 'test'

dataset_path = os.path.join(DATASET_BASE_PATH, dataset, dataset_type)

stride_list = get_file_paths(dataset_path, ['npy'])

for i, stride in stride_list:
    stride_array = np.load(stride)
    # when stride name is 00000.npy, save 0~9 images.
    # otherwise, save 4~9 images.




print('gg');


# load npy  = stride (consist of ten frames)
# if file name is 00000.npy, this file is convert to 10 frame
# otherwise, files are convert to 5 frame. these frames are added to list.
# when all files are loaded and converted, list is converted to video by FFMPEG.

