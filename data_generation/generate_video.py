import os
import sys
import subprocess as sp
from PIL import Image
import numpy as np
import glob

FFMPEG_BIN = "ffmpeg"
DATASET_BASE_PATH = os.environ['YCL_DATA_ROOT']
dataset = 'avenue'
dataset_type = 'test'

# dataset_path = fullfile(DATASET_BASE_PATH, dataset, dataset_type)

frame_list = glob.glob(dataset_path)
frame_list.sort()

# load npy  = stride (consist of ten frames)
# if file name is 00000.npy, this file is convert to 10 frame
# otherwise, files are convert to 5 frame. these frames are added to list.
# when all files are loaded and converted, list is converted to video by FFMPEG.

