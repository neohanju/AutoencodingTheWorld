import os
import sys
import subprocess as sp
from PIL import Image
import numpy as np
import glob

FFMPEG_BIN = "ffmpeg"
target_rows = 255
target_cols = 255
def make_dir(path):
    # if there is no directory, make a directory.
    if not os.path.exists(path):
        os.makedirs(path)
    return


print('Extract images...')
#back young ju (01104575) 19 Jun 13_1
filename = "video_train.mpg"
folder_path ="/home/leejeyeol/Data/endoscope_only"

# output directoryd
make_dir(os.path.join(folder_path, os.path.splitext(filename)[0]))
print(os.path.join(folder_path, os.path.splitext(filename)[0]))

# run ffmpeg command
print('\tFrom: ' + filename)
command = [FFMPEG_BIN,
           '-i', os.path.join(folder_path, filename),
           '-s', str(target_rows) + 'x' + str(target_cols),  # [rows x cols]
           '-pix_fmt', 'rgb24',
           os.path.join(folder_path, os.path.splitext(filename)[0], 'frame_%07d.png')]
sp.call(command)  # call command
print("Extraction is done")

