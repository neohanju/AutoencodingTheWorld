import os
import sys
from PIL import Image

target_rows = 227
target_cols = 227
target_length = 10
frame_stride = 2
sample_stride = 2


datasets = {}
datasets['avenue_train'] = {
    'path': '/home/neohanju/Workspace/dataset/abnormal_event_detection/CUHK_avenue/training_videos',
    'name_format': '%02d.avi',
    'num_videos': 16
}
datasets['avenue_test'] = {
    'path': '/home/neohanju/Workspace/dataset/abnormal_event_detection/CUHK_avenue/testing_video    s',
    'name_format': '%02d.avi',
    'num_videos': 21
}

# =============================================================================
# EXTRACT FRAMES FROM VIDEOS
# =============================================================================




# =============================================================================
# GENERATE SAMPLES WITH IMAGES
# =============================================================================