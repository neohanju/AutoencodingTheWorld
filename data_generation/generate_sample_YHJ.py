import os
import sys
import subprocess as sp
from PIL import Image
import numpy as np
import torch
import glob

FFMPEG_BIN = "ffmpeg"
DATASET_BASE_PATH = '/mnt/fastdataset/Datasets'

number_of_images_in_dataset_all = 0

target_rows = 227
target_cols = 227
target_length = 10

frame_stride = 2
sample_stride = 5

target_datasets = ['avenue_train','enter_train','exit_train']

datasets = dict(
    mytestvideo=dict(
        path=os.path.join(DATASET_BASE_PATH, 'Test_JY'),
        name='Test_JY',
        type='train',
        name_format='%02d.avi',
        num_videos=1
    ),
    avenue_train=dict(
        path=os.path.join(DATASET_BASE_PATH, 'avenue'),
        name='avenue',
        type='train',
        name_format='%02d.avi',
        num_videos=16
    ),
    avenue_test=dict(
        path=os.path.join(DATASET_BASE_PATH, 'avenue'),
        name='avenue',
        type='test',
        name_format='%02d.avi',
        num_videos=21
    ),
    enter_train = dict(
        path=os.path.join(DATASET_BASE_PATH, 'enter'),
        name='enter',
        type='train',
        name_format='%02d.avi',
        num_videos=1
    ),
    enter_test = dict(
        path=os.path.join(DATASET_BASE_PATH, 'enter'),
        name='enter',
        type='test',
        name_format='%02d.avi',
        num_videos=6
    ),
    exit_train = dict(
        path=os.path.join(DATASET_BASE_PATH, 'exit'),
        name='exit',
        type='train',
        name_format='%02d.avi',
        num_videos=1
    ),
    exit_test = dict(
        path=os.path.join(DATASET_BASE_PATH, 'exit'),
        name='exit',
        type='test',
        name_format='%02d.avi',
        num_videos=4
    )
)


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
# EXTRACT FRAMES FROM VIDEOS
# =============================================================================
def extract_video_frames():
    # FFMPEG -> frame extraction & rescale & gray scale  & save to disk(png)
    print('Extract images...')
    for name in target_datasets:
        for video in range(1, datasets[name]['num_videos'] + 1):
            filename = datasets[name]['name_format'] % video
            folder_path = os.path.join(datasets[name]['path'], '%sing_videos' % datasets[name]['type'])

            # output directory
            make_dir(os.path.join(folder_path, os.path.splitext(filename)[0]))

            # run ffmpeg command
            print('\tFrom: ' + filename)
            command = [FFMPEG_BIN,
                       '-i', os.path.join(folder_path, filename),
                       '-s', str(target_rows) + 'x' + str(target_cols),  # [rows x cols]
                       '-pix_fmt', 'gray',  # to gray scale
                       os.path.join(folder_path, os.path.splitext(filename)[0], 'frame_%05d.png')]
            sp.call(command)  # call command
    print("Extraction is done")


# make mean images
def get_mean_image():
    print('Get mean image...')
    for name in target_datasets:
        if datasets[name]['type'] is not 'train':
            continue

        # mean image over all videos in the dataset
        print('\tCalculate mean image with ' + name + '...')
        mean_image = np.zeros((target_rows, target_cols), dtype=np.float)  # mean image container
        count_image = 0.0
        for video in range(1, datasets[name]['num_videos'] + 1):
            sys.stdout.write('\t\tWith %s ... ' % datasets[name]['name_format'] % video)
            image_folder = os.path.join(datasets[name]['path'], '%sing_videos' % datasets[name]['type'],
                                        os.path.splitext(datasets[name]['name_format'] % video)[0])
            image_files = get_file_paths(image_folder, ['png', 'PNG'])
            for path in image_files:
                mean_image += np.array(Image.open(path), dtype=np.float)
                count_image += 1.0
            print('done!')

        assert count_image > 0.0
        mean_image /= count_image   # make mean image.

        # save mean image .npy and .png
        np.save(os.path.join(datasets[name]['path'], 'mean_image'), mean_image)
        Image.fromarray(np.uint8(mean_image)).save(os.path.join(datasets[name]['path'], 'mean_image.png'))

    print('Getting mean image is done')


# =============================================================================
# GENERATE SAMPLES
# =============================================================================
def generate_samples(preproc_type='centering'):
    # sample data container
    sample_data = np.zeros((target_length, target_rows, target_cols), dtype=np.float)
    for i, name in enumerate(target_datasets):
        print('Generate samples with "%s" dataset ... [%d/%d]' % (name, i+1, len(target_datasets)))

        # output folder
        make_dir(os.path.join(datasets[name]['path'], datasets[name]['type']))
        # load mean image
        mean_image = np.load(os.path.join(datasets[name]['path'], 'mean_image.npy'))

        # loop to generate samples
        sample_count = 0
        for video in range(1, datasets[name]['num_videos'] + 1):

            # get paths of target images
            image_folder = os.path.join(datasets[name]['path'], '%sing_videos' % datasets[name]['type'],
                                        os.path.splitext(datasets[name]['name_format'] % video)[0])
            image_paths = get_file_paths(image_folder, ['png', 'PNG'])
            target_image_paths = image_paths[0::frame_stride]
            num_frames = len(target_image_paths)

            # read target images and preprocess them
            sys.stdout.write('\tAt "%s"... [0/%d]' % (datasets[name]['name_format'] % video, num_frames))
            read_pos = 0
            for start_pos in range(0, num_frames, sample_stride):
                sys.stdout.write('\r\tAt "%s"... [%d/%d]'
                                 % (datasets[name]['name_format'] % video, start_pos, num_frames))

                # check end-of-processing
                if start_pos + target_length > num_frames:
                    break

                # reallocate already read images
                sample_data = np.roll(sample_data, sample_stride, axis=0)

                # read and preprocess only unread images
                reading_images = target_image_paths[read_pos:start_pos+target_length]
                for j, path in enumerate(reading_images):
                    image_data = np.array(Image.open(path), dtype=np.float)

                    # preprocessing
                    if preproc_type is 'centering':
                        # only subtract a mean image from frame images
                        image_data -= mean_image
                        image_data /= 255.0
                    elif preproc_type is 'thresholding':
                        # thresholding at difference between mean and frame images
                        pass
                    elif preproc_type is 'motion_blob':
                        # background subtraction with a single Gaussian model algorithm
                        pass
                    else:
                        pass

                    # insert to sample container
                    sample_data[read_pos-start_pos+j] = image_data

                torch.save(sample_data, os.path.join(datasets[name]['path'], datasets[name]['type'],
                                                     '%06d.t7' % sample_count))
                sample_count += 1
                read_pos = start_pos + target_length

            print('\r\tAt "' + datasets[name]['name_format'] % video + '"... done!')
        print('%d samples are generated.' % sample_count)



# todo - 저장한 파일의 명세서 저장할 것. 크기, 프레임 등
# =============================================================================
# MAIN PROCEDURE
# =============================================================================
extract_video_frames()
get_mean_image()
generate_samples()

# ()()
# ('')HAANJU.YOO

