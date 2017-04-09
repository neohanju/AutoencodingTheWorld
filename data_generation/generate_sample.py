import os
import subprocess as sp
from PIL import Image
import numpy
import shutil
import torch

FFMPEG_BIN = "ffmpeg"
DATASET_PATH = '/home/leejeyeol/Datasets'

number_of_images_in_dataset_all = 0

target_rows = 227
target_cols = 227

target_length = 10
frame_stride = 2
sample_stride = 5

dataset_List = ['avenue_train']

datasets = {}
datasets['mytestvideo'] = {
    # todo 지울것 : program test data set
    'path': DATASET_PATH + '/Test_JY/training_videos',
    'dataset': 'Test_JY',
    'type': 'train',
    'name_format': '%02d.avi',
    'num_videos': 1
}
datasets['avenue_train'] = {
    'path': DATASET_PATH + '/Avenue/training_videos',
    'dataset': 'Avenue',
    'type': 'train',
    'name_format': '%02d.avi',
    'num_videos': 16
}
datasets['avenue_test'] = {
    'path': DATASET_PATH + '/Avenue/testing_videos',
    'dataset': 'Avenue',
    'type': 'test',
    'name_format': '%02d.avi',
    'num_videos': 21
}
datasets['enter_train'] = {
    'path': DATASET_PATH + '/Enter/testing_videos',
    'dataset': 'Enter',
    'type': 'train',
    'name_format': '%02d.avi',
    'num_videos': 1
}
datasets['enter_test'] = {
    'path': DATASET_PATH + '/Enter/testing_videos',
    'dataset': 'Enter',
    'type': 'test',
    'name_format': '%02d.avi',
    'num_videos': 6
}


# =============================================================================
# SET DATASET LIST
# =============================================================================


def make_dir(path):
    # if there is no directory, make a directory.
    if not os.path.exists(path):
        os.makedirs(path)
    return


def call_files(path, file_type):
    # return file list of the given type in the given path.
    allfiles = os.listdir(path)
    imglist = [filename for filename in allfiles if filename[-(len(file_type[0])):] in file_type]
    imglist.sort()
    return imglist


# =============================================================================
# EXTRACT FRAMES FROM VIDEOS
# =============================================================================


def Video_to_image():
    # FFMPEG -> frame extraction & rescale & grayscale  & save to disk(png)
    for dataset_name in dataset_List:
        for video in range(1, datasets[dataset_name]['num_videos'] + 1):
            make_dir(datasets[dataset_name]['path'] + '/%02d' % video)
            # make command.
            command = [FFMPEG_BIN,
                       '-i', datasets[dataset_name]['path'] + '/%02d.avi' % video,
                       '-s', str(target_rows) + 'x' + str(target_cols),  # [rows x cols]
                       '-pix_fmt', 'gray',  # make to grayscale
                       datasets[dataset_name]['path'] + '/%02d' % video + '/output_%05d.png']
            sp.call(command)  # call command

    print("================Video to image done==================")
    return


# =============================================================================
# PREPROCESS
# =============================================================================
def Sample_data_and_count():
    # The number-of-images is obtained, and the images of each frame_stride-th copy are copied.
    global number_of_images_in_dataset_all

    for dataset_name in dataset_List:
        for video in range(1, datasets[dataset_name]['num_videos'] + 1):
            imglist = call_files(datasets[dataset_name]['path'] + '/%02d' % video, [".png", ",PNG"])
            number_of_images_in_dataset_all += len(imglist)  # number of images for mean image.

            sampled_image_path = datasets[dataset_name]['path'] + '/%02d' % video + '/sampled_image'
            make_dir(sampled_image_path)

            for stride in range(0, len(imglist), frame_stride):
                shutil.copy(datasets[dataset_name]['path'] + '/%02d' % video + "/" + imglist[stride],
                            sampled_image_path)
    print("============Sample data and count done===============")
    return


def Make_mean_image():
    # make mean images.
    for dataset_name in dataset_List:
        if (datasets[dataset_name]['type'] == 'train'):
            mean_image = numpy.zeros((target_rows, target_cols), numpy.float)  # mean image container
            for video in range(1, datasets[dataset_name]['num_videos'] + 1):
                image_path = datasets[dataset_name]['path'] + '/%02d' % video
                imglist = call_files(image_path, [".png", ".PNG"])

                for im in imglist:
                    imarr = numpy.array(Image.open(image_path + '/' + im), dtype=numpy.float)
                    mean_image = mean_image + imarr

            mean_image = mean_image / number_of_images_in_dataset_all   # make mean image.
            numpy.save(os.path.join(DATASET_PATH + '/' + datasets[dataset_name]['dataset'] + '/mean_image'),
                       mean_image)  # save mean image .npy
            #Image.fromarray(mean_image).show()  # make sure you make it right.
    print("===============Make mean image done==================")
    return


def Make_Image_minus_mean(sampled_image_path, dataset_name):
    imglist = call_files(sampled_image_path, [".png", ".PNG"])
    num_of_images = len(imglist)
    mean_image = numpy.load(DATASET_PATH + '/' + datasets[dataset_name]['dataset'] + '/mean_image.npy')  # load mean image
    for im in imglist:
        imarr = numpy.array(Image.open(sampled_image_path + '/' + im), dtype=numpy.float)

        out = imarr - mean_image
        out = out / 255
        torch.save(out, sampled_image_path + '/' + im[:-4] + '.t7') #save t7 file (1x277x277 float)

        # *this section causes overwrite sampled image(just for test)------------
        out = numpy.abs(out)
        out = out * 255
        out = numpy.array(numpy.round(out), dtype=numpy.uint8)
        Image.fromarray(out, mode="L").save(sampled_image_path + '/' + im)  # image visualize
        # ------------------------------------------------------------------------
    return num_of_images

def Make_Image_minus_mean_Threshold(sampled_image_path, dataset_name,threshold_value):
    imglist = call_files(sampled_image_path, [".png", ".PNG"])
    num_of_images = len(imglist)
    mean_image = numpy.load(DATASET_PATH + '/' + datasets[dataset_name]['dataset'] + '/mean_image.npy')  # load mean image
    for im in imglist:
        imarr = numpy.array(Image.open(sampled_image_path + '/' + im), dtype=numpy.float)

        out = imarr - mean_image
        out = numpy.where((numpy.abs(out) < threshold_value), 0, out)
        out = out / 255
        torch.save(out, sampled_image_path + '/' + im[:-4] + '.t7') #save t7 file (1x277x277 float)

        # *this section causes overwrite sampled image(just for test)------------
        out = numpy.abs(out)
        out = out * 255
        out = numpy.array(numpy.round(out), dtype=numpy.uint8)
        Image.fromarray(out, mode="L").save(sampled_image_path + '/' + im)  # image visualize
        # ------------------------------------------------------------------------
    return num_of_images


def Generate_InputData(preprocessing_type):
    for dataset_name in dataset_List:
        # make sure you get it right
        if (datasets[dataset_name]['type'] == 'train'):
            Inputdata_path = datasets[dataset_name]['path'] + '/finalDataset_train'
            make_dir(Inputdata_path)

        savedImageNumber = 0  # it will be used for naming.

        for video in range(1, datasets[dataset_name]['num_videos'] + 1):
            if (datasets[dataset_name]['type'] == 'test'):
                make_dir(datasets[dataset_name]['path'] + '/testDataset_%02d' % video)
            sampled_image_path = datasets[dataset_name]['path'] + '/%02d' % video + '/sampled_image'
            if preprocessing_type == 'mean image':
                num_of_images = Make_Image_minus_mean(sampled_image_path,dataset_name)
            elif preprocessing_type == 'mean image threshold':
                num_of_images = Make_Image_minus_mean_Threshold(sampled_image_path, dataset_name,
                                                                threshold_value=threshold_value)


        # Create input data by grouping the created files.
        imglist_f = call_files(sampled_image_path, [".t7", ".T7"])
        imglist_f.sort()
        start_frame = 0 #initialize start point
        end_frame = start_frame + target_length
        while (1):
            if (end_frame > num_of_images):
                break

            datatsr = [torch.load(sampled_image_path + '/' + imglist_f[k]) for k in range(start_frame, end_frame)]
            #make inputdata set (target_Length x 277 x 277 float)

            if (datasets[dataset_name]['type'] == 'train'):
                torch.save(datatsr, Inputdata_path + '/%06d.t7' % savedImageNumber)
            elif (datasets[dataset_name]['type'] == 'test'):
                torch.save(datatsr,
                           datasets[dataset_name][
                               'path'] + '/finalDataset_%02d' % video + '/%06d.t7' % savedImageNumber)
            savedImageNumber = savedImageNumber + 1
            start_frame = start_frame + sample_stride
            end_frame = end_frame + sample_stride

    print("==============Generate inputdata done================")
    return


# todo - 저장한 파일의 명세서 저장할 것. 크기, 프레임 등
# =============================================================================
# GENERATE SAMPLES WITH IMAGES
# =============================================================================

print("Dataset List : avenue_train,avenue_test,enter_train,enter_test \n you can select multiple dataset. \n "
      "if you want stop enter name, plz enter 'doen' \n")
# todo-원하는 데이터셋 입력
'''
while(1):
    dataset_name = input("plz enter dataset name : ")
    if dataset_name == 'done':
        print("==============")
        break
    if dataset_name not in list(datasets.keys()):
        print("sorry, it's not dataset name.")
    dataset_List.append(dataset_name)
'''


Video_to_image()
Sample_data_and_count()
# ---
Make_mean_image()
threshold_value = 15
Generate_InputData('mean image')
#'mean image' 'mean image threshold'
# ---


# todo-이미지 불러오고 저장하는 과정을 최대한 줄이자.
#
