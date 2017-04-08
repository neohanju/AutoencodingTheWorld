import os
import subprocess as sp
import sys
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
    #todo 지울것 : program test data set
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

# =============================================================================
# EXTRACT FRAMES FROM VIDEOS
# =============================================================================

# FFMPEG -> frame extraction & rescale & grayscale  & save to disk(png)
def Video_to_image():
    for dataset_name in dataset_List:
        for video in range(1, datasets[dataset_name]['num_videos']+1):
                if not os.path.exists(datasets[dataset_name]['path']+'/%02d' % video):
                    os.makedirs(datasets[dataset_name]['path']+'/%02d' % video)
                command = [FFMPEG_BIN,
                           '-i', datasets[dataset_name]['path']+'/%02d.avi' % video,
                           '-s', str(target_rows)+'x'+str(target_cols),
                           '-pix_fmt', 'gray',
                           datasets[dataset_name]['path']+'/%02d' % video+'/output_%05d.png']
                sp.call(command)

    print("================Video to image done==================")

    return

# =============================================================================
# PREPROCESS
# =============================================================================
#이미지를 열어서 갯수를 구하고, frame stride에 해당하는 이미지들을 복사한다. 그리고 그 이미지들의 개수를 센다.
#데이터셋이 여러개일 경우가 있어 따로 저장해야한다.
#todo 이거 없앨 수 있을 것 같다.

def Sample_data_and_count():
    global number_of_images_in_dataset_all

    for dataset_name in dataset_List:
        for video in range(1, datasets[dataset_name]['num_videos']+1):
            allfiles = os.listdir(datasets[dataset_name]['path']+'/%02d' % video)
            imglist = [filename for filename in allfiles if filename[-4:] in [".png", ".PNG"]]
            number_of_images_in_dataset_all += len(imglist)     # number of images for mean image.

            sampled_image_path = datasets[dataset_name]['path'] + '/%02d' % video+'/sampled_image'
            if not os.path.exists(sampled_image_path):
                os.makedirs(sampled_image_path)

            for stride in range(1, len(imglist), frame_stride):
                shutil.copy(datasets[dataset_name]['path'] + '/%02d' % video + "/" + imglist[stride], sampled_image_path)



    print("============Sample data and count done===============")
    return

def Make_mean_image():


    for dataset_name in dataset_List:
        mean_image = numpy.zeros((target_rows, target_cols), numpy.float)  # mean image container
        if(datasets[dataset_name]['type']=='train'):
            for video in range(1, datasets[dataset_name]['num_videos']+1):
                image_path = datasets[dataset_name]['path'] + '/%02d' % video
                allfiles = os.listdir(image_path)
                imglist = [filename for filename in allfiles if filename[-4:] in [".png", ".PNG"]]

                for im in imglist:
                    imarr = numpy.array(Image.open(image_path+'/'+im), dtype=numpy.float)
                    imarr = imarr/255 #normalize
                    mean_image = mean_image + imarr

        mean_image = mean_image / number_of_images_in_dataset_all
        numpy.save(os.path.join(DATASET_PATH + '/' + datasets[dataset_name]['dataset'] + '/mean_image'), mean_image)
        Image.fromarray(mean_image*255).show()
                #out.show()
    print("===============Make mean image done==================")
    return

def Generate_sample():

    for dataset_name in dataset_List:
        mean_image = numpy.load(DATASET_PATH + '/' + datasets[dataset_name]['dataset'] + '/mean_image.npy')
        Image.fromarray(mean_image*255).show()
        print(mean_image)
        if (datasets[dataset_name]['type'] == 'train'):
            final_datasample_path = datasets[dataset_name]['path'] + '/finalDataset_train'
            if not os.path.exists(final_datasample_path):
                os.makedirs(final_datasample_path)
        h = 0



        for video in range(1, datasets[dataset_name]['num_videos'] + 1):
            if(datasets[dataset_name]['type']=='test'):
               if not os.path.exists(datasets[dataset_name]['path'] + '/finalDataset_%02d' % video):
                    os.makedirs(datasets[dataset_name]['path'] + '/finalDataset_%02d' % video)

            sampled_image_path = datasets[dataset_name]['path'] + '/%02d' % video + '/sampled_image'
            allfiles = os.listdir(sampled_image_path)
            imglist = [filename for filename in allfiles if filename[-4:] in [".png", ".PNG"]]
            num_of_images = len(imglist)


            for im in imglist:
                imarr = numpy.array(Image.open(sampled_image_path + '/' + im), dtype=numpy.float)
                out = imarr - mean_image*255

                torch.save(out, sampled_image_path + '/' + im[:-4]+'.t7')


                out2 = numpy.array(numpy.round(out),dtype=numpy.uint8)
                Image.fromarray(out2*255, mode="L").save(sampled_image_path + '/' + im)#image visualize

            allfiles = os.listdir(sampled_image_path)
            imglist = [filename for filename in allfiles if filename[-3:] in [".t7", ".T7"]]
            i = 1
            j = i + target_length
            while (1):
                if (j > num_of_images):
                    break
 #               datatsr= [numpy.array(Image.open(sampled_image_path + '/' + imglist[k]), dtype=numpy.float) for k in range(i, i + target_length)]
                datatsr= [torch.load(sampled_image_path + '/' + imglist[k]) for k in range(i, i + target_length)]

                if(datasets[dataset_name]['type']=='train'):
                    numpy.
                    torch.save(datatsr, final_datasample_path+'/%06d.t7' % h)
                elif(datasets[dataset_name]['type']=='test'):
                    torch.save(datatsr, datasets[dataset_name]['path'] + '/finalDataset_%02d' % video +'/%06d.t7' % h)
                h = h+1
                i = i+sample_stride
                j = j+sample_stride
    print("===============Generate sample done==================")
    return
#todo - 저장한 파일의 명세서 저장할 것. 크기, 프레임 등
# =============================================================================
# GENERATE SAMPLES WITH IMAGES
# =============================================================================

print("Dataset List : avenue_train,avenue_test,enter_train,enter_test \n you can select multiple dataset. \n "
      "if you want stop enter name, plz enter 'doen' \n")
#todo-원하는 데이터셋 선택
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
Make_mean_image()
Generate_sample()


#todo-이미지 불러오고 저장하는 과정을 최대한 줄이자.
#