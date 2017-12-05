# data path
# groun truth path
# load data and GT
# load size of data
# deciding patch size(odd case, even case)
# main, (up, down, left, right)xn
# applies flip to each patch
# save to new folder
import os
import numpy as np
import torch
import glob

data_path = '/media/leejeyeol/74B8D3C8B8D38750/Data/CVC-ClinicDB/train'
ground_truth_path = '/media/leejeyeol/74B8D3C8B8D38750/Data/CVC-ClinicDB/Ground_Truth'
data_save_path = '/media/leejeyeol/74B8D3C8B8D38750/Data/CVC-ClinicDB/train_augmented'
ground_truth_save_path = '/media/leejeyeol/74B8D3C8B8D38750/Data/CVC-ClinicDB/Ground_Truth_augmented'


def make_dir(path):
    # if there is no directory, make a directory.
    # make_dir(save_path)
    if not os.path.exists(path):
        os.makedirs(path)
        print(path+" : the save directory is maked.")
    return



def get_mean_image():
    mean_image = np.load(os.path.join(os.path.dirname(data_path), "mean_image.npy"))
    mean_image = np.transpose(mean_image, (2, 0, 1))
    mean_image = torch.from_numpy(mean_image).float()
    return mean_image

make_dir(data_save_path)
make_dir(ground_truth_save_path)

cur_file_paths = glob.glob(data_path + '/*.npy')
cur_file_paths.sort()

cur_GT_paths = glob.glob(ground_truth_path + '/*.npy')
cur_GT_paths.sort()

data = np.load(cur_file_paths[0])
original_h = data.shape[0]
original_w = data.shape[1]
patch_h = int(original_h / 2)
patch_w = int(original_w / 2)
original_h = patch_h*2
original_w = patch_w*2

image_anchor = [[int(original_h/4),int(original_w/4)],[0,int(original_w/4)],[int(original_h/4),0],[int(original_h/4),int(original_w/2)],[int(original_h/2),int(original_w/4)]]

for i in range(len(cur_file_paths)):
    print("[%d/%d]"%(i,len(cur_file_paths)))

    data = np.load(cur_file_paths[i])
    GT = np.load(cur_GT_paths[i])
    for j, anchor in enumerate(image_anchor):
        patch = data[anchor[0]:anchor[0]+patch_h, anchor[1]:anchor[1]+patch_w, :]
        np.save(os.path.join(data_save_path, os.path.basename(cur_file_paths[i]).split('.')[0] + '_%d_0.npy' % j), patch)
        np.save(os.path.join(data_save_path, os.path.basename(cur_file_paths[i]).split('.')[0] + '_%d_1.npy' % j), np.flip(patch, 0))
        np.save(os.path.join(data_save_path, os.path.basename(cur_file_paths[i]).split('.')[0] + '_%d_2.npy' % j), np.flip(patch, 1))
        np.save(os.path.join(data_save_path, os.path.basename(cur_file_paths[i]).split('.')[0] + '_%d_3.npy' % j), np.flip(np.flip(patch, 0), 1))

        GT_patch = GT[anchor[0]:anchor[0]+patch_h, anchor[1]:anchor[1]+patch_w]
        np.save(os.path.join(ground_truth_save_path, os.path.basename(cur_GT_paths[i]).split('.')[0] + '_%d_0.npy' % j), GT_patch)
        np.save(os.path.join(ground_truth_save_path, os.path.basename(cur_GT_paths[i]).split('.')[0] + '_%d_1.npy' % j), np.flip(GT_patch, 0))
        np.save(os.path.join(ground_truth_save_path, os.path.basename(cur_GT_paths[i]).split('.')[0] + '_%d_2.npy' % j), np.flip(GT_patch, 1))
        np.save(os.path.join(ground_truth_save_path, os.path.basename(cur_GT_paths[i]).split('.')[0] + '_%d_3.npy' % j), np.flip(np.flip(GT_patch, 0), 1))





