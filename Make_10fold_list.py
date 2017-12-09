import os
import glob
import numpy as np
from collections import Counter


folder_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/CVC-ClinicDB"
original_image_path = os.path.join(folder_path, "train_augmented")
def get_file_paths(path, separator, file_type):
    # return file list of the given type in the given path.
    # image_files = get_file_paths(image_folder, '/*.', ['png', 'PNG'])
    file_paths = []
    if not os.path.exists(path):
        return file_paths
    for extension in file_type:
        file_paths += glob.glob(path + separator + extension)
    file_paths.sort()
    return file_paths

#=======================================================================================================================
#   Functions
#=======================================================================================================================
def Make_10fold_list(original_image_path):
    # call imagepath list
    image_list = get_file_paths(original_image_path, "/*.", ['npy', 'NPY'])
    # load image and ..
    num_of_test = int(len(image_list)/10)


    for i in range(0, 10):
        test_set = np.random.choice(image_list, num_of_test)
        test_set.sort()

        c_all = Counter(image_list)
        c_test = Counter(test_set)
        diff = c_all - c_test
        train_set = np.asarray(list(diff.elements()))

        np.save(os.path.join(folder_path, "10fold_%d_train" % i), train_set)
        np.save(os.path.join(folder_path, "10fold_%d_test" % i), test_set)


#=======================================================================================================================
#   run
#=======================================================================================================================

Make_10fold_list(original_image_path)