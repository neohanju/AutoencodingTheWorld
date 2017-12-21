import os
import utils
import numpy as np
from collections import Counter


folder_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/CVC-ClinicDB"
original_image_path = os.path.join(folder_path, "train_augmented")


#=======================================================================================================================
#   Functions
#=======================================================================================================================
def Make_10fold_list(original_image_path):
    # call imagepath list
    image_list = utils.get_file_paths(original_image_path, "/*.", ['npy', 'NPY'])
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