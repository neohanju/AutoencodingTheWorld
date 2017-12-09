import os
import cv2
import numpy as np
import glob


folder_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/CVC-ClinicDB"
original_image_path = os.path.join(folder_path, "Remove_Boundary")
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
def Make_mean_image(original_image_path):
    # call imagepath list
    image_list=get_file_paths(original_image_path, "/*.", ['png', 'PNG'])
    # load image and ..
    mean_image = None
    for (i, image) in enumerate(image_list):
        original_image = cv2.imread(image, -1)
        if mean_image == None:
            mean_image = original_image
        else :
            mean_image = (mean_image/2 + original_image/2)

        print("[%d/%d]" % (i+1, len(image_list)))
    cv2.imwrite(os.path.join(folder_path, 'mean_image.png'), mean_image)
    np.save(os.path.join(folder_path, 'mean_image'), mean_image)



#=======================================================================================================================
#   run
#=======================================================================================================================

Make_mean_image(original_image_path)