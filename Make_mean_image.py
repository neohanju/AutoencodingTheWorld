import os
import cv2
import numpy as np
import utils


folder_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/CVC-ClinicDB"
original_image_path = os.path.join(folder_path, "Remove_Boundary")

#=======================================================================================================================
#   Functions
#=======================================================================================================================
def Make_mean_image(original_image_path):
    # call imagepath list
    image_list=utils.get_file_paths(original_image_path, "/*.", ['png', 'PNG'])
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