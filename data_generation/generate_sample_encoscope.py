import numpy as np
import os
import glob
from PIL import Image


filenames = ["video_1", "video_2"]
folder_path = '/home/leejeyeol/Data/endoscope_old/Videos'
save_path = '/home/leejeyeol/Data/endoscope_old/Videos/frames'

# image size
x_size = 227
y_size = 227

def make_dir(path):
    # if there is no directory, make a directory.
    if not os.path.exists(path):
        os.makedirs(path)
    return

def get_file_paths(path, separator , file_type):
    # return file list of the given type in the given path.
    file_paths = []
    if not os.path.exists(path):
        return file_paths
    for extension in file_type:
        file_paths += glob.glob(path + separator + extension)
    file_paths.sort()
    return file_paths

print("save folder make done")
make_dir(save_path)
for file in filenames:
    print("%s file set" % file)
    mean_image = np.zeros((x_size, y_size,3), dtype=np.float)  # mean image container
    image_axis = None # image container
    save_image_idx = 0  # index for save
    make_dir(os.path.join(save_path, file))
    # call image frames per file
    image_folder = os.path.join(folder_path, os.path.splitext(file)[0])
    image_files = get_file_paths(image_folder, '/*.', ['png', 'PNG'])
    for path in image_files:
        # call image frame
        cur_image = np.array(Image.open(path), dtype=np.float)      # put image in container
        if image_axis is None:
            image_axis = cur_image      # if image axis is empty, put image in that variable
        if not np.array_equal(image_axis, cur_image):
                        # save final frame
            np.save(os.path.join(save_path, file, "%06d" % save_image_idx), np.swapaxes(image_axis, 0, 2))
            Image.fromarray(np.uint8(image_axis)).save(os.path.join(save_path, file, "%06d.png" % save_image_idx))
            print("%06d images saved" % save_image_idx)
            # calculate mean image
            mean_image = mean_image * ((save_image_idx+1)/(save_image_idx+2)) + image_axis / (save_image_idx+2)

            save_image_idx = save_image_idx + 1
            image_axis = None
    # save mean image
    np.save(os.path.join(save_path, file + "_mean_image"), np.swapaxes(mean_image, 0, 2))
    Image.fromarray(np.uint8(mean_image)).save(os.path.join(save_path, file + "_mean_image.png"))
    print("%s file end" % file)
print("done")


# remove duplicated images
# make mean image
# save npy for sample
# save png for checking

# Je Yeol Lee \[T]/ Jolly Co-operation

