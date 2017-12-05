import os
import sys
import subprocess as sp
from PIL import Image
import numpy as np
import glob

FFMPEG_BIN = "ffmpeg"
target_rows = 224
target_cols = 224
grid_unit = 4



def make_dir(path):
    # if there is no directory, make a directory.
    if not os.path.exists(path):
        os.makedirs(path)
    return


def get_file_paths(path, separator, file_type):
    # return file list of the given type in the given path.
    file_paths = []
    if not os.path.exists(path):
        return file_paths
    for extension in file_type:
        file_paths += glob.glob(path + separator + extension)
    file_paths.sort()
    return file_paths


def RedtoGreenScaler(value):
    # input : min~max
    # return : color(Red[255,0,0]~Green[0,255,0])
    min = 0
    max = 100

    assert value >= min

    # Clamp values above max
    if value >= max:
        value = max
    value = value * (100 / max)
    # scaled to 0 ~ 100

    R = (255 * value) / 100
    G = (255 * (100 - value)) / 100
    B = 0

    color = [int(R), int(G), int(B)]
    return color


def grid_axis_generator(grid_number, image_size, grid_unit=4):
    # grid_number = number of grid
    # image_size = size of original image divided by grid
    # grid_unit = how many sides of image were divided into several grids

    # return : the x,y coordinates and grid size corresponding to the grid number

    grid_number = int(grid_number)
    grid_size = int(image_size / grid_unit)

    if grid_number < (grid_unit * grid_unit):
        quo_grid_number = int(grid_number / grid_unit)
        rem_grid_number = grid_number % grid_unit

        grid_x = int(grid_size * quo_grid_number)
        grid_y = int(grid_size * rem_grid_number)
    else:
        sub_quo_grid_number = int((grid_number - (grid_unit * grid_unit)) / (grid_unit - 1))
        sub_rem_grid_number = (grid_number - (grid_unit * grid_unit)) % (grid_unit - 1)

        grid_x = int(grid_size * sub_quo_grid_number + grid_size / 2)
        grid_y = int(grid_size * sub_rem_grid_number + grid_size / 2)


    #data = data[:, grid_x:grid_x + grid_size, grid_y:grid_y + grid_size]
    #returns the coordinates corresponding to grid
    return grid_x, grid_y, grid_size-1

def border_drawer(frame, original_image, MSE_list):
    # frame : number of frame
    # original_image : original image
    # MSE_list : list of MSE(number of grid per frame * number of frame)

    bordered_image = original_image
    for (i, MSE) in enumerate(MSE_list):
        grid_x, grid_y, grid_size = grid_axis_generator(i, target_rows, grid_unit)
        MSE_color = RedtoGreenScaler(MSE)

        bordered_image[grid_y:grid_y + grid_size, grid_x, :] = MSE_color
        bordered_image[grid_y:grid_y + grid_size, grid_x + grid_size, :] = MSE_color
        bordered_image[grid_y, grid_x:grid_x + grid_size, :] = MSE_color
        bordered_image[grid_y + grid_size, grid_x:grid_x + grid_size, :] = MSE_color
    if ground_truth[frame] == 1:
        bordered_image[0:target_rows - 1, 0, :] = [0, 0, 255]
        bordered_image[0:target_rows - 1, target_rows - 1, :] = [0, 0, 255]
        bordered_image[0, 0:target_rows - 1, :] = [0, 0, 255]
        bordered_image[target_rows - 1, 0:target_rows - 1, :] = [0, 0, 255]

    # Returns the border of the color corresponding to the MSE of each grid on the original image
    return bordered_image




print('Frames loading...')
filename = "full_test"
original_image_path ="/home/leejeyeol/Data/endoscope_only/frames"
recon_image_path="/home/leejeyeol/git/AutoencodingTheWorld/training_result/endoscope/ShowImages"
cost_path = "/home/leejeyeol/git/AutoencodingTheWorld/training_result/endoscope/recon_costs"
ground_truth = np.load(os.path.join(cost_path, "Kim Jun Hong_ground_truth.npy"))
cost_list_per_image = np.load(os.path.join(cost_path, "endoscope_full_test_endoscope-BN_grid_mse_list.npy"))


image_files = get_file_paths(os.path.join(original_image_path, filename), '/*.', ['png', 'PNG'])
recon_image_files = get_file_paths(recon_image_path, '/*.', ['png', 'PNG'])
bordered_images_save_path = os.path.join(original_image_path, filename, "bordered")
make_dir(bordered_images_save_path)


#todo : add dual video container
dual_video_container = np.zeros((target_rows, target_cols*2, 3))

for (i, path) in enumerate(image_files):
    # call image frame
    cur_image = np.array(Image.open(path), dtype=np.float)

    cur_recon_image = np.array(Image.open(recon_image_files[i]), dtype=np.float)
    # call MSE list
    MSE_list = cost_list_per_image[i]
    # border drawer -> bordered image -> save
    #todo : add dual video
    a = np.uint8(border_drawer(i, cur_image, MSE_list))
    b = np.uint8(border_drawer(i, cur_recon_image, MSE_list))
    dual_video_container[0:target_rows, 0:target_cols, :] = a
    dual_video_container[0:target_rows, target_cols:(target_cols * 2), :] = b

    img=Image.fromarray(np.uint8(dual_video_container))
    img.save(os.path.join(bordered_images_save_path, "%06d.png" % i))
    print("[%06d/%06d]" % (i, len(image_files)))

# run ffmpeg command
# Create a video with the bordered images you created
print('\tFrom: ' + filename)
command = [FFMPEG_BIN,
           '-r', '10',
           '-f', 'image2',
           '-i', os.path.join(bordered_images_save_path, "%06d.png"),
           '-s', str(target_rows*2) + 'x' + str(target_cols),  # [rows x cols]
           '-pix_fmt', 'yuv420p',
           os.path.join(original_image_path, (filename + '_evaluate' + '.mp4'))]
sp.call(command)  # call command
print("Extraction is done")
