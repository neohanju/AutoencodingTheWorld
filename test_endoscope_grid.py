import argparse
import os
import random
import time
import torch.utils.data
import numpy as np
from torch.autograd import Variable
from models import init_model_and_loss
from data import Grid_RGBImageSets
import utils as util
from PIL import Image




def debug_print(arg):
    if not options.debug_print:
        return
    print(arg)


# =============================================================================
# OPTIONS
# =============================================================================
parser = argparse.ArgumentParser(description='Detecting abnormal behavior in videos')

# model related ---------------------------------------------------------------
parser.add_argument('--model_path', type=str, required=True, help='path to trained model file')
# output related --------------------------------------------------------------
parser.add_argument('--output_type', type=str, default='recon_costs', help='type of output')
# data related ----------------------------------------------------------------
parser.add_argument('--dataset', type=str, required=True, nargs='+',
                    help="all | avenue | ped1 | ped2 | enter | exit. 'all' means using entire data")
parser.add_argument('--videos', type=int, nargs='*', default=None, help='list of video indices to test.')
parser.add_argument('--data_root', type=str, required=True, help='path to base folder of entire dataset')
# display related -------------------------------------------------------------
parser.add_argument('--display', action='store_true', default=False,
                    help='visualize things with visdom or not. default=False')
# GPU related -----------------------------------------------------------------
parser.add_argument('--num_gpu', type=int, default=1, help='number of GPUs to use. default=1')
# ETC -------------------------------------------------------------------------
parser.add_argument('--random_seed', type=int, help='manual seed')
parser.add_argument('--debug_print', action='store_true', default=False, help='print debug information')
# -----------------------------------------------------------------------------

options = parser.parse_args()

# todo : add to test
cost_path = "/home/leejeyeol/git/AutoencodingTheWorld/training_result/endoscope/recon_costs"
ground_truth = np.load(os.path.join(cost_path, "Kim Jun Hong_ground_truth.npy"))
ground_truth[1] = 1

# =======================================







# seed
if options.random_seed is None:
    options.random_seed = random.randint(1, 10000)

# load options from metadata
metadata_path = options.model_path.replace('.pth', '.json')

train_info, options, saved_options = util.load_metadata(metadata_path, options)

# print test options
options_dict = util.namespace_to_dict(options)
print('Options={')
for k, v in options_dict.items():
    print('\t' + k + ':', v)
print('}')

# print network options
saved_option_dict = util.namespace_to_dict(saved_options)
print('Restored options={')
for k, v in saved_option_dict.items():
    print('\t' + k + ':', v)
print('}')


# =============================================================================
# INITIALIZATION PROCESS
# =============================================================================
cuda_available = torch.cuda.is_available()
torch.manual_seed(options.random_seed)
if cuda_available:
    torch.cuda.manual_seed_all(options.random_seed)

# result saving
save_path = os.path.join(os.path.dirname(options.model_path), options.output_type)
util.make_dir(save_path)

# visualization
win_recon_cost = None
win_images = dict(
    exist=False,
    input_frame=None,
    input_data=None,
    recon_data=None,
    recon_frame=None,
    recon_error=None
)

# =============================================================================
# DATA PREPARATION
# =============================================================================
dataset_paths = options.data_root
mean_image_path = os.path.join(dataset_paths, "mean_image.npy")
# todo : get video_ids by options
video_ids=[options.dataset[0]]
mean_image = np.load(mean_image_path)

# todo -
grid_unit = 4
grid_size = 56
temp = np.zeros((3, grid_size, grid_size))
for grid_number in range(0, 25):
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

    temp = temp + mean_image[:,
                  grid_x : grid_x + grid_size,
                  grid_y : grid_y + grid_size]
mean_image = temp/25


dataset = Grid_RGBImageSets(dataset_paths, centered=False, video_ids=video_ids)
dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=25, shuffle=False,
                                         num_workers=1, pin_memory=True)

# streaming buffer
tm_buffer_set = time.time()
input_batch = torch.FloatTensor(25, saved_options.nc, grid_size, grid_size)
recon_batch = torch.FloatTensor(25, saved_options.nc, grid_size, grid_size)
mu_batch = torch.FloatTensor(25, options.z_size[0], options.z_size[1], options.z_size[2])
logvar_batch = torch.FloatTensor(25, options.z_size[0], options.z_size[1], options.z_size[2])
debug_print('Stream buffers are set: %.3f sec elapsed' % (time.time() - tm_buffer_set))

if cuda_available:
    debug_print('Start transferring to CUDA')
    tm_gpu_start = time.time()
    input_batch = input_batch.cuda()
    recon_batch = recon_batch.cuda()
    mu_batch = mu_batch.cuda()
    logvar_batch = logvar_batch.cuda()
    debug_print('Transfer to GPU: %.3f sec elapsed' % (time.time() - tm_gpu_start))

tm_to_variable = time.time()
input_batch = Variable(input_batch)
recon_batch = Variable(recon_batch)
debug_print('To Variable for Autograd: %.3f sec elapsed' % (time.time() - tm_to_variable))

print('Data streaming is ready')

# for utility library
util.target_sample_index = 0
util.target_frame_index = int(saved_options.nc / 2)
util.mean_images = mean_image
debug_print('Utility library is ready')


# =============================================================================
# MODEL
# =============================================================================
model, our_loss = init_model_and_loss(options, cuda_available)
print(model)


# =============================================================================
# TESTING
# =============================================================================
model.eval()
recon_cost_dict = dict(recon=0)
recon_costs = {}
sample_index = 0

print('Start testing...')
tm_test_start = time.time()

cost_file_path = ''
z_file_path = ''
prev_dataset_name = ''
prev_video_name = ''
cnt_cost = 0

mean = 0

cost_npy = []
mse_list = []


for i, (data, grid_number) in enumerate(dataloader, 0):
    dataset_name = "endoscope"
    video_name = video_ids[0]
    if prev_dataset_name != dataset_name or prev_video_name != video_name:
        # new video is started
        print("Testing on '%s' dataset video '%s'... " % (dataset_name, video_name))


        # create new cost file
        cost_file_path = os.path.join(save_path, '%s_video_%s_%s.txt'
                                      % (dataset_name[0], video_name[0], saved_options.model))
        util.file_print_list(cost_file_path, [], overwrite=True)

        # draw new cost graph
        win_recon_cost = None

        # create new latent space
        z_file_path = os.path.join(save_path, '%s_video_%s_%s_latent_variables.txt'
                                   % (dataset_name[0], video_name[0], saved_options.model))
        util.file_print_list(z_file_path, [], overwrite=True)

        prev_dataset_name, prev_video_name = dataset_name, video_name
        cnt_cost = 0
        frame_MSE_container = 0

    # data load
    input_batch.data.copy_(data)

    # forward
    model.zero_grad()
    recon_batch = model(input_batch)
    loss, loss_detail, max_loss, mean_loss, min_loss, loss_list = our_loss.calculate(recon_batch, input_batch)

    cur_cost = loss_detail['recon']
    mean = mean * ((cnt_cost) / (cnt_cost + 1)) + (cur_cost / (cnt_cost + 1))
    cnt_cost += 1
    util.file_print_list(cost_file_path, [cur_cost], overwrite=False)
    cost_npy.append(cur_cost)
    mse_list.append(loss_list)
    print('%s_video_%s:[%07d/%07d]\t cost = %.3f \t max = %.3f\t min = %.3f\t mean = %.3f\t total_mean = %.3f'
          % (dataset_name, video_name, cnt_cost, len(dataloader), cur_cost, max_loss, min_loss, mean_loss, mean))
  #      frame_MSE_container = 0


    # save latent variables
    #np_mu = mu_batch.data[0, :, :, :].cpu().numpy().flatten()
    #np_sig = np.exp(0.5 * logvar_batch.data[0,:,0,0].cpu().numpy())
    #util.file_print_list(z_file_path, np_mu.tolist(), False)

    # visualization
    if options.display:
        win_images = util.draw_images_RGB(win_images, data, recon_batch.data, mean_image, setnames=dataset_name)
        win_recon_cost = util.viz_append_line_points(win=win_recon_cost,
                                                     lines_dict=dict(recon=cur_cost, zero=0),
                                                     x_pos=cnt_cost,
                                                     title='%s: video_%s' % (dataset_name, video_name),
                                                     ylabel='reconstruction cost', xlabel='sample index')

        recon_container = np.zeros((3, grid_size*grid_unit, grid_size*grid_unit))
        recon_container_2nd = np.zeros((3,grid_size*(grid_unit-1), grid_size*(grid_unit-1)))

        for x in range(0, grid_unit):
            for y in range(0, grid_unit):
                recon_container[:,
                y*grid_size:y*grid_size+grid_size,
                x*grid_size:x*grid_size+grid_size] = \
                    recon_batch.data[x+grid_unit*y].view(3, grid_size, grid_size).cpu().numpy()


        for x in range(0, (grid_unit-1)):
            for y in range(0, (grid_unit-1)):
                recon_container_2nd[:,
                int((y*grid_size)):int((y*grid_size)+grid_size),
                int((x*grid_size)):int((x*grid_size)+grid_size)] = \
                    recon_batch.data[grid_unit*grid_unit+(x+((grid_unit-1)*y))].view((3, grid_size, grid_size)).cpu().numpy()


        recon_container[:,
        int(grid_size/2):int(grid_size/2+grid_size*(grid_unit-1)),
        int(grid_size/2):int(grid_size/2+grid_size*(grid_unit-1))] = \
            (recon_container_2nd+recon_container[:,
                                 int(grid_size/2):int(grid_size/2+grid_size*(grid_unit-1)),
                                 int(grid_size/2):int(grid_size/2+grid_size*(grid_unit-1))])/2

        recon_container = np.uint8(((recon_container + 1) / 2) * 255)
        Image.fromarray(np.transpose(np.uint8(recon_container))).save(
            "/home/leejeyeol/git/AutoencodingTheWorld/training_result/endoscope/ShowImages/%06d.png" % i)

        '''
        recon_container = np.uint8(((recon_container+1)/2)*255)
        recon_container_2nd = np.uint8(((recon_container_2nd+1)/2)*255)
        Image.fromarray(np.transpose(recon_container, (1,2,0)), 'RGB').save("/home/leejeyeol/git/AutoencodingTheWorld/training_result/endoscope/ShowImages/%06d_recon_1.png"%i)
        Image.fromarray(np.transpose(recon_container_2nd, (1,2,0)), 'RGB').save("/home/leejeyeol/git/AutoencodingTheWorld/training_result/endoscope/ShowImages/%06d_recon_2.png"%i)
        '''



        '''
        in_img = util.pick_frame_from_batch_RGB(input_batch)
        re_img = util.pick_frame_from_batch_RGB(recon_batch)
        in_img = util.sample_batch_to_image_RGB(input_batch)
        re_img = util.sample_batch_to_image_RGB(recon_batch)
        re_img.shape = (grid_size, grid_size, 3)
        in_img.shape = (grid_size, grid_size, 3)

        in_img2 = Image.fromarray(in_img, mode="RGB")
        re_img2 = Image.fromarray(re_img, mode="RGB")
        in_img2.save("/home/leejeyeol/git/AutoencodingTheWorld/training_result/endoscope/ShowImages/%06d_input.png"%i)
        re_img2.save("/home/leejeyeol/git/AutoencodingTheWorld/training_result/endoscope/ShowImages/%06d_recon.png"%i)
        '''
        time.sleep(0.005)  # for reliable drawing

np.save(os.path.join(save_path, '%s_%s_%s.npy'
                                      % (dataset_name, video_name, saved_options.model)),cost_npy)
np.save(os.path.join(save_path, '%s_%s_%s_grid_mse_list.npy'
                                      % (dataset_name, video_name, saved_options.model)),mse_list)
print(mean)
print(os.path.join(save_path, '%s_%s_%s.npy' % (dataset_name, video_name, saved_options.model)))
# ()()
# ('')HAANJU.YOO
