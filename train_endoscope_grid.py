import argparse
import os
import time
import random
import socket
import torch.utils.data
import torch.optim as optim
from torch.autograd import Variable
from models import init_model_and_loss
from data import Grid_RGBImageSets
import utils as util
import numpy as np


def debug_print(arg):
    if not options.debug_print:
        return
    print(arg)

# =============================================================================
# OPTIONS
# =============================================================================
parser = argparse.ArgumentParser(description='Detecting abnormal behavior in videos')

# model related ---------------------------------------------------------------
parser.add_argument('--model', type=str, default='endoscope-BN', help='AE | AE-LTR | VAE | VAE-LTR | VAE-NARROW')
parser.add_argument('--nc', type=int, default=3, help='number of input channel. default=3')
parser.add_argument('--nz', type=int, default=10, help='size of the latent z vector. default=100')
parser.add_argument('--nf', type=int, default=16, help='size of lowest image filters. default=32')
parser.add_argument('--l1_coef', type=float, default=0, help='coef of L1 regularization on the weights. default=0')
parser.add_argument('--l2_coef', type=float, default=0, help='coef of L2 regularization on the weights. default=0')
parser.add_argument('--var_loss_coef', type=float, default=1.0, help='balancing coef of vairational loss. default=0')
# training related ------------------------------------------------------------
parser.add_argument('--model_path', type=str, default='', help='path of pretrained network. default=""')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size. default=64')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train for. default=25')
parser.add_argument('--max_iter', type=int, default=150000, help='number of iterations to train for. default=150,000')
# data related ----------------------------------------------------------------
parser.add_argument('--dataset', type=str, required=True, nargs='+',
                    help="all | avenue | ped1 | ped2 | enter | exit. 'all' means using entire data")
parser.add_argument('--data_root', type=str, required=True, help='path to base folder of entire dataset')
parser.add_argument('--image_size', type=int, default=224, help='input image size (width=height). default=224')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
# optimization related --------------------------------------------------------
parser.add_argument('--optimizer', type=str, default='adagrad',
                    help='type of optimizer: adagrad | adam | asgd | sgd. default=adagrad')
parser.add_argument('--learning_rate', type=float, default=0.0002, help='learning rate. default=0.0002')
parser.add_argument('--learning_rate_decay', type=float, default=0, help='learning rate decay. default=0')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer. default=0.5')
# display related -------------------------------------------------------------
parser.add_argument('--display', action='store_true', default=False,
                    help='visualize things with visdom or not. default=False')
parser.add_argument('--display_interval', type=int, default=1, help='display interval w.r.t. epoch. default=1')
# GPU related -----------------------------------------------------------------
parser.add_argument('--num_gpu', type=int, default=0,
                    help='number of GPUs to use. It will be ignored when gpu_ids options is given. default=0')
parser.add_argument('--gpu_ids', type=int, default=[], nargs='*',
                    help='Indices of GPUs in use. If you give this, num_gpu option input will be ignored. default=[]')
# network saving related ------------------------------------------------------
parser.add_argument('--save_interval', type=int, default=1,
                    help='network saving interval w.r.t. epoch number. default=100')
parser.add_argument('--save_path', type=str, default='./training_result/endoscope',
                    help='path to trained network. default=./training_result')
parser.add_argument('--save_name', type=str, default='endoscope_grid', help='name for network saving')
# ETC -------------------------------------------------------------------------
parser.add_argument('--random_seed', type=int, help='manual seed')
parser.add_argument('--debug_print', action='store_true', default=False, help='print debug information')
# -----------------------------------------------------------------------------

options = parser.parse_args()

# seed
if options.random_seed is None:
    options.random_seed = random.randint(1, 10000)

options.continue_train = options.model_path != ''
if options.continue_train:
    print("load model from '%s'" % os.path.basename(options.model_path))

    # load metadata
    metadata_path = options.model_path.replace('.pth', '.json')
    prev_train_info, options, _ = util.load_metadata(metadata_path, options)

    print('Loaded model was trained %d epochs with %d iterations'
          % (prev_train_info['epoch_count'], prev_train_info['iter_count']))
    print('Starting loss : %.3f' % prev_train_info['total_loss'])

# loss type
options.variational = options.model.find('VAE') != -1

# latent space size
if options.model.find('-LTR') != -1:
    options.z_size = [options.nz, 13, 13]
else:
    options.z_size = [options.nz, 1, 1]

# gpu number
if 0 == len(options.gpu_ids):
    if 0 < options.num_gpu <= torch.cuda.device_count():
        # auto generate GPU indices
        pass
    else:
        options.num_gpu = torch.cuda.device_count()
        if options.num_gpu > torch.cuda.device_count():
            print('[WARNING] Unfortunately, there are not enough # of GPUs as many as you want. Only %d is available.'
                  % options.num_gpu)
    options.gpu_ids = list(range(options.num_gpu))
else:
    # remove redundant, or too small, or too big indices
    candidate_ids = list(set(options.gpu_ids))
    options.gpu_ids = []
    for idx in candidate_ids:
        if idx >= options.num_gpu:
            print("[WARNING] To large index for you GPU setting. discard '%d'" % idx)
        elif idx < 0:
            print("[WARNING] Negative index. discard '%d'" % idx)
        else:
            options.gpu_ids.append(idx)
    options.num_gpu = len(options.gpu_ids)

# print options
options_dict = util.namespace_to_dict(options)
print('Options={')
for k, v in options_dict.items():
    print('\t' + k + ':', v)
print('}')


# =============================================================================
# INITIALIZATION PROCESS
# =============================================================================
cuda_available = torch.cuda.is_available()

# seed
torch.manual_seed(options.random_seed)
if cuda_available:
    torch.cuda.manual_seed_all(options.random_seed)

# network saving
model_folder_name = '%s_%s_%s_%s' % (options.model, util.now_to_string(),
                                     '-'.join(options.dataset), socket.gethostname())
save_path = options.save_path
util.make_dir(save_path)
print("All results will be saved at '%s'" % save_path)

# visualization
win_loss = None
win_time = None
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
# set data loader
dataset_paths = options.data_root
dataset = Grid_RGBImageSets(dataset_paths, centered=False, video_ids=["video_train"])
dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=options.batch_size, shuffle=True,
                                         num_workers=options.workers)

debug_print('Data loader is ready')

# streaming buffer
tm_buffer_set = time.time()
input_batch = torch.FloatTensor(options.batch_size, options.nc, 56, 56)
recon_batch = torch.FloatTensor(options.batch_size, options.nc, 56, 56)
mu_batch = torch.FloatTensor(options.batch_size, options.z_size[0], options.z_size[1], options.z_size[2])
logvar_batch = torch.FloatTensor(options.batch_size, options.z_size[0], options.z_size[1], options.z_size[2])
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
print('input_batch:', input_batch.size())
print('recon_batch:', recon_batch.size())
debug_print('Data streaming is ready')

# for utility library
util.target_sample_index = 0
util.target_frame_index = int(options.nc / 2)
util.mean_images = dataset.get_mean_image()
debug_print('Utility library is ready')

mean_image_path = os.path.join(dataset_paths, "mean_image.npy")
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


    temp = temp + mean_image[:, grid_x:grid_x + grid_size, grid_y:grid_y + grid_size]
mean_image = temp/25


dataset_name = "endoscope"

# =============================================================================
# MODEL & LOSS FUNCTION
# =============================================================================
model, our_loss = init_model_and_loss(options, cuda_available)
print(model)

# =============================================================================
# TRAINING
# =============================================================================
print('Start training...')
model.train()

# optimizer
model_params = model.parameters()
if 'adagrad' == options.optimizer:
    optimizer = optim.Adagrad(model_params, lr=options.learning_rate, lr_decay=options.learning_rate_decay,
                              weight_decay=options.l2_coef)
elif 'adam' == options.optimizer:
    optimizer = optim.Adam(model_params, lr=options.learning_rate, betas=(options.beta1, 0.999),
                           weight_decay=options.l2_coef)
elif 'asgd' == options.optimizer:
    optimizer = optim.ASGD(model_params, lr=options.learning_rate, weight_decay=options.l2_coef)
elif 'sgd' == options.optimizer:
    optimizer = optim.SGD(model_params, lr=options.learning_rate, weight_decay=options.l2_coef)
assert optimizer

print("set optimizer")

# timer related
tm_data_load_total = 0
tm_iter_total = 0
tm_loop_start = time.time()
time_info = dict(cur_iter=0, train=0, visualize=0, ETC=0)
time_info_vis = dict()

# counters
iter_count = 0
recent_loss = 0
num_iters_in_epoch = 0

# for logging
loss_info = dict()
loss_info_vis = dict()
train_info = dict(model=options.model, dataset=options.dataset, epoch_count=0, iter_count=0, total_loss=0,
                  options=options_dict)
if options.continue_train:
    train_info['prev_epoch_count'] = prev_train_info['epoch_count']
    train_info['prev_iter_count'] = prev_train_info['iter_count']
    train_info['prev_total_loss'] = prev_train_info['total_loss']
    if 'prev_epoch_count' in prev_train_info:  # this means that the meta data already has previous training info.
        # accumulate counters
        train_info['prev_epoch_count'] += prev_train_info['prev_epoch_count']
        train_info['prev_iter_count'] += prev_train_info['prev_iter_count']

print("start epoch")
# main loop of training
for epoch in range(options.epochs):
    tm_cur_epoch_start = tm_cur_iter_start = time.time()
    for i, (data, grid_number) in enumerate(dataloader, 1):
        num_iters_in_epoch = i

        # ============================================
        # DATA FEED
        # ============================================
        if data.size() != input_batch.data.size():
            print(data.size())
            print(input_batch.data.size())
            print("data size not matched")
            # input_batch.data.resize_(data.size())
            # recon_batch.data.resize_(data.size())
            # this will be deprecated by 'last_drop' attributes of dataloader
            continue
        input_batch.data.copy_(data)

        # ============================================
        # TRAIN
        # ============================================
        # forward
        tm_train_start = time.time()
        model.zero_grad()
        recon_batch = model(input_batch)

        # backward
        loss, loss_detail, max_loss, mean_loss, min_loss, loss_list = our_loss.calculate(recon_batch, input_batch)
        loss.backward()

        # update
        optimizer.step()
        tm_train_iter_consume = time.time() - tm_train_start
        time_info['train'] += tm_train_iter_consume

        # logging losses
        recent_loss = loss.data[0]
        loss_info = util.add_dict(loss_info, loss_detail)

        # ============================================
        # VISUALIZATION
        # ============================================
        tm_visualize_start = time.time()
        if options.display:
            # draw input/recon images
            win_images = util.draw_images_RGB(win_images, data, recon_batch.data, mean_image, dataset_name)
        tm_visualize_consume = time.time() - tm_visualize_start

        # print iteration's summary
        print('[%4d/%4d][%3d/%3d] Iter:%4d\t %s \tTotal time elapsed: %s'
              % (epoch+1, options.epochs, i, len(dataloader), iter_count+1, util.get_loss_string(loss_detail),
                 util.formatted_time(time.time() - tm_loop_start)))

        # ============================================
        # NETWORK BACK-UP
        # ============================================
        # save network and meta data
        train_info['iter_count'] = iter_count
        train_info['total_loss'] = recent_loss
        train_info['epoch_count'] = epoch
        util.save_model(os.path.join(save_path, options.save_name + '_latest.pth'), model.state_dict(), train_info)

        tm_iter_consume = time.time() - tm_cur_iter_start
        tm_etc_consume = tm_iter_consume - tm_train_iter_consume - tm_visualize_consume
        time_info['cur_iter'] += tm_iter_consume
        time_info['ETC'] += tm_etc_consume
        time_info['visualize'] += tm_visualize_consume
        # ===============================================
        tm_cur_iter_start = time.time()  # to measure the time of enumeration of the loop controller, set timer at here
        iter_count += 1

    average_loss_info = {key: value / num_iters_in_epoch for key, value in loss_info.items()}
    average_time_info = {key: value / num_iters_in_epoch for key, value in time_info.items()}
    loss_info = dict.fromkeys(loss_info, 0)
    time_info = dict.fromkeys(time_info, 0)

    print('====> Epoch %d is terminated: Epoch time is %s, Average loss is %.3f'
          % (epoch+1, util.formatted_time(time.time() - tm_cur_epoch_start), average_loss_info['total']))

    # draw graph at every drawing period (always draw at the beginning(= epoch zero))
    if options.display:
        loss_info_vis = util.add_dict(average_loss_info, loss_info_vis)
        time_info_vis = util.add_dict(average_time_info, time_info_vis)
        if 0 == (epoch+1) % options.display_interval or 0 == epoch:
            # averaging w.r.t. display frequency
            if 1 != options.display_interval and 0 != epoch:
                loss_info_vis = {key: value / options.display_interval for key, value in loss_info_vis.items()}
                time_info_vis = {key: value / options.display_interval for key, value in time_info_vis.items()}
            # draw graphs
            win_loss = util.viz_append_line_points(win_loss, loss_info_vis, epoch)
            win_time = util.viz_append_line_points(win_time, time_info_vis, epoch,
                                                   title='times at each epoch',
                                                   ylabel='time')
            # reset buffers
            loss_info_vis = dict.fromkeys(loss_info_vis, 0)
            time_info_vis = dict.fromkeys(time_info_vis, 0)

    # checkpoint w.r.t. epoch
    if 0 == (epoch+1) % options.save_interval:
        util.save_model(os.path.join(save_path, '%s_epoch_%03d.pth')
                        % (options.save_name, epoch+1), model.state_dict(), train_info, True)


#()()
#('')HAANJU.YOO
