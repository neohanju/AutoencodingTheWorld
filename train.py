import argparse
import os
import time
import random
import socket
import json
import numpy as np
import torch
import torch.nn.parallel
import torch.nn.init
import torch.utils.data
import torch.optim as optim
from torch.autograd import Variable
from models import AE, VAE, AE_LTR, VAE_LTR, OurLoss
from data import VideoClipSets
import utils as util


def debug_print(arg):
    if not options.debug_print:
        return
    print(arg)

# =============================================================================
# OPTIONS
# =============================================================================
parser = argparse.ArgumentParser(description='Detecting abnormal behavior in videos')

# model related ---------------------------------------------------------------
parser.add_argument('--model', type=str, required=True, help='AE | AE_LTR | VAE | VAE_LTR')
parser.add_argument('--nc', type=int, default=10, help='number of input channel. default=10')
parser.add_argument('--nz', type=int, default=200, help='size of the latent z vector. default=100')
parser.add_argument('--nf', type=int, default=64, help='size of lowest image filters. default=64')
parser.add_argument('--l1_coef', type=float, default=0, help='coef of L1 regularization on the weights. default=0')
parser.add_argument('--l2_coef', type=float, default=0, help='coef of L2 regularization on the weights. default=0')
parser.add_argument('--var_loss_coef', type=float, default=1.0, help='balancing coef of vairational loss. default=0')
# training related ------------------------------------------------------------
parser.add_argument('--batch_size', type=int, default=64, help='input batch size. default=64')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train for. default=25')
parser.add_argument('--max_iter', type=int, default=150000, help='number of iterations to train for. default=150,000')
parser.add_argument('--partial_learning', type=float, default=1,
                    help='ratio of partial data for training. At least one sample from each file. default=1')
parser.add_argument('--continue_train', action='store_true', default=False,
                    help='load the latest model to continue the training, default=False')
# data related ----------------------------------------------------------------
parser.add_argument('--dataset', type=str, required=True,
                    help="all | avenue | ped1 | ped2 | enter | exit. 'all' means using entire data")
parser.add_argument('--data_root', type=str, required=True, help='path to base folder of entire dataset')
parser.add_argument('--image_size', type=int, default=227, help='input image size (width=height). default=227')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
# optimization related --------------------------------------------------------
parser.add_argument('--optimiser', type=str, default='adam', help='type of optimizer: adagrad | adam')
parser.add_argument('--learning_rate', type=float, default=0.0002, help='learning rate. default=0.0002')
# parser.add_argument('--weight_decay', type=float, default=0.0005,
#                     help='weight decay coefficient for regularization. default=0.0005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer. default=0.5')
# display related -------------------------------------------------------------
parser.add_argument('--display', action='store_true', default=False,
                    help='visualize things with visdom or not. default=False')
parser.add_argument('--display_freq', type=int, default=100, help='display frequency w.r.t. iterations. default=5')
# GPU related -----------------------------------------------------------------
parser.add_argument('--num_gpu', type=int, default=1, help='number of GPUs to use. default=1')
# network saving related ------------------------------------------------------
parser.add_argument('--save_freq', type=int, default=100,
                    help='network saving frequency w.r.t. epoch number. default=500')
parser.add_argument('--save_path', type=str, default='./training_result',
                    help='path to trained network. default=./training_result')
parser.add_argument('--save_name', type=str, default='', help='name for network saving')
# ETC -------------------------------------------------------------------------
parser.add_argument('--random_seed', type=int, help='manual seed')
parser.add_argument('--debug_print', action='store_true', default=False, help='print debug information')
# -----------------------------------------------------------------------------

options = parser.parse_args()

# seed
if options.random_seed is None:
    options.random_seed = random.randint(1, 10000)

# loss type
options.variational = options.model.find('VAE') != -1

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
                                     options.dataset.replace('|', '-'), socket.gethostname())
save_path = options.save_path
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
dataset_paths, mean_images = util.get_dataset_paths_and_mean_images(options.dataset, options.data_root, 'train')
dataset = VideoClipSets(dataset_paths, centered=False)
# TODO: find out the way to streaming data directly into GPU
dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=options.batch_size, shuffle=True,
                                         num_workers=options.workers)
for path in dataset_paths:
    print("Dataset from '%s'" % path)
debug_print('Data loader is ready')

# streaming buffer
tm_buffer_set = time.time()
# TODO: change it to pinned memory
input_batch = torch.FloatTensor(options.batch_size, options.nc, options.image_size, options.image_size)
recon_batch = torch.FloatTensor(options.batch_size, options.nc, options.image_size, options.image_size)
debug_print('Stream buffers are set: %.3f sec elapsed' % (time.time() - tm_buffer_set))

if cuda_available:
    debug_print('Start transferring to CUDA')
    tm_gpu_start = time.time()
    input_batch = input_batch.cuda()
    recon_batch = recon_batch.cuda()
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
util.mean_images = mean_images
debug_print('Utility library is ready')


# =============================================================================
# MODEL & LOSS FUNCTION
# =============================================================================
# create model instance
# TODO: load pretrained model
if 'AE-LTR' == options.model:
    model = AE_LTR(options.nc)
elif 'VAE-LTR' == options.model:
    model = VAE_LTR(options.nc)
elif 'AE' == options.model:
    model = AE(options.nc, options.nz, options.nf)
elif 'VAE' == options.model:
    model = VAE(options.nc, options.nz, options.nf)
assert model
print(options.model + ' is generated')
print(model)

# loss & criterion
our_loss = OurLoss(cuda_available)

# to gpu
if cuda_available:
    debug_print('Start transferring model to CUDA')
    tm_gpu_start = time.time()
    model.cuda()
    debug_print('Transfer to GPU: %.3f sec elapsed' % (time.time() - tm_gpu_start))


# =============================================================================
# TRAINING
# =============================================================================

print('Start training...')
model.train()

# TODO: add ADAGRAD as an option
# optimizer
optimizer = optim.Adagrad(model.parameters(),
                          lr=options.learning_rate,
                          weight_decay=options.l2_coef)

tm_data_load_total = 0
tm_iter_total = 0
tm_loop_start = time.time()
time_info = dict(cur_iter=0, train=0, visualize=0, ETC=0)

# TODO: modify iter_count and epoch range with pretrained model's metadata
iter_count = 0
recent_loss = 0
loss_info = dict()
train_info = dict(model=options.model, dataset=options.dataset, iter_count=0, total_loss=0, options=options_dict)
display_data_count = 0

for epoch in range(options.epochs):
    loss_per_epoch = []
    tm_cur_epoch_start = tm_cur_iter_start = time.time()
    for i, (data, setnames) in enumerate(dataloader, 0):

        # ============================================
        # DATA FEED
        # ============================================
        input_batch.data.resize_(data.size()).copy_(data)
        recon_batch.data.resize_(data.size())

        # ============================================
        # TRAIN
        # ============================================
        # forward
        tm_train_start = time.time()
        model.zero_grad()
        recon_batch, mu_batch, logvar_batch = model(input_batch)

        # backward
        loss, loss_detail = our_loss.calculate(recon_batch, input_batch, options, mu_batch, logvar_batch)
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
        display_data_count += 1
        tm_visualize_start = time.time()
        if options.display:
            # draw input/recon images
            win_images = util.draw_images(win_images, data, recon_batch.data, setnames)

            # draw graph at every drawing period
            if 0 == iter_count % options.display_freq:
                loss_info = {key: value / display_data_count for key, value in loss_info.items()}
                win_loss = util.viz_append_line_points(win_loss, loss_info, iter_count)
                loss_info = dict.fromkeys(loss_info, 0)

                time_info = {key: value / display_data_count for key, value in time_info.items()}
                win_time = util.viz_append_line_points(win_time, time_info, iter_count,
                                                       title='times at each iteration',
                                                       ylabel='time', xlabel='iterations')
                time_info = dict.fromkeys(time_info, 0)
                display_data_count = 0

        # print iteration's summary
        print('[%4d/%4d][%3d/%3d] Iter:%4d\t %s \tTotal time elapsed: %s'
              % (epoch + 1, options.epochs, i, len(dataloader), iter_count + 1, util.get_loss_string(loss_detail),
                 util.formatted_time(time.time() - tm_loop_start)))

        tm_visualize_consume = time.time() - tm_visualize_start

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
        # ===============================================
        tm_cur_iter_start = time.time()  # to measure the time of enumeration of the loop controller, set timer at here
        iter_count += 1

    print('====> Epoch %d is terminated: Epoch time is %s'
          % (epoch+1, util.formatted_time(time.time() - tm_cur_epoch_start)))

    # checkpoint w.r.t. epoch
    if 0 == (epoch+1) % options.save_freq:
        util.save_model(os.path.join(save_path, '%s_epoch_%03d.pth')
                        % (options.save_name, epoch+1), model.state_dict(), train_info, True)


#()()
#('')HAANJU.YOO
