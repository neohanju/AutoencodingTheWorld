import argparse
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.init
import torch.utils.data
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from models import AE, VAE
from data import VideoClipSets


def debug_print(arg):
    if not options.debug_print:
        return
    print(arg)


# =============================================================================
# OPTIONS
# =============================================================================
parser = argparse.ArgumentParser(description='Detecting abnormal behavior in videos')

# model related ---------------------------------------------------------------
parser.add_argument('--model', type=str, required=True, help='AE | AE_NEW | VAE | VAE_NEW')
parser.add_argument('--model_path', type=str, required=True, help='path to trained model file')
# data related ----------------------------------------------------------------
parser.add_argument('--input_path', type=str, required=True, help='path to a folder containing input data')
parser.add_argument('--batch_size', type=int, default=1, help='batch for testing. default=1')
parser.add_argument('--image_size', type=int, default=227, help='input image size (width=height). default=227')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
# output related --------------------------------------------------------------
parser.add_argument('--output_path', type=str, required=True, default=2, help='path for saving test results')
# display related -------------------------------------------------------------
parser.add_argument('--display', action='store_true', default=False,
                    help='visualize things with visdom or not. default=False')
# GPU related -----------------------------------------------------------------
parser.add_argument('--num_gpu', type=int, default=1, help='number of GPUs to use. default=1')

options = parser.parse_args()

# seed
if options.random_seed is None:
    options.random_seed = random.randint(1, 10000)

if options.model.find('VAE') != -1:
    options.variational = True

print(options)


# =============================================================================
# INITIALIZATION PROCESS
# =============================================================================

cuda_available = torch.cuda.is_available()

torch.manual_seed(options.random_seed)
if cuda_available:
    torch.cuda.manual_seed_all(options.random_seed)

cudnn.benchmark = True

# network saving
save_path = options.save_path + '/' + options.model
try:
    os.makedirs(save_path)
except OSError:
    debug_print('WARNING: Cannot make saving folder')
    pass

# visualization
if options.display:
    from visdom import Visdom
    viz = Visdom()


# =============================================================================
# DATA PREPARATION
# =============================================================================

frames_per_sample = options.nc

dataset = VideoClipSets([options.input_path])
dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=options.batch_size, shuffle=True,
                                         num_workers=options.workers)
print('Data loader is ready')

# streaming buffer
tm_buffer_set = time.time()
input_batch = torch.FloatTensor(options.batch_size, frames_per_sample, options.image_size, options.image_size)
recon_batch = torch.FloatTensor(options.batch_size, frames_per_sample, options.image_size, options.image_size)
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

viz_target_frame_index = int(options.nc / 2)


def sample_batch_to_image(batch_data):
    single_image = batch_data[0, viz_target_frame_index].cpu().numpy()
    # un-normalize
    return np.uint8(single_image[np.newaxis, :, :].repeat(3, axis=0))

print('Data streaming is ready')


# =============================================================================
# MODEL & LOSS FUNCTION
# =============================================================================

# create model instance
# TODO: load pretrained model
if 'AE' == options.model:
    model = AE(num_in_channels=frames_per_sample, z_size=options.nz, num_filters=options.nf)
elif 'VAE' == options.model:
    model = VAE(num_in_channels=frames_per_sample, z_size=options.nz, num_filters=options.nf)
assert model
print(options.model + ' is generated')

# ()()
# ('')HAANJU.YOO
