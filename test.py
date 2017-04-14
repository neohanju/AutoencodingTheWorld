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
from torch.autograd import Variable
from data import VideoClipSets
from models import AE, VAE, AE_LTR, VAE_LTR
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
parser.add_argument('--model_path', type=str, required=True, help='path to trained model file')
# data related ----------------------------------------------------------------
parser.add_argument('--dataset', type=str, required=True,
                    help="all | avenue | ped1 | ped2 | enter | exit. 'all' means using entire data")
parser.add_argument('--data_root', type=str, required=True, help='path to base folder of entire dataset')
parser.add_argument('--batch_size', type=int, default=1, help='batch for testing. default=1')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
# output related --------------------------------------------------------------
parser.add_argument('--save_path', type=str, required=True, default='./testing_result',
                    help='path for saving test results')
# display related -------------------------------------------------------------
parser.add_argument('--display', action='store_true', default=False,
                    help='visualize things with visdom or not. default=False')
# GPU related -----------------------------------------------------------------
parser.add_argument('--num_gpu', type=int, default=1, help='number of GPUs to use. default=1')

options = parser.parse_args()

# seed
if options.random_seed is None:
    options.random_seed = random.randint(1, 10000)

# load options from metadata
train_info = np.load(os.path.join(os.path.dirname(options.model_path), 'train_info.npy')).item()
options.model = train_info.model
options.nc = train_info.options.nc
options.nz = train_info.options.nz
options.nf = train_info.options.nf
options.image_size = train_info.options.image_size

print(options)


# =============================================================================
# INITIALIZATION PROCESS
# =============================================================================

cuda_available = torch.cuda.is_available()
torch.manual_seed(options.random_seed)
if cuda_available:
    torch.cuda.manual_seed_all(options.random_seed)

# cudnn
cudnn.benchmark = True

# result saving
save_path = os.path.join(options.save_path, options.model)
try:
    os.makedirs(save_path)
except OSError:
    debug_print('WARNING: Cannot make saving folder')
    pass

# =============================================================================
# DATA PREPARATION
# =============================================================================

dataset_paths, mean_images = util.get_dataset_paths_and_mean_images(options.dataset, options.data_root, 'test')
dataset = VideoClipSets([options.input_path])
dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=options.batch_size, shuffle=False,
                                         num_workers=options.workers)
print('Data loader is ready')

# streaming buffer
tm_buffer_set = time.time()
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

print('Data streaming is ready')

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
if 'AE_LTR' == options.model:
    model = AE_LTR(options.nc)
elif 'VAE_LTR' == options.model:
    model = VAE_LTR(options.nc)
elif 'AE' == options.model:
    model = AE(options.nc, options.nz, options.nf)
elif 'VAE' == options.model:
    model = VAE(options.nc, options.nz, options.nf)
assert model
print(options.model + ' is generated')

# ()()
# ('')HAANJU.YOO
