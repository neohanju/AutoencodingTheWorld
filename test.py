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
from models import AE, VAE, AE_LTR, VAE_LTR, OurLoss
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
util.make_dir(save_path)

# visualization
if options.display:
    win_recon_cost = None


# =============================================================================
# DATA PREPARATION
# =============================================================================
dataset_paths, mean_images = util.get_dataset_paths_and_mean_images(options.dataset, options.data_root, 'test')
dataset = VideoClipSets([options.input_path])
dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=options.workers)
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
# MODEL
# =============================================================================
# create model instance
if 'AE_LTR' == options.model:
    model = AE_LTR(options.nc)
elif 'VAE_LTR' == options.model:
    model = VAE_LTR(options.nc)
elif 'AE' == options.model:
    model = AE(options.nc, options.nz, options.nf)
elif 'VAE' == options.model:
    model = VAE(options.nc, options.nz, options.nf)
assert model
model.load_state_dict(torch.load(options.model_path))
print(options.model + ' is loaded')
print(model)

# loss & criterions
our_loss = OurLoss(cuda_available)

# to gpu
if cuda_available:
    debug_print('Start transferring model to CUDA')
    tm_gpu_start = time.time()
    model.cuda()
    debug_print('Transfer to GPU: %.3f sec elapsed' % (time.time() - tm_gpu_start))


# =============================================================================
# TESTING
# =============================================================================
model.eval()
cur_dataset = None
cur_save_path = None
recon_cost_dict = dict(recon=0)
recon_costs = {}
sample_index = 0
for i, (data, setname) in enumerate(dataloader, 0):

    # for multiple datasets
    if cur_dataset is not setname[0]:
        cur_dataset = setname[0]
        recon_costs[cur_dataset] = []
        win_recon_cost = None
        print("Testing with '%s' dataset" % cur_dataset)
    assert cur_dataset is not None and cur_save_path is not None

    # data load
    input_batch.data.resize_(data.size()).copy_(data)
    recon_batch.data.resize_(data.size())

    # forward
    tm_forward_start = time.time()
    recon_batch, mu_batch, logvar_batch = model(input_batch)
    loss, loss_detail = our_loss.get(recon_x=recon_batch, x=input_batch, mu=mu_batch, logvar=logvar_batch)
    tm_forward_consume = time.time() - tm_forward_start

    # reconstruction cost
    recon_costs[cur_dataset].append(loss_detail['recon'])
    if options.display:
        recon_cost_dict['recon'] = loss_detail['recon']
        win_recon_cost = util.viz_append_line_points(win_recon_cost, recon_cost_dict, len(recon_costs[cur_dataset]),
                                                     'Reconstruction costs of %s' % cur_dataset)

# save reconstruciton costs
for (setname, costs) in recon_costs.items():
    cur_save_path = os.path.join(save_path, cur_dataset, '_recon_costs')
    util.make_dir(cur_save_path)
    util.file_print_recon_costs(os.path.join(cur_save_path, '%s_%s.txt' % (setname, options.model)))


# ()()
# ('')HAANJU.YOO
