import importlib
import argparse
import os
import random
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.init
import torch.utils.data
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

from models import AE, VAE
from data import VideoClipSets


# =============================================================================
# OPTIONS
# =============================================================================
parser = argparse.ArgumentParser(description='Detecting abnormal behavior in videos')

# model related ---------------------------------------------------------------
parser.add_argument('--model', type=str, required=True, help='AE | AE_NEW | VAE | VAE_NEW')
parser.add_argument('--nz', type=int, default=200, help='size of the latent z vector. default=100')
parser.add_argument('--nf', type=int, default=64, help='size of lowest image filters. default=64')
parser.add_argument('--l1_coef', type=float, default=0, help='coef of L1 regularization on the weights. default=0')
parser.add_argument('--l2_coef', type=float, default=0, help='coef of L2 regularization on the weights. default=0')
# training related ------------------------------------------------------------
parser.add_argument('--batch_size', type=int, default=64, help='input batch size. default=64')
parser.add_argument('--epochs', type=int, default=25, help='number of epochs to train for. default=25')
parser.add_argument('--max_iter', type=int, default=150000, help='number of iterations to train for. default=150,000')
parser.add_argument('--partial_learning', type=float, default=1,
                    help='ratio of partial data for training. At least one sample from each file. default=1')
parser.add_argument('--continue_train', action='store_true', default=False,
                    help='load the latest model to continue the training, default=False')
# data related ----------------------------------------------------------------
parser.add_argument('--dataset', type=str, required=True,
                    help="all | avenue | ped1 | ped2 | enter | exit. 'all' means using entire data")
parser.add_argument('--data_root', type=str, required=True, help='path to base folder of entire datasets')
parser.add_argument('--image_size', type=int, default=227, help='input image size (width=height). default=227')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
# optimization related --------------------------------------------------------
parser.add_argument('--optimiser', type=str, default='adam', help='type of optimizer: adagrad | adam')
parser.add_argument('--learning_rate', type=float, default=0.0002, help='learning rate. default=0.0002')
parser.add_argument('--weight_decay', type=float, default=0.0005,
                    help='weight decay coefficient for regularization. default=0.0005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer. default=0.5')
# display related -------------------------------------------------------------
parser.add_argument('--display', action='store_true', default=False,
                    help='visualize things with visdom or not. default=False')
parser.add_argument('--display_freq', type=int, default=5, help='display frequency w.r.t. iterations. default=5')
# GPU related -----------------------------------------------------------------
parser.add_argument('--cpu_only', action='store_true', help='CPU only (useful if GPU memory is too low)')
parser.add_argument('--num_gpu', type=int, default=1, help='number of GPUs to use. default=1')
# network saving related ------------------------------------------------------
parser.add_argument('--save_freq', type=int, default=500,
                    help='network saving frequency w.r.t. iteration number. default=500')
parser.add_argument('--save_path', type=str, default='./training_result',
                    help='path to trained network. default=./training_result')
parser.add_argument('--save_name', type=str, default='', help='name for network saving')
# ETC -------------------------------------------------------------------------
parser.add_argument('--random_seed', type=int, help='manual seed')
parser.add_argument('--debug_print', action='store_true', default=False, help='print debug information')
# -----------------------------------------------------------------------------

options = parser.parse_args()


def debug_print(arg):
    if not options.debug_print:
        return
    print(arg)

# cuda
options.cuda = not options.cpu_only and torch.cuda.is_available()
if torch.cuda.is_available() and not options.cuda:
    debug_print('WARNING: You have a CUDA device, so you should probably run without --cpu_only')

# TODO: visdom package handling

# seed
if options.random_seed is None:
    options.random_seed = random.randint(1, 10000)

if options.model.find('VAE') != -1:
    options.variational = True

print(options)


# =============================================================================
# INITIALIZATION PROCESS
# =============================================================================

# torch setup

torch.manual_seed(options.random_seed)
if options.cuda:
    torch.cuda.manual_seed_all(options.random_seed)

# network saving
save_path = options.save_path + '/' + options.model
try:
    os.makedirs(save_path)
except OSError:
    debug_print('WARNING: Cannot make network saving folder')
    pass

tm_start = time.time()
tm_epoch = time.time()
tm_iter = time.time()
tm_optimize = time.time()
tm_forward = time.time()


# =============================================================================
# DATA PREPARATION
# =============================================================================

frames_per_sample = 10

# set data loader
options.dataset.replace(' ', '')  # remove white space
dataset_paths = []
if options.dataset is 'all':
    options.dataset = 'avenue|ped1|ped2|enter|exit'
if 'avenue' in options.dataset:
    dataset_paths.append(options.data_root + '/avenue/train')
# TODO: tokenize dataset string with '|'

dataset = VideoClipSets(dataset_paths)
dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=options.batch_size, shuffle=True,
                                         num_workers=options.workers)
print('Data loader is ready')


# streaming buffer
tm_buffer_set = time.time()
input_batch = torch.FloatTensor(options.batch_size, frames_per_sample, options.image_size, options.image_size)
recon_batch = torch.FloatTensor(options.batch_size, frames_per_sample, options.image_size, options.image_size)
debug_print('Stream buffers are set: {} sec elapsed'.format(time.time() - tm_buffer_set))

if options.cuda:
    debug_print('Start transferring to CUDA')
    tm_gpu_start = time.time()
    input_batch = input_batch.cuda()
    recon_batch = recon_batch.cuda()
    debug_print('Transfer to GPU: {} sec elapsed'.format(time.time() - tm_gpu_start))

tm_to_variable = time.time()
input_batch = Variable(input_batch)
recon_batch = Variable(recon_batch)
debug_print('To Variable for Autograd: {} sec elapsed'.format(time.time() - tm_to_variable))

print('Data streaming is ready')

# =============================================================================
# MODEL & LOSS FUNCTION
# =============================================================================

# create model instance
if options.model == 'AE':
    model = AE(frames_per_sample, options.nz, options.nf)
elif options.model == 'VAE':
    model = VAE(frames_per_sample, options.nz, options.nf)
assert model
print('{} is generated'.format(options.model))

# criterions
reconstruction_loss = nn.MSELoss()
variational_loss = nn.KLDivLoss()

# to gpu
if options.cuda:
    debug_print('Start transferring to CUDA')
    tm_gpu_start = time.time()
    model.cuda()
    reconstruction_loss.cuda()
    variational_loss.cuda()
    debug_print('Transfer to GPU: {} sec elapsed'.format(time.time() - tm_gpu_start))

# for display
mse_loss, kld_loss, reg_l1_loss, reg_l2_loss = 0, 0, 0, 0
params = model.parameters()


def loss_function(recon_x, x, mu=None, logvar=None):
    # thanks to Autograd, you can train the net by just summing-up all losses and propagating them
    mse_loss = reconstruction_loss(recon_x, x)
    loss = mse_loss
    if options.variational:
        assert mu is not None and logvar is not None
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        kld_loss = torch.sum(kld_element).mul_(-0.5)
        loss += kld_loss
    if options.coefL1 != 0.0:
        reg_l1_loss = options.l1_coef * torch.norm(params, 1)
        # params.data -= options.learning_rate * params.grad.data
        loss += reg_l1_loss
    if options.coefL2 != 0.0:
        reg_l2_loss = options.l2_coef * torch.norm(params, 2) ^ 2 / 2
        loss += reg_l2_loss

    return loss


# =============================================================================
# OPTIMIZATION
# =============================================================================
print('Start training...')
model.train()

optimizer = optim.Adam(model.parameters(), lr=options.learning_rate, betas=(options.beta1, 0.999))
#
iter_count = 0
for epoch in range(options.epochs):
    train_loss = 0;
    for iter, data in enumerate(dataloader, 0):

        # data feed
        batch_size = data.size(0)
        input_batch.data.resize_(data.size()).copy_(data)
        recon_batch.data.resize_(data.size())

        # forward
        model.zero_grad()
        recon_batch, mu_batch, logvar_batch = model(input_batch)

        # backward
        loss = loss_function(recon_batch, input_batch, mu_batch, logvar_batch)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()

        # visualize
        # TODO: visualize input / reconstruction pair
        # TODO: find input index and set latent vector of that index


#()()
#('')HAANJU.YOO

