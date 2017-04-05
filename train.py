import importlib
import argparse
import os
import random
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.init
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
parser.add_argument('--sparse', action='store_ture', default=False,
                    help='assign sparsity constraint on the latent variable')
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
# -----------------------------------------------------------------------------

options = parser.parse_args()

# cuda
options.cuda = not options.cpu_only and torch.cuda.is_available()
if torch.cuda.is_available() and not options.cuda:
    print('WARNING: You have a CUDA device, so you should probably run without --cpu_only')

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
    print('WARNING: Cannot make network saving folder. Training result will be discarded.')
    pass

tm_start = time.time()
tm_epoch = time.time()
tm_iter = time.time()
tm_data = time.time()
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

dataloader = VideoClipSets(paths=dataset_paths,
                           batch_size=options.batch_size,
                           shuffle=True,
                           num_workers=options.workers)

# streaming buffer
input = torch.FloatTensor(options.batch_size, frames_per_sample, options.image_size, options.image_size)
recon = torch.FloatTensor(options.batch_size, frames_per_sample, options.image_size, options.image_size)

if options.cuda:
    input = input.cuda()
    recon = recon.cuda()

input = Variable(input)
recon = Variable(recon)
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
regularization_loss = nn.L1Loss()

# to gpu
if options.cuda:
    model.cuda()
    reconstruction_loss.cuda()
    variational_loss.cuda()
    regularization_loss.cuda()

# for display
mse_loss, kld_loss, sparse_loss = 0, 0

def loss_function(recon_x, x, mu, logvar):
    mse_loss = reconstruction_loss(recon_x, x)
    loss = mse_loss
    if options.variational:
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        kld_loss = torch.sum(kld_element).mul_(-0.5)
        loss += kld_loss
    if options.sparse:
        sparse_loss = regularization_loss(mu, 0)
        loss += sparse_loss

    return loss


# =============================================================================
# OPTIMIZATION
# =============================================================================
optimizer = optim.adam(model.parameters(), lr=options.learning_rate, betas=(options.beta1, 0.999))

iter_count = 0
for epoch in range(options.epochs):
    for iter, data in enumerate(dataloader, 0):

        batch_size = data.size(0)
        input.data.resize_(data.size()).copy_(data)
        recon.data.resize_(data.size())

        model.zero_grad()
        recon = model(input)



#()()
#('')HAANJU.YOO

