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
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from models import AE, VAE


# =============================================================================
# OPTIONS
# =============================================================================
parser = argparse.ArgumentParser(description='Detecting abnormal behavior in videos')

# model related
parser.add_argument('--model', type=str, required=True, help='AE | AE_NEW | VAE | VAE_NEW')
parser.add_argument('--nz', type=int, default=200, help='size of the latent z vector. default=100')
parser.add_argument('--nf', type=int, default=64, help='size of lowest image filters. default=64')
parser.add_argument('--l1_coef', type=float, default=0, help='coef of L1 regularization on the weights. default=0')
parser.add_argument('--l2_coef', type=float, default=0, help='coef of L2 regularization on the weights. default=0')
parser.add_argument('--sparse', action='store_ture', default=False,
                    help='assign sparsity constraint on the latent variable')

# training related
parser.add_argument('--batchSize', type=int, default=64, help='input batch size. default=64')
parser.add_argument('--epochs', type=int, default=25, help='number of epochs to train for. default=25')
parser.add_argument('--max_iter', type=int, default=150000, help='number of iterations to train for. default=150,000')
parser.add_argument('--partial_learning', type=float, default=1,
                    help='ratio of partial data for training. At least one sample from each file. default=1')
parser.add_argument('--continue_train', action='store_true', default=False,
                    help='load the latest model to continue the training, default=False')

# data loading related
parser.add_argument('--dataset', type=str, required=True,
                    help='all | CUHK | UCSD_PED1 | UCSD_PED2 | Subway_enter | Subway_exit. all means using entire data')
parser.add_argument('--data_root', type=str, required=True, help='path to base folder of entire datasets')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')

# optimization related
parser.add_argument('--optimiser', type=str, default='adam', help='type of optimizer: adagrad | adam')
parser.add_argument('--learning_rate', type=float, default=0.0002, help='learning rate. default=0.0002')
parser.add_argument('--weight_decay', type=float, default=0.0005,
                    help='weight decay coefficient for regularization. default=0.0005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer. default=0.5')

# display related
parser.add_argument('--display', action='store_true', default=False,
                    help='visualize things with visdom or not. default=False')
parser.add_argument('--display_freq', type=int, default=5, help='display frequency w.r.t. iterations. default=5')

# GPU related
parser.add_argument('--cpu_only', action='store_true', help='CPU only (useful if GPU memory is too low)')
parser.add_argument('--num_gpu', type=int, default=1, help='number of GPUs to use. default=1')

# network saving related
parser.add_argument('--save_freq', type=int, default=500,
                    help='network saving frequency w.r.t. iteration number. default=500')
parser.add_argument('--save_path', type=str, default='./training_result',
                    help='path to trained network. default=./training_result')
parser.add_argument('--save_name', type=str, default='', help='name for network saving')

# ETC
parser.add_argument('--manualSeed', type=int, help='manual seed')

options = parser.parse_args()
options.cuda = not options.cpu_only and torch.cuda.is_available()
if torch.cuda.is_available() and not options.cuda:
    print("WARNING: You have a CUDA device, so you should probably run without --cpu_only")

print(options)

if options.model.find('VAE') != -1:
    options.variational = True


# =============================================================================
# INITIALIZATION PROCESS
# =============================================================================

# torch setup
torch.manual_seed(options.seed)
if options.cuda:
    torch.cuda.manual_seed(options.seed)

# network saving
save_path = options.save_path + '/' + options.model
try:
    os.makedirs(save_path)
except OSError:
    print("WARNING: Cannot make network saving folder. Training result will be discarded.")
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
num_frames_per_sample = 10



# =============================================================================
# MODEL & LOSS FUNCTION
# =============================================================================
if options.model == 'AE':
    model = AE(num_frames_per_sample, options.nz, options.nf)
elif options.model == 'VAE':
    model = VAE(num_frames_per_sample, options.nz, options.nf)

# criterions
reconstruction_loss = nn.MSELoss()
variational_loss = nn.KLDivLoss()
regularization_loss = nn.L1Loss()

def loss_function(recon_x, x, mu, logvar):
    loss = reconstruction_loss(recon_x, x)
    if options.variational:
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element).mul_(-0.5)
        loss += KLD
    if options.sparse:

    return loss

#()()
#('')HAANJU.YOO

