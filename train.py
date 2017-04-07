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
parser.add_argument('--data_root', type=str, required=True, help='path to base folder of entire dataset')
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
cuda_available = torch.cuda.is_available()

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

torch.manual_seed(options.random_seed)
if cuda_available:
    torch.cuda.manual_seed_all(options.random_seed)

# network saving
save_path = options.save_path + '/' + options.model
try:
    os.makedirs(save_path)
except OSError:
    debug_print('WARNING: Cannot make network saving folder')
    pass


# =============================================================================
# DATA PREPARATION
# =============================================================================

frames_per_sample = 10

# set data loader
options.dataset.replace(' ', '')  # remove white space
dataset_paths = []
if 'all' == options.dataset:
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

# =============================================================================
# MODEL & LOSS FUNCTION
# =============================================================================

# create model instance
if 'AE' == options.model:
    model = AE(num_in_channels=frames_per_sample, z_size=options.nz, num_filters=options.nf)
elif 'VAE' == options.model:
    model = VAE(num_in_channels=frames_per_sample, z_size=options.nz, num_filters=options.nf)
assert model
print(options.model + ' is generated')

# criterion
reconstruction_loss = nn.MSELoss()
variational_loss = nn.KLDivLoss()

# to gpu
if cuda_available:
    debug_print('Start transferring to CUDA')
    tm_gpu_start = time.time()
    model.cuda()
    reconstruction_loss.cuda()
    variational_loss.cuda()
    debug_print('Transfer to GPU: %.3f sec elapsed' % (time.time() - tm_gpu_start))

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
    if 0.0 != options.l1_coef:
        reg_l1_loss = options.l1_coef * torch.norm(params, 1)
        # params.data -= options.learning_rate * params.grad.data
        loss += reg_l1_loss
    if 0.0 != options.l2_coef:
        reg_l2_loss = options.l2_coef * torch.norm(params, 2) ^ 2 / 2
        loss += reg_l2_loss

    return loss


def save_model(filename):
    torch.save(model.state_dict(), '%s/%s' % (options.save_path, filename))
    print('Model is saved at ' + filename)


# =============================================================================
# OPTIMIZATION
# =============================================================================

print('Start training...')
model.train()

optimizer = optim.Adam(model.parameters(), lr=options.learning_rate, betas=(options.beta1, 0.999))

total_loss_history = []
recon_loss_history = []
variational_loss_history = []
reg_l1_loss_history = []
reg_l2_loss_history = []

tm_data_load_total = 0
tm_iter_total = 0
tm_loop_start = time.time()

iter_count = 0

for epoch in range(options.epochs):
    tm_cur_iter_start = time.time()
    for i, data in enumerate(dataloader, 0):

        # data feed
        # input_batch = Variable(data)
        # if cuda_available:
        #     input_batch = input_batch.cuda()
        batch_size = data.size(0)
        input_batch.data.resize_(data.size()).copy_(data)
        recon_batch.data.resize_(data.size())

        tm_train_start = time.time()

        # forward
        model.zero_grad()
        recon_batch, mu_batch, logvar_batch = model(input_batch)

        # backward
        loss = loss_function(recon_x=recon_batch, x=input_batch, mu=mu_batch, logvar=logvar_batch)
        loss.backward()
        total_loss_history.append(loss.data[0])
        optimizer.step()

        tm_train_iter_consume = time.time() - tm_train_start

        # visualize
        tm_visualize_start = time.time()
        # TODO: visualize input / reconstruction pair
        # TODO: find input index and set latent vector of that index
        tm_visualize_consume = time.time() - tm_visualize_start

        recon_loss_history.append(mse_loss)
        variational_loss_history.append(kld_loss)
        reg_l1_loss_history.append(reg_l1_loss)
        reg_l2_loss_history.append(reg_l2_loss)

        print('[%02d/%02d][%04d/%04d] Iter:%06d Total: %.4f Recon: %.4f Var: %.4f L1: %.4f L2: %.4f'
              % (epoch, options.epochs, i, len(dataloader), iter_count,
                 loss.data[0], mse_loss, kld_loss, reg_l1_loss, reg_l2_loss))

        # checkpoint w.r.t. iteration number
        iter_count += 1
        if 0 == iter_count % options.save_freq:
            save_model('%s_%s_iter_%03d.pth' % (options.dataset, options.model, iter_count))

        tm_iter_consume = time.time() - tm_cur_iter_start
        print('\tTime consume (secs) Total: %.3f CurIter: %.3f, Train: %.3f, Vis.: %.3f ETC: %.3f'
              % (time.time() - tm_loop_start, tm_iter_consume, tm_train_iter_consume, tm_visualize_consume,
                 tm_iter_consume - tm_train_iter_consume - tm_visualize_consume))
        tm_cur_iter_start = time.time()

    print('====> Epoch %d is ternimated: Total loss is %f' % (epoch, total_loss_history[-1]))
    # checkpoint w.r.t. epoch
    save_model('%s_%s_epoch_%03d.pth' % (options.dataset, options.model, epoch))


#()()
#('')HAANJU.YOO

