import argparse
import os
import time
import random
import socket
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from models import DCGAN_Generator, DCGAN_Discriminator, OurLoss
from data import VideoClipSets
import utils as util
import math


def debug_print(arg):
    if not options.debug_print:
        return
    print(arg)

# =============================================================================
# OPTIONS
# =============================================================================
parser = argparse.ArgumentParser(description='Detecting abnormal behavior in videos')

# model related ---------------------------------------------------------------
parser.add_argument('--model', type=str, default='DCGAN', help='DCGAN')
parser.add_argument('--nc', type=int, default=10, help='number of input channel. default=10')
parser.add_argument('--nz', type=int, default=200, help='size of the latent z vector. default=100')
parser.add_argument('--nf', type=int, default=64, help='size of lowest image filters. default=64')
parser.add_argument('--l1_coef', type=float, default=0, help='coef of L1 regularization on the weights. default=0')
parser.add_argument('--l2_coef', type=float, default=0, help='coef of L2 regularization on the weights. default=0')
parser.add_argument('--var_loss_coef', type=float, default=1.0, help='balancing coef of vairational loss. default=0')
# training related ------------------------------------------------------------
parser.add_argument('--load_model_path', type=str, default="",
                    help='path of pretrained networks. If you put only one, that is regarded as the directory '
                         'containing networks. default=""')
parser.add_argument('--load_model_type', type=str, default="_latest")
parser.add_argument('--batch_size', type=int, default=64, help='input batch size. default=64')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train for. default=25')
parser.add_argument('--max_iter', type=int, default=150000, help='number of iterations to train for. default=150,000')
# data related ----------------------------------------------------------------
parser.add_argument('--dataset', type=str, required=True, nargs='+',
                    help="all | avenue | ped1 | ped2 | enter | exit. 'all' means using entire data")
parser.add_argument('--data_root', type=str, required=True, help='path to base folder of entire dataset')
parser.add_argument('--image_size', type=int, default=227, help='input image size (width=height). default=227')
parser.add_argument('--workers', type=int, default=2, help='noumber of data loading workers')
# optimization related --------------------------------------------------------
parser.add_argument('--optimizer', type=str, default='adagrad',
                    help='type of optimizer: adagrad | adam | asgd | sgd. default=adagrad')
parser.add_argument('--learning_rate', type=float, default=0.0002, help='learning rate. default=0.0002')
parser.add_argument('--learning_rate_decay', type=float, default=0, help='learning rate decay. default=0')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer. default=0.5')
# display related -------------------------------------------------------------
parser.add_argument('--display', action='store_true', default=False,
                    help='visualize things with visdom or not. default=False')
parser.add_argument('--display_freq', type=int, default=100, help='display frequency w.r.t. iterations. default=5')
parser.add_argument('--display_maxclip', action='store_true', default=False,
                    help='loss larger then 10,000 will not be dwarwn at the loss graph')
# GPU related -----------------------------------------------------------------
parser.add_argument('--num_gpu', type=int, default=0,
                    help='number of GPUs to use. It will be ignored when gpu_ids options is given. default=0')
parser.add_argument('--gpu_ids', type=int, default=[], nargs='*',
                    help='Indices of GPUs in use. If you give this, num_gpu option input will be ignored. default=[]')
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

options.continue_train = options.load_model_path != ''
if options.continue_train:
    options.load_model_path = options.load_model_path+ '/' + os.path.basename(options.load_model_path)
    print("load model from '%s'" % os.path.basename(options.load_model_path))

    # load metadata
    # you can use _netG instead. they are same thing.
    metadata_path = options.load_model_path + '_netD' + options.load_model_type+'.json'
    print(metadata_path)
    prev_train_info, options = util.load_metadata(metadata_path, options)


    print('Loaded model was trained %d epochs with %d iterations'
          % (prev_train_info['epoch_count'], prev_train_info['iter_count']))
    print('Starting loss : %.3f' % prev_train_info['total_loss'])


# loss type
options.variational = options.model.find('VAE') != -1

# latent space size
if options.model.find('-LTR') != -1:
    options.z_size = [128, 13, 13]
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
dataset_paths, mean_images = util.get_dataset_paths_and_mean_images(options.dataset, options.data_root, 'train')
dataset = VideoClipSets(dataset_paths, centered=False)
dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=options.batch_size, shuffle=True,
                                         num_workers=options.workers, pin_memory=True)
for path in dataset_paths:
    print("Dataset from '%s'" % path)
debug_print('Data loader is ready')

# streaming buffer
tm_buffer_set = time.time()
input_batch = torch.FloatTensor(options.batch_size, options.nc, options.image_size, options.image_size).pin_memory()
noise = torch.FloatTensor(options.batch_size, options.nz, 1, 1)
fixed_noise = torch.FloatTensor(options.batch_size, options.nz, 1, 1).normal_(0, 1)
label = torch.FloatTensor(options.batch_size)
real_label = 1
fake_label = 0


debug_print('Stream buffers are set: %.3f sec elapsed' % (time.time() - tm_buffer_set))


# for utility library
util.target_sample_index = 0
util.target_frame_index = int(options.nc / 2)
util.mean_images = mean_images
debug_print('Utility library is ready')


# =============================================================================
# MODEL & LOSS FUNCTION
# =============================================================================
# create model instance
if 'DCGAN' == options.model:
    netG = DCGAN_Generator(options.nc, options.nz, options.nf)
    netD = DCGAN_Discriminator(options.nc, options.nf)
assert netG
if options.continue_train:
    # todo load model path for netG
    netG.load_state_dict(torch.load(options.load_model_path + '_netG' + options.load_model_type + '.pth'))
    print(options.model + ' is loaded')
else:
    print(options.model + ' is generated')
print(netG)

assert netD
if options.continue_train:
    # todo load model path for netD
    netD.load_state_dict(torch.load(options.load_model_path + '_netD' + options.load_model_type + '.pth'))
    print(options.model + ' is loaded')
else:
    print(options.model + ' is generated')
print(netD)


# loss & criterion
our_loss = OurLoss()


# to gpu


if cuda_available:
    debug_print('Start transferring model to CUDA')
    tm_gpu_start = time.time()
    input_batch, label = input_batch.cuda(async=True), label.cuda(async=True)
    noise, fixed_noise = noise.cuda(async=True), fixed_noise.cuda(async=True)

    # # multi-GPU
    # model = torch.nn.DataParallel(model, device_ids=options.gpu_ids)
    netG.cuda()
    netD.cuda()

    debug_print('Transfer to GPU: %.3f sec elapsed' % (time.time() - tm_gpu_start))

tm_to_variable = time.time()
input_batch = Variable(input_batch)
label = Variable(label)
noise = Variable(noise)
fixed_noise = Variable(fixed_noise)


debug_print('To Variable for Autograd: %.3f sec elapsed' % (time.time() - tm_to_variable))
print('input_batch:', input_batch.size())

debug_print('Data streaming is ready')

# =============================================================================
# TRAINING
# =============================================================================

print('Start training...')

# optimizer
if 'adagrad' == options.optimizer:
    optimizerD = optim.Adagrad(netD.parameters(), lr=options.learning_rate, lr_decay=options.learning_rate_decay,
                              weight_decay=options.l2_coef)
    optimizerG = optim.Adagrad(netG.parameters(), lr=options.learning_rate, lr_decay=options.learning_rate_decay,
                               weight_decay=options.l2_coef)
elif 'adam' == options.optimizer:
    optimizerD = optim.Adam(netD.parameters(), lr=options.learning_rate, betas=(options.beta1, 0.999),
                            weight_decay=options.l2_coef)
    optimizerG = optim.Adam(netG.parameters(), lr=options.learning_rate, betas=(options.beta1, 0.999),
                            weight_decay=options.l2_coef)

elif 'asgd' == options.optimizer:
    optimizerD = optim.ASGD(netD.parameters(), lr=options.learning_rate, weight_decay=options.l2_coef)
    optimizerG = optim.ASGD(netG.parameters(), lr=options.learning_rate, weight_decay=options.l2_coef)
elif 'sgd' == options.optimizer:
    optimizerD = optim.SGD(netD.parameters(), lr=options.learning_rate, weight_decay=options.l2_coef)
    optimizerG = optim.SGD(netD.parameters(), lr=options.learning_rate, weight_decay=options.l2_coef)
assert optimizerD
assert optimizerG

# timer related
tm_data_load_total = 0
tm_iter_total = 0
tm_loop_start = time.time()
time_info = dict(cur_iter=0, train=0, visualize=0, ETC=0)

# counters
iter_count = 0
recent_loss = 0
display_data_count = 0

# for logging
loss_info = dict()
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

# main loop of training
for epoch in range(options.epochs):
    loss_per_epoch = []
    tm_cur_epoch_start = tm_cur_iter_start = time.time()
    for i, (real_cpu, setname, _) in enumerate(dataloader, 1):

        # ============================================
        # DATA FEED
        # ============================================
        if real_cpu.size() != input_batch.data.size():
            # input_batch.data.resize_(data.size())
            # recon_batch.data.resize_(data.size())
            # this will be deprecated by 'last_drop' attributes of dataloader
            continue

        # ============================================
        # Update D network
        # ============================================
        # train with real
        tm_train_start = time.time()
        netD.zero_grad()
        batch_size = real_cpu.size(0)
        input_batch.data.copy_(real_cpu)
        label.data.fill_(real_label)

        output = netD(input_batch)
        errD_real = our_loss.calculate_GAN(output, label)
        errD_real.backward()
        D_x = output.data.mean()

        # train with fake
        noise.data.resize_(batch_size, options.nz, 1, 1)
        noise.data.normal_(0, 1)
        fake = netG(noise)
        label.data.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = our_loss.calculate_GAN(output, label)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()

        # ============================================
        # Update G network
        # ============================================
        netG.zero_grad()
        label.data.fill_(real_label)
        output = netD(fake)
        errG, loss_per_batch = our_loss.calculate_GAN(output, label, True)
        errG.backward()
        D_G_z2 = output.data.mean()
        optimizerG.step()

        loss_info['Loss_D'] = errD.data[0]
        loss_info['Loss_G'] = errG.data[0]
        loss_info['D(x)'] = D_x
        loss_info['D(G(z))'] = D_G_z1/D_G_z2

        best_sample_indexes = util.find_best_sample_indexes(loss_per_batch)
        # ============================================
        # Save model and visualize model
        # ============================================

        tm_train_iter_consume = time.time() - tm_train_start
        time_info['train'] += tm_train_iter_consume

        # ============================================
        # VISUALIZATION
        # ============================================
        display_data_count += 1
        tm_visualize_start = time.time()
        if options.display:
            # draw input/recon images
            print(real_cpu.size())
            win_images = util.draw_images_GAN(win_images, fake, best_sample_indexes)

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
        print('[%4d/%4d][%3d/%3d] Iter:%4d\t Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f '
              '\tTotal time elapsed: %s'

              % (epoch+1, options.epochs, i, len(dataloader), iter_count+1,
                 errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2,
                 util.formatted_time(time.time() - tm_loop_start)))

        tm_visualize_consume = time.time() - tm_visualize_start

        # ============================================
        # NETWORK BACK-UP
        # ============================================
        # save network and meta data
        train_info['iter_count'] = iter_count
        train_info['total_loss'] = recent_loss
        train_info['epoch_count'] = epoch
        util.save_model(os.path.join(save_path, options.save_name + '_netD_latest.pth'), netD.state_dict(), train_info)
        util.save_model(os.path.join(save_path, options.save_name + '_netG_latest.pth'), netG.state_dict(), train_info)

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
                        % (options.save_name + '_netD', epoch + 1), netD.state_dict(), train_info, True)
        util.save_model(os.path.join(save_path, '%s_epoch_%03d.pth')
                        % (options.save_name + '_netG', epoch + 1), netG.state_dict(), train_info, True)


#()()
#('')HAANJU.YOO
