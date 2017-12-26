import argparse
import os
import random
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import utils
import time
from torch.autograd import Variable
import torch.optim as optim

import Datasets.RGBImageSet_augmented as dset
import Models.AutoEncoder as model
import PathManager as pm

# ======================================================================================================================
# OPTIONS
# ======================================================================================================================
parser = argparse.ArgumentParser()
# paths
parser.add_argument('--dataset', default='MNIST', help='what is dataset?')
parser.add_argument('--dataroot', default=os.path.join(pm.datasetroot, 'train_augmented'),
                    help='root path to dataset')
parser.add_argument('--net', default='', help="path of networks.(to continue training)")
parser.add_argument('--outf', default='./output', help="folder to output images and model checkpoints")
# model
parser.add_argument('--model', type=str, default='InfoGAN', help='Model name')
parser.add_argument('--nc', type=int, default=3, help='number of input channel.')
parser.add_argument('--nz', type=int, default=400, help='latent size.')
parser.add_argument('--nf', type=int, default=64, help='number of filter.(first layer)')
# GPU
parser.add_argument('--cuda', default=True, action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=2, help='number of GPUs to use')
parser.add_argument('--display', default=True, help='display options. default:False. NOT IMPLEMENTED')
parser.add_argument('--workers', type=int, default=1, help='number of data loading workers')
# optimizer
parser.add_argument('--iteration', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam.')
# metadata for testing
parser.add_argument('--batchSize', type=int, default=20, help='input batch size')
parser.add_argument('--imageSize', type=int, default=224, help='the height / width of the input image to network')
# etc
parser.add_argument('--seed', type=int, help='manual seed')

options = parser.parse_args()
print(options)

# output directory
try:
    os.makedirs(options.outf)
except OSError:
    pass

# random seed
if options.seed is None:
    options.seed = random.randint(1, 10000)
print("Random Seed: ", options.seed)
random.seed(options.seed)
torch.manual_seed(options.seed)

# cuda related
if torch.cuda.is_available() and not options.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
if options.cuda:
    torch.cuda.manual_seed(options.seed)
torch.backends.cudnn.benchmark = True
cudnn.benchmark = True

# visualization (visdom)
win_recon_cost = None


# ======================================================================================================================
# MAIN LOOP
# ======================================================================================================================
cnt = 0
for fold_number in range(10):
    dataloader = torch.utils.data.DataLoader(
        dset.RGBImageSet_augmented(options.dataroot, op_type='train', centered=False, fold_number=fold_number),
        batch_size=options.batchSize, shuffle=True, num_workers=options.workers)

    # normalize to -1~1
    ngpu = int(options.ngpu)
    nz = int(options.nz)
    nc = int(options.nc)
    nf = int(options.nf)
    image_size = int(options.imageSize)
    batch_size = int(options.batchSize)
    # ==================================================================================================================
    # Models
    # ==================================================================================================================

    # AutoEncoder ======================================================================================================
    net = model.AE(nc, nz, nf, ngpu)
    net.apply(model.weight_init)
    if options.net != '':
        net.load_state_dict(torch.load(options.net))
    print(net)

    # ==================================================================================================================
    # Training
    # ==================================================================================================================

    # criterion set
    criterion = nn.MSELoss()

    # setup optimizer ==================================================================================================
    optimizer = optim.Adam(net.parameters(), betas=(0.5, 0.999), lr=2e-4)

    # container generate
    input_tensor = torch.FloatTensor(batch_size, nc, image_size, image_size)
    mask_tensor = torch.FloatTensor(batch_size, nc, image_size, image_size)

    if options.cuda:
        net.cuda()
        criterion.cuda()
        input_tensor = input_tensor.cuda()
        mask_tensor = mask_tensor.cuda()

    # make to variables
    input_tensor = Variable(input_tensor)
    mask_tensor = Variable(mask_tensor)

    # training start
    print("Training Start!")
    for epoch in range(options.iteration):
        for i, (data, mask_, _) in enumerate(dataloader, 0):
            ############################
            # (1) Update D network
            ###########################
            # train with real data
            optimizer.zero_grad()

            real_cpu = data
            batch_size = real_cpu.size(0)
            input_tensor.data.resize_(real_cpu.size()).copy_(real_cpu)
            mask_tensor.data.resize_(real_cpu.size()).copy_(mask_.float())

            output, z = net(input_tensor)
            output_for_vis = output.data

            if os.path.basename(options.dataroot) == "train_augmented":
                input_tensor = input_tensor * mask_tensor
                output = output * mask_tensor
                loss = criterion(output, input_tensor)

            elif os.path.basename(options.dataroot) == "error_image":
                loss = criterion(output, mask_tensor)

            # todo mask
            loss.backward()

            optimizer.step()

            # visualize
            print('[%d][%d/%d][%d/%d] Loss : %0.5f'
                  % (fold_number,epoch, options.iteration, i, len(dataloader), loss.data[0]))

            #if i == len(dataloader)-1:
                #vutils.save_image(real_cpu, '%s/real_samples_.png' % options.outf, normalize=True)
                #vutils.save_image(output_for_vis, '%s/recon_samples_%d.png' % (options.outf,epoch), normalize=True)
                #vutils.save_image(mask.data, '%s/mask_samples.png' % (options.outf), normalize=True)

            if options.display:
                win_recon_cost = utils.viz_append_line_points(win=win_recon_cost,
                                                             lines_dict=dict(recon=loss.data[0], zero=0),
                                                             x_pos=cnt,
                                                             title="training",
                                                             ylabel='reconstruction cost', xlabel='step')
                cnt = cnt +1
                time.sleep(0.005)  # for reliable drawing

        # checkpoint operation
        if (epoch+1) % options.iteration == 0:

            if os.path.basename(options.dataroot) == "train_augmented":
                torch.save(net.state_dict(),
                           '%s/network_epoch_%d_error-mask_fold%d.pth' % (options.outf, epoch, fold_number))

            elif os.path.basename(options.dataroot) == "error_image":
                torch.save(net.state_dict(),
                           '%s/error-mask-network_epoch_%d_fold_%d.pth' % (options.outf, epoch, fold_number))


# Je Yeol. Lee \[T]/
# Jolly Co-operation
#()()
#('')HAANJU.YOO
