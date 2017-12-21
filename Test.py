import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.init
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.optim as optim
import cv2
# import custom package
import numpy as np
import Datasets.RGBImageSet_augmented as dset
import Models.AutoEncoder as model



#=======================================================================================================================
# Options
#=======================================================================================================================
parser = argparse.ArgumentParser()
# Options for path =====================================================================================================
parser.add_argument('--dataset', default='MNIST', help='what is dataset?')
parser.add_argument('--dataroot', default='/media/leejeyeol/74B8D3C8B8D38750/Data/CVC-ClinicDB/train_augmented', help='path to dataset')
parser.add_argument('--net', default='/home/leejeyeol/Git/AutoencodingTheWorld/output/network_epoch_900_fold_0.pth', help="path of networks.(to continue training)")
parser.add_argument('--outf', default='./Evaluation', help="folder to output images and model checkpoints")

parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--display', default=False, help='display options. default:False. NOT IMPLEMENTED')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--workers', type=int, default=1, help='number of data loading workers')
parser.add_argument('--iteration', type=int, default=1, help='number of epochs to train for')

# these options are saved for testing
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--imageSize', type=int, default=224, help='the height / width of the input image to network')
parser.add_argument('--model', type=str, default='autoencoder', help='Model name')
parser.add_argument('--nc', type=int, default=3, help='number of input channel.')
parser.add_argument('--nz', type=int, default=400, help='latent size.')
parser.add_argument('--nf', type=int, default=64, help='number of filter.(first layer)')

parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam.')


parser.add_argument('--seed', type=int, help='manual seed')

options = parser.parse_args()
print(options)



# save directory make   ================================================================================================
try:
    os.makedirs(options.outf)
except OSError:
    pass

# seed set  ============================================================================================================
if options.seed is None:
    options.seed = random.randint(1, 10000)
print("Random Seed: ", options.seed)
random.seed(options.seed)
torch.manual_seed(options.seed)

# cuda set  ============================================================================================================
if options.cuda:
    torch.cuda.manual_seed(options.seed)

torch.backends.cudnn.benchmark = True
cudnn.benchmark = True
if torch.cuda.is_available() and not options.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


error_save_path = '/media/leejeyeol/74B8D3C8B8D38750/Data/CVC-ClinicDB/error_image'
if not os.path.exists(error_save_path):
    os.makedirs(error_save_path)
    print(error_save_path + " : the save directory is maked.")



#=======================================================================================================================
# Data and Parameters
#=======================================================================================================================

# MNIST call and load   ================================================================================================

# todo fold number
for fold_number in range(10):


    dataloader = torch.utils.data.DataLoader(dset.RGBImageSet_augmented(options.dataroot, type='test', centered=False, fold_number=fold_number),batch_size=options.batchSize, shuffle=False, num_workers=options.workers)

    # normalize to -1~1
    ngpu = int(options.ngpu)
    nz = int(options.nz)
    nc = int(options.nc)
    nf = int(options.nf)
    image_size = int(options.imageSize)
    batch_size = int(options.batchSize)
    #=======================================================================================================================
    # Models
    #=======================================================================================================================

    if os.path.basename(options.dataroot) == "train_augmented":
        options.net ='/home/leejeyeol/Git/AutoencodingTheWorld/output/network_epoch_19_error-mask_fold%d.pth'%fold_number
    elif os.path.basename(options.dataroot) == "error_image":
        options.net = '/home/leejeyeol/Git/AutoencodingTheWorld/output/error-mask-network_epoch_19_fold_%d.pth' % fold_number

    # AutoEncoder ============================================================================================================
    net = model.AE(nc,nz,nf)
    net.apply(model.weight_init)
    if options.net != '':
        net.load_state_dict(torch.load(options.net))
    print(net)

    #=======================================================================================================================
    # Training
    #=======================================================================================================================

    # criterion set
    criterion = nn.MSELoss()

    # setup optimizer   ====================================================================================================

    optimizer = optim.Adam(net.parameters(), betas=(0.5, 0.999), lr=2e-4)



    # container generate
    input = torch.FloatTensor(batch_size, nc, image_size, image_size)
    mask = torch.FloatTensor(batch_size, nc, image_size, image_size)

    if options.cuda:
        net.cuda()
        criterion.cuda()
        input = input.cuda()
        mask = mask.cuda()


    # make to variables ====================================================================================================
    input = Variable(input, volatile=True)
    mask = Variable(mask, volatile=True)
    total_loss = []

    # training start
    print("Training Start!")
    for epoch in range(options.iteration):
        for i, (data, mask_, data_name) in enumerate(dataloader, 0):
            ############################
            # (1) Update D network
            ###########################
            # train with real data  ========================================================================================
            optimizer.zero_grad()



            real_cpu = data
            batch_size = real_cpu.size(0)
            input.data.resize_(real_cpu.size()).copy_(real_cpu)
            mask.data.resize_(real_cpu.size()).copy_(mask_.float())

            output, _ = net(input)
            output_for_vis = output.data
            if os.path.basename(options.dataroot) == "train_augmented":

                error_image = input - output
                error_image = error_image.data.cpu().numpy()
                error_image = np.reshape(error_image, [error_image.shape[1], error_image.shape[2], error_image.shape[3]])
                input = input * mask
                output = output * mask
                masked_error_image = input - output
                loss = criterion(output, input)

            elif os.path.basename(options.dataroot) == "error_image":
                loss = criterion(output, mask)

            #total_loss.append(loss)

            print('[%d/%d][%d/%d] Loss : %0.5f'
                  % (fold_number, 10, i, len(dataloader), loss.data[0]))


            #vutils.save_image(real_cpu, '%s/%04d_real_samples.png' % (options.outf, i))
            #vutils.save_image(output_for_vis, '%s/%05d_recon_samples.png' % (options.outf, i))
            #vutils.save_image(mask.data, '%s/%05d_mask_samples.png' % (options.outf, i))
            if os.path.basename(options.dataroot) == "train_augmented":
                np.save(os.path.join(error_save_path, data_name[0]), error_image)

            #cv2.imwrite('%s/%04d_error_image.png' % (options.outf, i),error_image)
            #vutils.save_image(error_image, '%s/%04d_error_image.png' % (options.outf, i))
            #vutils.save_image(masked_error_image.data, '%s/%04d_masked_error_image.png' % (options.outf, i))


        np.save('%s/total_loss' % (options.outf), total_loss)



# Je Yeol. Lee \[T]/
# Jolly Co-operation