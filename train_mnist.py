from __future__ import print_function
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import math
import numpy as np
import matplotlib.pyplot as plt
import operator
import utils as util
import os


parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--vae', action='store_true', default=False, help='add variational loss')
parser.add_argument('--perturb', action='store_true', default=False, help='perturbation')
parser.add_argument('--save', type=int, default=50, help='saving interval')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./MNIST_DATA', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./MNIST_DATA', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=False, **kwargs)


def imshow(img, title=None):
    npimg = img.cpu().numpy().reshape((28, 28))
    plt.imshow(npimg)
    plt.show()
    if title is not None:
        plt.suptitle(title)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 20)
        self.fc31 = nn.Linear(20, 2)
        self.fc32 = nn.Linear(20, 2)
        self.fc4 = nn.Linear(2, 20)
        self.fc5 = nn.Linear(20, 400)
        self.fc6 = nn.Linear(400, 784)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h2 = self.relu(self.fc1(x))
        h3 = self.relu(self.fc2(h2))
        return self.fc31(h3), self.fc32(h3)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if args.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h4 = self.relu(self.fc4(z))
        h5 = self.relu(self.fc5(h4))
        return self.sigmoid(self.fc6(h5))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE()
if args.cuda:
    model.cuda()

reconstruction_function = nn.MSELoss()
reconstruction_function.size_average = False


def loss_function(recon_x, x, mu, logvar, margin):
    # thanks to Autograd, you can train the net by just summing-up all losses and propagating them
    size_mini_batch = x.data.size()
    num_samples = size_mini_batch[0]
    num_elements = size_mini_batch[1] * size_mini_batch[2] * size_mini_batch[3]

    # MSE
    mse_batch = x.sub(recon_x).pow(2)
    mse_of_sample = []
    for i in range(num_samples):
        cur_mse = mse_batch[i, :, :, :].sum().data[0]
        mse_of_sample += [cur_mse]

    # MSE with margin
    per_pixel_margin = math.sqrt(margin / num_elements)
    clamped_recon_x = x.sub(recon_x).clamp(-per_pixel_margin, +per_pixel_margin).add(recon_x)
    MSE_margin = reconstruction_function(clamped_recon_x, x).div(num_samples)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5).div_(num_samples)

    loss_info = dict(KLD=KLD.data[0], MSE_margin=MSE_margin.data[0])

    if args.vae:
        total_loss = MSE_margin + KLD
    else:
        total_loss = MSE_margin
    return total_loss, loss_info, mse_of_sample


optimizer = optim.Adam(model.parameters(), lr=1e-3)
MSE_stable_upper_bound = 30


def train(epoch, margin, do_perturb):
    model.train()
    train_loss = 0
    MSE = []
    MSE_max = []
    recon_best = None
    recon_worst = None
    min_MSE = 10000
    max_MSE = -1
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss, loss_info, mse_of_sample = loss_function(recon_batch, data, mu, logvar, margin)
        if do_perturb:
            perturb_power = max(mse_of_sample)
            h = mu.register_hook(lambda grad: grad * perturb_power)
            loss.backward()
            h.remove()
        else:
            loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))

        MSE += mse_of_sample

        min_index, min_value = min(enumerate(mse_of_sample), key=operator.itemgetter(1))
        if min_MSE > min_value:
            min_MSE = min_value
            recon_best = recon_batch[min_index, :].data

        max_index, max_value = max(enumerate(mse_of_sample), key=operator.itemgetter(1))
        if max_MSE < max_value:
            max_MSE = max_value
            recon_worst = recon_batch[max_index, :].data

    MSE_np = np.array(MSE)
    MSE_mean = np.mean(MSE_np)
    MSE_std = np.std(MSE_np)
    MSE_max = max(MSE)

    # show result
    # fig, axarr = plt.subplots(1, 2)
    # fig.suptitle("Reconstructions", fontsize=16)
    # axarr[0].set_title('Worst')
    # axarr[0].imshow(recon_worst.cpu().numpy().reshape((28, 28)))
    # axarr[1].set_title('Best')
    # axarr[1].imshow(recon_best.cpu().numpy().reshape((28, 28)))
    # plt.draw()

    # decide when it is need to be perturb
    is_need_perturb = False
    num_outliers = 0
    for mse in MSE:
        if mse > MSE_mean + 2.5 * MSE_std:
            num_outliers += 1
    is_stable = False
    outlier_ratio = num_outliers / len(MSE)
    if outlier_ratio < 0.1:
        is_stable = True
    if MSE_stable_upper_bound > MSE_mean and is_stable and args.perturb:
        is_need_perturb = True

    print('====> Epoch: {} Average loss: {:.4f} MSE: max={:.4f}, mean={:.4f}, std={:.4f}, Outlier rat.: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset), MSE_max, MSE_mean, MSE_std, outlier_ratio))

    return is_need_perturb

str_perturb = '_perturb' if args.perturb else ''
str_variational = '_vae' if args.vae else ''
file_base_name = 'latent%s%s' % (str_perturb, str_variational)
filename_path = './data/mnist/'


def test(epoch, save_path):
    model.eval()
    test_loss = 0

    for data, labels in test_loader:
        if args.cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        recon_batch, mu, logvar = model(data)
        loss, loss_info, mse_of_sample = loss_function(recon_batch, data, mu, logvar, 0)
        test_loss += loss.data[0]

        # file print
        for i in range(mu.data.size()[0]):
            output_info = np.append(mu[i].data.cpu().numpy().flatten(), mse_of_sample[i])
            output_info = np.append(output_info, labels[i])
            util.file_print_list(save_path, output_info, overwrite=False)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


util.make_dir('./data/mnist')

for epoch in range(1, args.epochs + 1):
    train(epoch, 0, args.perturb)

    if epoch % args.save == 0:
        result_path = os.path.join(filename_path, '%s_%04d.txt' % (file_base_name, epoch))
        util.file_print_list(result_path, [], overwrite=True)
        test(epoch, result_path)

plt.show()