import torch
import torch.nn as nn
import torch.nn.init
from torch.autograd import Variable


# xavier_init
def weight_init(module):
    classname = module.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.xavier_normal(module.weight.data)
        # module.weight.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        module.weight.data.normal_(1.0, 0.02)
        module.bias.data.fill_(0)


def init_model_and_loss(options, cuda=False):

    # create model instance
    if 'AE-LTR' == options.model:
        model = AE_LTR(options.nc)
    elif 'VAE-LTR' == options.model:
        model = VAE_LTR(options.nc)
    elif 'AE' == options.model:
        model = AE(options.nc, options.nz, options.nf)
    elif 'VAE' == options.model:
        model = VAE(options.nc, options.nz, options.nf)
    elif 'VAE-NARROW' == options.model:
        model = VAE_NARROW(options.nc, options.nz)
    assert model

    if options.model_path != '':
        model.load_state_dict(torch.load(options.model_path))
        print(options.model + ' is loaded')
    else:
        print(options.model + ' is generated')

    # model to CUDA
    if cuda and torch.cuda.is_available():
        model.cuda()

    # loss
    loss = OurLoss(cuda)
    return model, loss


    # =============================================================================
# Loss function
# =============================================================================
class EBCAELoss:
    def __init__(self, cuda=False):
        self.reconstruction_criteria = nn.MSELoss(size_average=False)
        if cuda and torch.cuda.is_available():
            self.reconstruction_criteria.cuda()

    def calculate(self, recon_x, x, options, mu=None, logvar=None):
        # thanks to Autograd, you can train the net by just summing-up all losses and propagating them
        size_mini_batch = x.data.size()[0]
        recon_loss = self.reconstruction_criteria(recon_x, x).div_(size_mini_batch)
        total_loss = recon_loss
        loss_info = {'recon': recon_loss.data[0]}

        if options.variational:
            assert mu is not None and logvar is not None
            # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            kld_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
            kld_loss = torch.sum(kld_element).mul_(-0.5)
            kld_loss_final = kld_loss.div_(size_mini_batch).mul_(options.var_loss_coef)
            loss_info['variational'] = kld_loss_final.data[0]
            total_loss += kld_loss_final

        loss_info['total'] = total_loss.data[0]

        return total_loss, loss_info


class OurLoss:
    def __init__(self, cuda=False):
        self.reconstruction_criteria = nn.MSELoss(size_average=False)
        self.GAN_criteria = nn.BCELoss(size_average=False)
        # l1_regularize_criteria = nn.L1Loss(size_average=False)
        # l1_target = Variable([])
        if cuda and torch.cuda.is_available():
            self.reconstruction_criteria.cuda()
            self.GAN_criteria.cuda()

    def calculate(self, recon_x, x, options, mu=None, logvar=None):
        # thanks to Autograd, you can train the net by just summing-up all losses and propagating them
        size_mini_batch = x.data.size()[0]
        recon_loss = self.reconstruction_criteria(recon_x, x).div_(size_mini_batch)
        total_loss = recon_loss
        loss_info = {'recon': recon_loss.data[0]}

        if options.variational:
            assert mu is not None and logvar is not None
            # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            kld_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
            kld_loss = torch.sum(kld_element).mul_(-0.5)
            kld_loss_final = kld_loss.div_(size_mini_batch).mul_(options.var_loss_coef)
            loss_info['variational'] = kld_loss_final.data[0]
            total_loss += kld_loss_final

        # if 0.0 != options.l1_coef:
        #     l1_loss = options.l1_coef * l1_regularize_criteria(model.parameters(), l1_target)
        #     loss_info['l1_reg'] = l1_loss.data[0]
        #     # params.data -= options.learning_rate * params.grad.data
        #     total_loss += l1_loss

        loss_info['total'] = total_loss.data[0]

        return total_loss, loss_info

    def calculate_GAN(self, output, label, loss_per_batch=False):
        batch_loss = []
        loss = 0
        size_mini_batch = output.data.size()[0]
        if loss_per_batch:
            for i in range(0, size_mini_batch):
                batch_loss.append(self.GAN_criteria(output[i], label[i]))
                loss = batch_loss[i]+loss

            loss = loss.div_(size_mini_batch)
            return loss, batch_loss
        else:
            loss = self.GAN_criteria(output, label).div_(size_mini_batch)
            return loss

        '''
        if options.variational:
            assert mu is not None and logvar is not None
            # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            kld_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
            kld_loss = torch.sum(kld_element).mul_(-0.5)
            kld_loss_final = kld_loss.div_(size_mini_batch).mul_(options.var_loss_coef)
            loss_info['variational'] = kld_loss_final.data[0]
            total_loss += kld_loss_final
        '''
        # if 0.0 != options.l1_coef:
        #     l1_loss = options.l1_coef * l1_regularize_criteria(model.parameters(), l1_target)
        #     loss_info['l1_reg'] = l1_loss.data[0]
        #     # params.data -= options.learning_rate * params.grad.data
        #     total_loss += l1_loss


# =============================================================================
# Autoencoder [original]
# =============================================================================
class AE_LTR(nn.Module):  # autoencoder struction from "Learning temporal regularity in video sequences"
    def __init__(self, num_in_channels, num_filters=512):

        super().__init__()

        # encoder layers
        self.conv1 = nn.Conv2d(num_in_channels, num_filters, 11, 4)
        self.encode_act1 = nn.Tanh()
        # (batch_size) x 512 x 55 x 55
        self.pool1 = nn.MaxPool2d(2, stride=2, return_indices=True)
        # (batch_size) x 512 x 27 x 27
        self.conv2 = nn.Conv2d(num_filters, int(num_filters / 2), 5, 1, 2)
        self.encode_act2 = nn.Tanh()
        # (batch_size) x 256 x 27 x 27
        self.pool2 = nn.MaxPool2d(2, stride=2, return_indices=True)
        # (batch_size) x 256 x 13 x 13
        self.conv3 = nn.Conv2d(int(num_filters / 2), int(num_filters / 4), 3, 1, 1)
        self.encode_act3 = nn.Tanh()
        # (batch_size) x 128 x 13 x 13

        # decoder layers
        self.deconv1 = nn.ConvTranspose2d(int(num_filters / 4), int(num_filters / 2), 3, 1, 1)
        self.decode_act1 = nn.Tanh()
        # (batch_size) x 256 x 13 x 13
        self.unpool1 = nn.MaxUnpool2d(2, stride=2)
        # (batch_size) x 256 x 27 x 27
        self.deconv2 = nn.ConvTranspose2d(int(num_filters / 2), num_filters, 5, 1, 2)
        self.decode_act2 = nn.Tanh()
        # (batch_size) x 512 x 27 x 27
        self.unpool2 = nn.MaxUnpool2d(2, stride=2)
        # (batch_size) x 512 x 55 x 55
        self.deconv3 = nn.ConvTranspose2d(num_filters, num_in_channels, 11, 4)
        self.decode_act3 = nn.Tanh()
        # (batch_size) x 10 x 227 x 227

        self.weight_init()

    def encode(self, x):
        encode1 = self.encode_act1(self.conv1(x))
        encode2, index1 = self.pool1(encode1)
        encode3, index2 = self.pool2(self.encode_act2(self.conv2(encode2)))
        code = self.encode_act3(self.conv3(encode3))
        size1, size2 = encode1.size()[3], encode2.size()[3]
        return code, index1, size1, index2, size2

    def decode(self, code, index1, size1, index2, size2):
        decode1 = self.unpool1(self.decode_act1(self.deconv1(code)), index2,
                               output_size=torch.Size([-1, -1, size2, size2]))
        decode2 = self.unpool2(self.decode_act2(self.deconv2(decode1)), index1,
                               output_size=torch.Size([-1, -1, size1, size1]))
        return self.decode_act3(self.deconv3(decode2))

    def forward(self, x):
        code, index1, size1, index2, size2 = self.encode(x)
        return self.decode(code, index1, size1, index2, size2), code, None

    def weight_init(self):
        self.conv1.apply(weight_init)
        self.encode_act1.apply(weight_init)
        self.pool1.apply(weight_init)
        self.conv2.apply(weight_init)
        self.encode_act2.apply(weight_init)
        self.pool2.apply(weight_init)
        self.conv3.apply(weight_init)
        self.encode_act3.apply(weight_init)

        self.deconv1.apply(weight_init)
        self.decode_act1.apply(weight_init)
        self.unpool1.apply(weight_init)
        self.deconv2.apply(weight_init)
        self.decode_act2.apply(weight_init)
        self.unpool2.apply(weight_init)
        self.deconv3.apply(weight_init)
        self.decode_act3.apply(weight_init)


# =============================================================================
# Variational Autoencoder [original]
# =============================================================================
class VAE_LTR(AE_LTR):  # autoencoder struction from "Learning temporal regularity in video sequences"
    def __init__(self, num_in_channels, num_filters=512):

        super().__init__(num_in_channels, num_filters)

        # (batch_size) x 128 x 13 x 13
        self.mu = self.conv3
        self.mu_act = self.encode_act3
        self.logvar = nn.Conv2d(int(num_filters / 2), int(num_filters / 4), 3, 1, 1)
        self.logvar_act = nn.Tanh()

        # init weights
        self.mu.apply(weight_init)
        self.mu_act.apply(weight_init)
        self.logvar.apply(weight_init)
        self.logvar_act.apply(weight_init)

    def encode(self, x):
        encode1 = self.encode_act1(self.conv1(x))
        encode2, index1 = self.pool1(encode1)
        encode3, index2 = self.pool2(self.encode_act2(self.conv2(encode2)))
        mu = self.mu_act(self.mu(encode3))
        logvar = self.logvar_act(self.logvar(encode3))
        size1, size2 = encode1.size()[3], encode2.size()[3]
        return mu, logvar, index1, size1, index2, size2

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()  # Compute σ = exp(1/2 log σ^2)
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar, index1, size1, index2, size2 = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z, index1, size1, index2, size2), mu, logvar


# =============================================================================
# Autoencoder [new]
# =============================================================================
class AE(nn.Module):
    def __init__(self, num_in_channels, z_size=200, num_filters=64):
        super().__init__()
        self.encoder = nn.Sequential(
            # expected input: (L) x 227 x 227
            nn.Conv2d(num_in_channels, num_filters, 5, 2, 1),
            nn.BatchNorm2d(num_filters),
            nn.LeakyReLU(0.2, True),
            # state size: (nf) x 113 x 113
            nn.Conv2d(num_filters, 2 * num_filters, 5, 2, 1),
            nn.BatchNorm2d(2 * num_filters),
            nn.LeakyReLU(0.2, True),
            # state size: (2 x nf) x 56 x 56
            nn.Conv2d(2 * num_filters, 4 * num_filters, 5, 2, 1),
            nn.BatchNorm2d(4 * num_filters),
            nn.LeakyReLU(0.2, True),
            # state size: (4 x nf) x 27 x 27
            nn.Conv2d(4 * num_filters, 8 * num_filters, 5, 2, 1),
            nn.BatchNorm2d(8 * num_filters),
            nn.LeakyReLU(0.2, True),
            # state size: (8 x nf) x 13 x 13
            nn.Conv2d(8 * num_filters, 8 * num_filters, 5, 2, 1),
            nn.BatchNorm2d(8 * num_filters),
            nn.LeakyReLU(0.2, True)
            # state size: (8 x nf) x 6 x 6
        )
        self.z = nn.Conv2d(8 * num_filters, z_size, 6)
        self.decoder = nn.Sequential(
            # expected input: (nz) x 1 x 1
            nn.ConvTranspose2d(z_size, 8 * num_filters, 6, 2, 0),
            nn.BatchNorm2d(8 * num_filters),
            nn.LeakyReLU(0.2, True),
            # state size: (8 x nf) x 6 x 6
            nn.ConvTranspose2d(8 * num_filters, 8 * num_filters, 5, 2, 1),
            nn.BatchNorm2d(8 * num_filters),
            nn.LeakyReLU(0.2, True),
            # state size: (8 x nf) x 13 x 13
            nn.ConvTranspose2d(8 * num_filters, 4 * num_filters, 5, 2, 1),
            nn.BatchNorm2d(4 * num_filters),
            nn.LeakyReLU(0.2, True),
            # state size: (4 x nf) x 27 x 27
            nn.ConvTranspose2d(4 * num_filters, 2 * num_filters, 5, 2, 1),
            nn.BatchNorm2d(2 * num_filters),
            nn.LeakyReLU(0.2, True),
            # state size: (2 x nf) x 55 x 55
            nn.ConvTranspose2d(2 * num_filters, num_filters, 5, 2),
            nn.BatchNorm2d(num_filters),
            nn.LeakyReLU(0.2, True),
            # state size: (nf) x 113 x 113
            nn.ConvTranspose2d(num_filters, num_in_channels, 5, 2, 1),
            nn.Tanh()
            # state size: (L) x 227 x 227
        )

        # init weights
        self.weight_init()

    def encode(self, x):
        return self.z(self.encoder(x))

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z, None

    def weight_init(self):
        self.encoder.apply(weight_init)
        self.z.apply(weight_init)
        self.decoder.apply(weight_init)


# =============================================================================
# Variational Autoencoder [default]
# =============================================================================
class VAE(AE):
    def __init__(self, num_in_channels, z_size, num_filters):

        super().__init__(num_in_channels, z_size, num_filters)

        self.mu = self.z                                     # Mean μ of Z
        self.logvar = nn.Conv2d(8 * num_filters, z_size, 6)  # Log variance σ^2 of Z (diagonal covariance)

        # init weights
        self.mu.apply(weight_init)
        self.logvar.apply(weight_init)

    def encode(self, x):
        encoding_result = self.encoder(x)
        return self.mu(encoding_result), self.logvar(encoding_result)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()  # Compute σ = exp(1/2 log σ^2)
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar


# =============================================================================
# Variational Autoencoder [default]
#    Follow the LTR's philosophy, but more recent architecture
# =============================================================================
class VAE_NARROW(nn.Module):
    def __init__(self, num_in_channels, z_size):
        super().__init__()
        self.encoder = nn.Sequential(
            # expected input: (L) x 227 x 227
            nn.Conv2d(num_in_channels, 512, 11, 4, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # state size: (512) x 55 x 55
            nn.Conv2d(512, 256, 7, 2, 3, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            # state size: (256) x 28 x 28
            nn.Conv2d(256, 128, 5, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            # state size: (128) x 13 x 13
        )
        self.mu = nn.Conv2d(128, z_size, 13)
        self.logvar = nn.Conv2d(128, z_size, 13)
        # state size: (z_size) x 1 x 1
        self.decoder = nn.Sequential(
            # expected input: (nz) x 1 x 1
            nn.ConvTranspose2d(z_size, 128, 13),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            # state size: (128) x 13 x 13
            nn.ConvTranspose2d(128, 256, 5, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            # state size: (256) x 27 x 27
            nn.ConvTranspose2d(256, 512, 7, 2, 2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # state size: (512) x 55 x 55
            nn.ConvTranspose2d(512, num_in_channels, 11, 4),
            nn.Tanh()
            # state size: (L) x 227 x 227
        )

        # init weights
        self.weight_init()

    def encode(self, x):
        encoding_result = self.encoder(x)
        return self.mu(encoding_result), self.logvar(encoding_result)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()  # Compute σ = exp(1/2 log σ^2)
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        return self.decoder(z)

    def weight_init(self):
        self.encoder.apply(weight_init)
        self.mu.apply(weight_init)
        self.logvar.apply(weight_init)
        self.decoder.apply(weight_init)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar


# =============================================================================
# Generator in DC-GAN
# =============================================================================
class DCGAN_Generator(nn.Module):
    def __init__(self, num_target_channels, z_size, num_filters):
        super().__init__()
        self.num_target_channels = num_target_channels
        self.z_size = z_size
        self.num_filters = num_filters
        self.main = nn.Sequential(
            # expected input: (nz) x 1 x 1
            nn.ConvTranspose2d(z_size, 8 * num_filters, 6, 2, 0, bias=False),
            nn.BatchNorm2d(8 * num_filters),
            nn.ReLU(True),
            # state size: (8 x nf) x 6 x 6
            nn.ConvTranspose2d(8 * num_filters, 8 * num_filters, 5, 2, 1, bias=False),
            nn.BatchNorm2d(8 * num_filters),
            nn.ReLU(True),
            # state size: (8 x nf) x 13 x 13
            nn.ConvTranspose2d(8 * num_filters, 4 * num_filters, 5, 2, 1, bias=False),
            nn.BatchNorm2d(4 * num_filters),
            nn.ReLU(True),
            # state size: (4 x nf) x 27 x 27
            nn.ConvTranspose2d(4 * num_filters, 2 * num_filters, 5, 2, 1, bias=False),
            nn.BatchNorm2d(2 * num_filters),
            nn.ReLU(True),
            # state size: (2 x nf) x 55 x 55
            nn.ConvTranspose2d(2 * num_filters, num_filters, 5, 2, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(True),
            # state size: (nf) x 113 x 113
            nn.ConvTranspose2d(num_filters, num_target_channels, 5, 2, 1),
            nn.Tanh()
            # state size: (c) x 227 x 227
        )

        # init weights
        self.main.apply(weight_init)

    def forward(self, x, gpu_ids=None):
        if gpu_ids is not None:
            return nn.parallel.data_parallel(self.main, x, gpu_ids)
        else:
            return self.main(x)


# =============================================================================
# Discriminator in DC-GAN
# =============================================================================
class DCGAN_Discriminator(nn.Module):
    def __init__(self, num_input_channels, num_filters):
        super().__init__()
        self.num_input_channels = num_input_channels
        self.num_filters = num_filters
        self.main = nn.Sequential(
            # expected input: (L) x 227 x 227
            nn.Conv2d(num_input_channels, num_filters, 5, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),
            # state size: (nf) x 113 x 113
            nn.Conv2d(num_filters, 2 * num_filters, 5, 2, 1, bias=False),
            nn.BatchNorm2d(2 * num_filters),
            nn.LeakyReLU(0.2, True),
            # state size: (2 x nf) x 56 x 56
            nn.Conv2d(2 * num_filters, 4 * num_filters, 5, 2, 1, bias=False),
            nn.BatchNorm2d(4 * num_filters),
            nn.LeakyReLU(0.2, True),
            # state size: (4 x nf) x 27 x 27
            nn.Conv2d(4 * num_filters, 8 * num_filters, 5, 2, 1, bias=False),
            nn.BatchNorm2d(8 * num_filters),
            nn.LeakyReLU(0.2, True),
            # state size: (8 x nf) x 13 x 13
            nn.Conv2d(8 * num_filters, 8 * num_filters, 5, 2, 1, bias=False),
            nn.BatchNorm2d(8 * num_filters),
            nn.LeakyReLU(0.2, True),
            # state size: (8 x nf) x 6 x 6
            nn.Conv2d(8 * num_filters, 1, 6, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, gpu_ids=None):
        if gpu_ids is not None:
            output = nn.parallel.data_parallel(self.main, x, gpu_ids)
        else:
            output = self.main(x)
        return output.view(-1, 1)

#()()
#('')HAANJU.YOO
