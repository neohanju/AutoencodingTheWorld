import torch
import torch.nn as nn
from torch.autograd import Variable

# weight initialization function from DCGAN example. (resemble with Xavier's)
def weight_init(module):
    # TODO: change to Xavier init : http://wiseodd.github.io/techblog/2017/01/24/vae-pytorch/
    classname = module.__class__.__name__
    if classname.find('Conv') != -1:
        module.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        module.weight.data.normal_(1.0, 0.02)
        module.bias.data.fill_(0)


# =============================================================================
# Autoencoder [original]
# =============================================================================
class AE_LTR(nn.Module):  # autoencoder struction from "Learning temporal regularity in video sequences"
    def __init__(self, num_in_channels, num_filters=512, num_gpu=1):
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
    def __init__(self, num_in_channels, num_filters=512, num_gpu=1):
        super().__init__(num_in_channels, num_filters, num_gpu)
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
    def __init__(self, num_in_channels, z_size=200, num_filters=64, num_gpu=1):
        super().__init__()
        self.num_gpu = num_gpu
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
        self.mu = self.z                                  # Mean μ of Z
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


#()()
#('')HAANJU.YOO
