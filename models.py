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
# Autoencoder [default]
# =============================================================================
class AE(nn.Module):
    def __init__(self, num_in_channels, z_size, num_filters):
        super().__init__()
        self.nc = num_in_channels # number of input channel
        self.nz = z_size  # size of latent variable
        self.nf = num_filters     # number of fundamental filters
        self.encoder = nn.Sequential(
            # expected input: (L) x 227 x 227
            nn.Conv2d(self.nc, self.nf, 5, 2, 1),
            nn.BatchNorm2d(self.nf),
            nn.LeakyReLU(0.2, True),
            # state size: (nf) x 113 x 113
            nn.Conv2d(self.nf, 2 * self.nf, 5, 2, 1),
            nn.BatchNorm2d(2 * self.nf),
            nn.LeakyReLU(0.2, True),
            # state size: (2 x nf) x 56 x 56
            nn.Conv2d(2 * self.nf, 4 * self.nf, 5, 2, 1),
            nn.BatchNorm2d(4 * self.nf),
            nn.LeakyReLU(0.2, True),
            # state size: (4 x nf) x 27 x 27
            nn.Conv2d(4 * self.nf, 8 * self.nf, 5, 2, 1),
            nn.BatchNorm2d(8 * self.nf),
            nn.LeakyReLU(0.2, True),
            # state size: (8 x nf) x 13 x 13
            nn.Conv2d(8 * self.nf, 8 * self.nf, 5, 2, 1),
            nn.BatchNorm2d(8 * self.nf),
            nn.LeakyReLU(0.2, True)
            # state size: (8 x nf) x 6 x 6
        )
        self.z = nn.Conv2d(8 * self.nf, self.nz, 6)
        self.decoder = nn.Sequential(
            # expected input: (nz) x 1 x 1
            nn.ConvTranspose2d(self.nz, 8 * self.nf, 6, 2, 0),
            nn.BatchNorm2d(8 * self.nf),
            nn.LeakyReLU(0.2, True),
            # state size: (8 x nf) x 6 x 6
            nn.ConvTranspose2d(8 * self.nf, 8 * self.nf, 5, 2, 1),
            nn.BatchNorm2d(8 * self.nf),
            nn.LeakyReLU(0.2, True),
            # state size: (8 x nf) x 13 x 13
            nn.ConvTranspose2d(8 * self.nf, 4 * self.nf, 5, 2, 1),
            nn.BatchNorm2d(4 * self.nf),
            nn.LeakyReLU(0.2, True),
            # state size: (4 x nf) x 27 x 27
            nn.ConvTranspose2d(4 * self.nf, 2 * self.nf, 5, 2, 1),
            nn.BatchNorm2d(2 * self.nf),
            nn.LeakyReLU(0.2, True),
            # state size: (2 x nf) x 56 x 56
            nn.ConvTranspose2d(2 * self.nf, self.nf, 5, 2, 1),
            nn.BatchNorm2d(self.nf),
            nn.LeakyReLU(0.2, True),
            # state size: (nf) x 113 x 113
            nn.ConvTranspose2d(self.nf, self.nc, 5, 2, 1),
            nn.Tanh()
            # state size: (L) x 227 x 227
        )

        # init weights
        self.encoder.apply(weight_init)
        self.z.apply(weight_init)
        self.decoder.apply(weight_init)

    def encode(self, x):
        return self.z(self.encoder(x))

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


# =============================================================================
# Variational Autoencoder [default]
# =============================================================================
class VAE(AE):
    def __init__(self, num_in_channels, z_size, num_filters):
        super().__init__(num_in_channels, z_size, num_filters)
        self.mu     = self.z                              # Mean μ of Z
        self.logvar = nn.Conv2d(8 * self.nf, self.nz, 6)  # Log variance σ^2 of Z (diagonal covariance)

        # init weights
        self.mu.apply(weight_init)
        self.logvar.apply(weight_init)

    def encode(self, x):
        encoding_result = self.encoder(x)
        return self.mu(encoding_result), self.logvar(encoding_result)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_() # Compute σ = exp(1/2 log σ^2)
        if self.options.cuda:
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
# =============================================================================


#()()
#('')HAANJU.YOO
