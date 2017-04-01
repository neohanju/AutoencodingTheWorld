import torch
import torch.nn as nn
from torch.autograd import Variable


# =============================================================================
# Autoencoder [default]
# =============================================================================
class AE(nn.Module):
    def __init__(self, num_input_channels, options):
        super().__init__()
        self.options = options
        self.encoder = nn.Sequential(
            # expected input: (L) x 227 x 227
            nn.Conv2d(num_input_channels, options.nf, 5, 2, 1),
            nn.BatchNorm2d(options.nf),
            nn.LeakyReLU(0.2, True),
            # state size: (nf) x 113 x 113
            nn.Conv2d(options.nf, 2 * options.nf, 5, 2, 1),
            nn.BatchNorm2d(2 * options.nf),
            nn.LeakyReLU(0.2, True),
            # state size: (2 x nf) x 56 x 56
            nn.Conv2d(2 * options.nf, 4 * options.nf, 5, 2, 1),
            nn.BatchNorm2d(4 * options.nf),
            nn.LeakyReLU(0.2, True),
            # state size: (4 x nf) x 27 x 27
            nn.Conv2d(4 * options.nf, 8 * options.nf, 5, 2, 1),
            nn.BatchNorm2d(8 * options.nf),
            nn.LeakyReLU(0.2, True),
            # state size: (8 x nf) x 13 x 13
            nn.Conv2d(8 * options.nf, 8 * options.nf, 5, 2, 1),
            nn.BatchNorm2d(8 * options.nf),
            nn.LeakyReLU(0.2, True)
            # state size: (8 x nf) x 6 x 6
        )
        self.z = nn.Conv2d(8 * options.nf, options.nz, 6)
        self.decoder = nn.Sequential(
            # expected input: (nz) x 1 x 1
            nn.ConvTranspose2d(options.nz, 8 * options.nf, 6, 2, 0),
            nn.BatchNorm2d(8 * options.nf),
            nn.LeakyReLU(0.2, True),
            # state size: (8 x nf) x 6 x 6
            nn.ConvTranspose2d(8 * options.nf, 8 * options.nf, 5, 2, 1),
            nn.BatchNorm2d(8 * options.nf),
            nn.LeakyReLU(0.2, True),
            # state size: (8 x nf) x 13 x 13
            nn.ConvTranspose2d(8 * options.nf, 4 * options.nf, 5, 2, 1),
            nn.BatchNorm2d(4 * options.nf),
            nn.LeakyReLU(0.2, True),
            # state size: (4 x nf) x 27 x 27
            nn.ConvTranspose2d(4 * options.nf, 2 * options.nf, 5, 2, 1),
            nn.BatchNorm2d(2 * options.nf),
            nn.LeakyReLU(0.2, True),
            # state size: (2 x nf) x 56 x 56
            nn.ConvTranspose2d(2 * options.nf, options.nf, 5, 2, 1),
            nn.BatchNorm2d(options.nf),
            nn.LeakyReLU(0.2, True),
            # state size: (nf) x 113 x 113
            nn.ConvTranspose2d(options.nf, num_input_channels, 5, 2, 1),
            nn.Tanh()
            # state size: (L) x 227 x 227
        )

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
    def __init__(self, num_input_channels, options):
        super().__init__()
        self.mu     = nn.Conv2d(8 * options.nf, options.nz, 6) # Mean μ of Z
        self.logvar = nn.Conv2d(8 * options.nf, options.nz, 6) # Log variance σ^2 of Z (diagonal covariance)

    def encode(self, x):
        encoding_result = self.encoder(x)
        return self.mean(encoding_result), self.log_var(encoding_result)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_() # Compute σ = exp(1/2 log σ^2)
        if self.options.cpu_only:
            eps = torch.FloatTensor(std.size()).normal_()
        else:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
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
