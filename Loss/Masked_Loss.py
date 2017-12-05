import torch
import torch.nn as nn
import torch.nn.init
import math

class MarginLoss:
    def __init__(self, cuda=False):
        self.reconstruction_criteria = nn.MSELoss(size_average=False)
        # l1_regularize_criteria = nn.L1Loss(size_average=False)
        # l1_target = Variable([])
        if cuda and torch.cuda.is_available():
            self.reconstruction_criteria.cuda()

    def calculate(self, recon_x, x, margin, options):
        # thanks to Autograd, you can train the net by just summing-up all losses and propagating them
        size_mini_batch = x.data.size()
        num_samples = size_mini_batch[0]
        num_elements = size_mini_batch[1] * size_mini_batch[2] * size_mini_batch[3]


        # MSE with margin
        per_pixel_margin = math.sqrt(margin / num_elements)
        clampled_recon_x = x.sub(recon_x).clamp(-per_pixel_margin, +per_pixel_margin).add(recon_x)
        # mse_per_sample_with_margin = x.sub(clampled_recon_x).pow(2).sum().div(num_samples).cpu().data[0]
        loss = self.reconstruction_criteria(clampled_recon_x, x).div(num_samples)

        return loss