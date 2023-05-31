# This code belongs to the paper
#
# F. Altekrüger, A. Denker, P. Hagemann, J. Hertrich, P. Maass and G. Steidl (2023).
# PatchNR: Learning from Very Few Images by Patch Normalizing Flow Regularization.
# Inverse Problems, vol. 39, no. 6.
#
# Please cite the paper, if you use the code.
# The functions are adapted from 
# 
# J. Hertrich, A. Houdard and C. Redenbach (2022). 
# Wasserstein Patch Prior for Image Superresolution. 
# IEEE Transactions on Computational Imaging.
# (https://github.com/johertrich/Wasserstein_Patch_Prior)
# 
# and 
# 
# A. Houdard, A. Leclaire, N. Papadakis and J. Rabin. 
# Wasserstein Generative Models for Patch-based Texture Synthesis. 
# ArXiv Preprint#2007.03408
# (https://github.com/ahoudard/wgenpatex)

import torch
from torch import nn
import skimage.io as io
import numpy as np
import math

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def imread(img_name):
    """
    loads an image as torch.tensor on the selected device
    """ 
    np_img = io.imread(img_name)
    tens_img = torch.tensor(np_img, dtype=torch.float, device=DEVICE)
    if torch.max(tens_img) > 1:
        tens_img/=255
    if len(tens_img.shape) < 3:
        tens_img = tens_img.unsqueeze(2)
    if tens_img.shape[2] > 3:
        tens_img = tens_img[:,:,:3]
    tens_img = tens_img.permute(2,0,1)
    return tens_img.unsqueeze(0)

def save_img(tensor_img, name):
	'''
	save img (tensor form) with the name
	'''
	img = np.clip(tensor_img.squeeze().detach().cpu().numpy(),0,1)
	io.imsave(str(name)+'.png', img)
	return         

class gaussian_downsample(nn.Module):
    """
    Downsampling module with Gaussian filtering
    """ 
    def __init__(self, kernel_size, sigma, stride, pad=False):
        super(gaussian_downsample, self).__init__()
        self.gauss = nn.Conv2d(1, 1, kernel_size, stride=stride, groups=1, bias=False)     
        gaussian_weights = self.init_weights(kernel_size, sigma)
        self.gauss.weight.data = gaussian_weights.to(DEVICE)
        self.gauss.weight.requires_grad_(False)
        self.pad = pad
        self.padsize = kernel_size-1

    def forward(self, x):
        if self.pad:
            x = torch.cat((x, x[:,:,:self.padsize,:]), 2)
            x = torch.cat((x, x[:,:,:,:self.padsize]), 3)
        return self.gauss(x)

    def init_weights(self, kernel_size, sigma):
        x_cord = torch.arange(kernel_size)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)

        mean = (kernel_size - 1)/2.
        variance = sigma**2.
        gaussian_kernel = (1./(2.*math.pi*variance))*torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1)/(2*variance))
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        return gaussian_kernel.view(1, 1, kernel_size, kernel_size)

class patch_extractor(nn.Module):
    """
    Module for creating custom patch extractor
    """ 
    def __init__(self, patch_size, pad=False,center=False):
        super(patch_extractor, self).__init__()
        self.im2pat = nn.Unfold(kernel_size=patch_size)
        self.pad = pad
        self.padsize = patch_size-1
        self.center=center
        self.patch_size=patch_size

    def forward(self, input, batch_size=0):
        if self.pad:
            input = torch.cat((input, input[:,:,:self.padsize,:]), 2)
            input = torch.cat((input, input[:,:,:,:self.padsize]), 3)
        patches = self.im2pat(input).squeeze(0).transpose(1,0)

        if batch_size > 0:
            idx = torch.randperm(patches.size(0))[:batch_size]
            patches = patches[idx,:]
        if self.center:
            patches = patches - torch.mean(patches,-1).unsqueeze(-1)
        return patches
