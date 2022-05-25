# This code belongs to the paper
#
# F. Altekrüger, A. Denker, P. Hagemann, J. Hertrich, P. Maass and G. Steidl.
# PatchNR: Learning from Small Data by Patch Normalizing Flow Regularization
# ArXiv Preprint#2205.12021
#
# Please cite the paper, if you use the code.
#
# The script reproduces the numerical example Superresolution with the 
# cPatchNR in the paper.

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import model

from tqdm import tqdm
import utils


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)

def Downsample(input_img, scale = 0.25):
    ''' 
    downsamples an img by factor 4 using gaussian downsample from wgenpatex.py
    '''
    if scale > 1:
        print('Error. Scale factor is larger than 1.')
        return
    gaussian_std = 2
    kernel_size = 16
    gaussian_down = utils.gaussian_downsample(kernel_size,gaussian_std,int(1/scale),pad=False) #gaussian downsample with zero padding
    out = gaussian_down(input_img).to(DEVICE)
    return out


def cPatchNR(img, lam, patch_size, n_patches_out, patchNR, n_iter_max):
    """
    Defines the reconstruction using patchNR as regularizer
    """
                         
    # fixed parameters
    lr_img = img.to(DEVICE)
    operator = Downsample
    center = False 
    init = F.interpolate(img,scale_factor=4,mode='bicubic')

    # create patch extractors
    input_im2pat = utils.patch_extractor(patch_size, pad=False, center=center)

    # intialize optimizer for image
    fake_img = torch.tensor(init.clone(),dtype=torch.float,device=DEVICE,requires_grad = True)
    optim_img = torch.optim.Adam([fake_img], lr=0.03)

    bic_patches = input_im2pat(init)
    # Main loop
    for it in tqdm(range(n_iter_max)):
        optim_img.zero_grad()     
           
        fake_data = input_im2pat(fake_img)
        perm = torch.randperm(len(fake_data))[:n_patches_out]
        fake_data = fake_data[perm]
        lr_data = bic_patches[perm]        
        #patchNR
        pred_inv, log_det_inv = patchNR(fake_data, c = lr_data, rev=True)    
        reg = torch.mean(torch.sum(pred_inv**2,dim=1)/2) - torch.mean(log_det_inv)
       
        #data fidelity
        tmp = torch.nn.functional.pad(fake_img, pad = (6,6,6,6), mode= 'reflect')
        data_fid = torch.sum((operator(tmp) - lr_img)**2)
        
        #loss
        loss =  data_fid + lam*reg
        loss.backward()
        optim_img.step()    
        
    return fake_img


if __name__ == '__main__':
    #input parameters
    patch_size = 6
    num_layers = 5
    subnet_nodes = 512

    net = model.create_cNF(num_layers, subnet_nodes, dimension=patch_size**2,
                      dimension_condition = patch_size**2)
    weights = torch.load('patchNR_weights/weights_cond_material.pth')
    net.load_state_dict(weights['net_state_dict'])
	
    hr = utils.imread('input_imgs/img_test_material.png')
    lr = Downsample(hr)
    lr = lr + 0.01*torch.randn(lr.shape,device=DEVICE)

    lam = 0.15
    n_pat = 128000
    iteration = 500
    rec = cPatchNR(lr,lam = lam, patch_size = patch_size, n_patches_out = n_pat,
                  patchNR = net, n_iter_max = iteration)
    utils.save_img(rec,'results/cPatchNR_material')
    

