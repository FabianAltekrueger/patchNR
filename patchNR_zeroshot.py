# This code belongs to the paper
#
# F. Altekrüger, A. Denker, P. Hagemann, J. Hertrich, P. Maass and G. Steidl.
# PatchNR: Learning from Small Data by Patch Normalizing Flow Regularization
# ArXiv Preprint#2205.12021
#
# Please cite the paper, if you use the code.
#
# The script reproduces the zero-shot superresolution example in the paper.

import torch 
from torch import nn
import numpy as np 
import random
from model import create_NF
from tqdm import tqdm
from utils import *

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# downsample operator
def Downsample(input_img, scale = 0.5):
    gaussian_std = 1.
    kernel_size = 16
    gaussian_down = gaussian_downsample(kernel_size,gaussian_std,int(1/scale),pad=False)
    out = gaussian_down(input_img)
    return out

def train_patchNR(patchNR, img, patch_size, steps, batch_size, center):
    """
    Train the patchNR for the given img (low resolution)
    """
    batch_size = batch_size
    optimizer_steps = steps
    center = center 
    optimizer = torch.optim.Adam(patchNR.parameters(), lr = 1e-4)
    im2patch = patch_extractor(patch_size=patch_size, center = center)
    patches = torch.empty(0,device=DEVICE)
    #enlarge training patches by rotation and mirroring
    for j in range(2): 
        if j == 0:
            tmp = img
        elif j == 1:
            tmp = torch.flip(img,[1])
        for i in range(4):
            patches = torch.cat([patches,im2patch(torch.rot90(tmp,i,[0,1]))])
            
    for k in tqdm(range(optimizer_steps)):
		#extract patches
        idx = torch.tensor(random.sample(range(patches.shape[0]),batch_size))
        patch_example = patches[idx,:]   
        
        #compute loss
        loss = 0
        invs, jac_inv = patchNR(patch_example, rev = True)
        loss +=  torch.mean(0.5 * torch.sum(invs**2, dim=1) - jac_inv)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    weights = dict()
    weights['batch_size'] = batch_size
    weights['optimizer_steps'] = optimizer_steps
    weights['patch_size'] = patch_size
    weights['net_state_dict'] = patchNR.state_dict()
    #torch.save(weights, 'patchNR_zeroshot_weights.pt')


def patchNR(img, lam, patch_size, n_patches_out, flow, n_iter_max, center, operator):
    """
    Defines the reconstruction using patchNR as regularizer
    """
    # fixed parameters
    operator = operator
    center = center 
    init = torch.nn.functional.interpolate(img,scale_factor=2,mode='bicubic')
    #save_img(init,'bicubic')
    
    # create patch extractors
    input_im2pat = patch_extractor(patch_size, pad=False, center=center)

    # intialize optimizer for image
    fake_img = init.clone().detach().requires_grad_(True).to(DEVICE)
    optim_img = torch.optim.Adam([fake_img], lr=0.01)

    # Main loop
    for it in tqdm(range(n_iter_max)):
        optim_img.zero_grad()        
        tmp = torch.nn.functional.pad(fake_img, pad = (7,7,7,7), mode= 'reflect')
        fake_data = input_im2pat(tmp,n_patches_out)
        
        #patchNR
        pred_inv, log_det_inv = flow(fake_data,rev=True)    
        reg = torch.mean(torch.sum(pred_inv**2,dim=1)/2) - torch.mean(log_det_inv)
        #data fidelity
        data_fid = torch.sum((operator(tmp) - img)**2)

        #loss
        loss =  data_fid + lam*reg
        loss.backward()
        optim_img.step()
    return fake_img
    
def run_ZeroShot_patchNR(img, load_model = False):
    patch_size = 6
    center = False
    #params for training
    train_steps = 10000
    batch_size = 128
    
    #params for reconstruction
    n_pat = 80000
    lam = 0.25
    n_iter = 60
    model = create_NF(num_layers = 5, sub_net_size = 512, dimension=patch_size**2)
    if load_model:
        weights = torch.load('patchNR_zeroshot_weights.pt')
        patch_size = weights['patch_size']
        model.load_state_dict(weights['net_state_dict'])
    else:
        train_patchNR(model, img, patch_size, train_steps, batch_size, center = center)
    reco = patchNR(img, lam, patch_size, n_pat, model, n_iter, center = center, operator = Downsample)
    return reco

if __name__ == '__main__':
    hr = imread('input_imgs/img_test_bsd.png')
    lr = Downsample(hr)
    lr += 0.01 * torch.randn_like(lr)
    #save_img(lr,'lr_img')
    pred = run_ZeroShot_patchNR(lr, load_model = False)
    save_img(pred,'results/patchNR_zeroshot')
    

