# This code belongs to the paper
#
# F. Altekrüger, A. Denker, P. Hagemann, J. Hertrich, P. Maass and G. Steidl (2023).
# PatchNR: Learning from Very Few Images by Patch Normalizing Flow Regularization.
# Inverse Problems, vol. 39, no. 6.
#
# Please cite the paper, if you use the code.
#
# The script reproduces the numerical example CT in the paper.

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import model
from tqdm import tqdm
import utils
import dival
from dival import get_standard_dataset
import odl
from odl.contrib.torch import OperatorModule
from dival.util.torch_losses import poisson_loss
from functools import partial

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)


def patchNR(img, lam, patch_size, n_patches_out, patchNR, n_iter_max, operator):
    """
    Defines the reconstruction using patchNR as regularizer
    """
                         
    # fixed parameters
    obs = img.to(DEVICE)
    operator = operator
    center = False 
    init = fbp(obs)
    pad_size = 4 #pad the image before extracting patches to avoid boundary effects
    pad = [pad_size]*4  

    # create patch extractors
    input_im2pat = utils.patch_extractor(patch_size, pad=False, center=center)

    # intialize optimizer for image
    fake_img = torch.tensor(init.clone(),dtype=torch.float,device=DEVICE,requires_grad = True)
    optim_img = torch.optim.Adam([fake_img], lr=0.005)
    
    #define the poisson loss
    photons_per_pixel = 4096
    mu_max = 81.35858
    criterion = partial(poisson_loss,photons_per_pixel=photons_per_pixel,mu_max=mu_max)
    
    # Main loop
    for it in tqdm(range(n_iter_max)):
        optim_img.zero_grad()
        tmp = nn.functional.pad(fake_img,pad,mode='reflect')
        fake_data = input_im2pat(tmp,n_patches_out)
        
        #patchNR
        pred_inv, log_det_inv = patchNR(fake_data,rev=True)    
        reg = torch.mean(torch.sum(pred_inv**2,dim=1)/2) - torch.mean(log_det_inv)
       
        #data fidelity
        data_fid = criterion(operator(fake_img),obs)
        
        #loss
        loss =  data_fid + lam*reg
        loss.backward()
        optim_img.step()    
        
    return fake_img


if __name__ == '__main__':
    #choose the number of angles
    angle_types = ['full','limited']
    angle_type = angle_types[1]	
    
    #input parameters
    patch_size = 6
    num_layers = 5
    subnet_nodes = 512

    #load model
    net = model.create_NF(num_layers, subnet_nodes, dimension=patch_size**2)
    weights = torch.load('patchNR_weights/weights_lung.pth')
    net.load_state_dict(weights['net_state_dict'])
	
    #load images
    dataset = get_standard_dataset('lodopab', impl='astra_cuda')
    test = dataset.create_torch_dataset(part='test',
                        reshape=((1,1,) + dataset.space[0].shape,
                        (1,1,) + dataset.space[1].shape))
    ray_trafo = dataset.ray_trafo                    
    if angle_type == 'limited':
        lim_dataset = dival.datasets.angle_subset_dataset.AngleSubsetDataset(dataset,
	                   slice(100,900),impl='astra_cuda')                   
        test = lim_dataset.create_torch_dataset(part='test',
                        reshape=((1,1,) + lim_dataset.space[0].shape,
                        (1,1,) + lim_dataset.space[1].shape))  
        ray_trafo = lim_dataset.ray_trafo                            

    gt = test[64][1]
    obs = test[64][0]
    gt = test[39][1]
    obs = test[39][0]
    
    #load operator and FBP
    operator = OperatorModule(ray_trafo).to(DEVICE)
    fbp = odl.tomo.analytic.filtered_back_projection.fbp_op(ray_trafo,
	                    filter_type = 'Hann', frequency_scaling = 0.641025641025641)
    fbp = OperatorModule(fbp)
    
    lam = 700
    n_pat = 40000
    iteration = 300
    if angle_type == 'limited':
        iteration = 3000
    
    rec = patchNR(obs,lam = lam, patch_size = patch_size, n_patches_out = n_pat,
                  patchNR = net, n_iter_max = iteration, operator = operator)
    utils.save_img(rec,'results/patchNR_'+angle_type+'_angleCT')
    #torch.save(rec,'results/patchNR_'+angle_type+'_angleCT_tens.pt')
    

