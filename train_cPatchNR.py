# This code belongs to the paper
#
# F. Altekrüger, A. Denker, P. Hagemann, J. Hertrich, P. Maass and G. Steidl.
# PatchNR: Learning from Small Data by Patch Normalizing Flow Regularization
# ArXiv Preprint#2205.12021
#
# Please cite the paper, if you use the code.
# The script trains the cPatchNR

import torch
from torch import nn
import torch.nn.functional as F
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import numpy as np
import model
from tqdm import tqdm
import utils
from cPatchNR_superres import Downsample
from patchNR_deblurring import Blur


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#choose the image class out of image_classes
image_classes = ['material', 'lentils']
image_class = image_classes[0] 

def cond_operator(condition):
    """
    defines the operator which extracts information of the condition
    for material: bicubic interpolation
    for lentils: identity
    """
    if image_class == 'material':
        return F.interpolate(condition,scale_factor=4,mode='bicubic')
    elif image_class == 'lentils':
        return condition

if image_class == 'material':
    example_img = utils.imread('input_imgs/img_learn_material.png')
    operator = Downsample
    noise = 0.01
    
    val_img = utils.imread('input_imgs/img_val_material.png')
    val_obs = Downsample(val_img)
    val_obs = Downsample(val_img) + 0.01 * torch.randn(val_obs.shape,device=DEVICE)
    condition_val = F.interpolate(val_obs, scale_factor = 4, mode='bicubic')
    val_img = val_img[...,6:-6,6:-6]
    
elif image_class == 'lentils':    
    example_img = utils.imread('input_imgs/img_learn_lentils.png')
    operator = Blur(blur_id = 0).to(DEVICE)
    noise = 5/255
    
    val_img = utils.imread('input_imgs/img_val_lentils.png')
    val_obs = operator(val_img)
    val_obs = val_obs + 5/255 * torch.randn(val_obs.shape,device=DEVICE)
    condition_val = val_obs
    
else:
    print('Image class is not known')
    exit()

if __name__ == '__main__':
    patch_size = 6
    num_layers = 5
    subnet_nodes = 512
    patchNR = model.create_cNF(num_layers, subnet_nodes, dimension=patch_size**2,
                               dimension_condition = patch_size**2)

    batch_size = 32
    optimizer_steps = 750000
    optimizer = torch.optim.Adam(patchNR.parameters(), lr = 1e-4)
    im2patch = utils.patch_extractor(patch_size=patch_size)
    for k in tqdm(range(optimizer_steps)):
        #extract patches
        idx = np.random.randint(0,example_img.shape[0])
        img_gt = example_img[idx].unsqueeze(0)
        
        tmp = operator(img_gt)
        img_noise = tmp + noise * torch.randn(tmp.shape,device=DEVICE) #sample noisy img each iteration
        condition = cond_operator(img_noise)
        if image_class == 'material':
            img_gt = img_gt[...,6:-6,6:-6]
        
        patch_example = im2patch(img_gt)
        patch_cond = im2patch(condition)
        perm = torch.randperm(patch_example.shape[0])[:batch_size]
        patch_example = patch_example[perm,:]
        patch_cond = patch_cond[perm,:]
        
        #compute loss
        loss = 0
        invs, jac_inv = patchNR(patch_example, c = patch_cond, rev = True)
        loss +=  torch.mean(0.5 * torch.sum(invs**2, dim=1) - jac_inv)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #validation step
        if k%1000 ==0: 
            with torch.no_grad():
                patch_val = im2patch(val_img)
                patch_cond_val = im2patch(condition_val)
                perm = torch.randperm(patch_val.shape[0])[:batch_size]
                patch_val = patch_val[perm,:]
                patch_cond_val = patch_cond_val[perm,:]
                invs, jac_inv = patchNR(patch_val, c = patch_cond_val, rev = True)
                val_loss =  torch.mean(0.5 * torch.sum(invs**2, dim=1) - jac_inv).item()
            print(k)
            print(loss.item())
            print(val_loss)
        #save weights    
        if (k+1) % 50000 == 0:
            it = int((k+1)/1000)
            #torch.save({'net_state_dict': patchNR.state_dict()}, 'patchNR_weights/weights_cond_'+image_class + '_'+str(it) + '.pth')        	
    torch.save({'net_state_dict': patchNR.state_dict()}, 'patchNR_weights/weights_cond_'+image_class + '.pth')
