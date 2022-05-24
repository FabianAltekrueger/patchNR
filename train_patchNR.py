# This code belongs to the paper
#
# F. Alteküger, A. Denker, P. Hagemann, J. Hertrich, P. Maass and G. Steidl.
# PatchNR: Learning from Small Data by Patch Normalizing Flow Regularization
# ArXiv Preprint#xxxx.xxxxx
#
# Please cite the paper, if you use the code.
# The script trains the patchNR

import torch
from torch import nn
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import numpy as np
import model
from tqdm import tqdm
import utils

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#choose the image class out of image_classes
image_classes = ['material', 'lung', 'lentils']
image_class = image_classes[2]
print('Image class:' + image_class) 

if image_class == 'material':
    example_img = utils.imread('input_imgs/img_learn_material.png')
    val_img = utils.imread('input_imgs/img_val_material.png')

elif image_class == 'lung':
    from dival import get_standard_dataset
    dataset = get_standard_dataset('lodopab', impl='astra_cuda')
    train = dataset.create_torch_dataset(part='train',
                        reshape=((1,1,) + dataset.space[0].shape,
                        (1,1,) + dataset.space[1].shape))
    val_set = dataset.create_torch_dataset(part='validation',
                        reshape=((1,1,) + dataset.space[0].shape,
                        (1,1,) + dataset.space[1].shape))
    example_img = torch.cat([train[3][1],train[5][1],train[8][1],
                    train[11][1],train[37][1],train[75][1]]).to(DEVICE)
    val_img = val_set[1][1].to(DEVICE)     
               
elif image_class == 'lentils':    
    example_img = utils.imread('input_imgs/img_learn_lentils.png')
    val_img = utils.imread('input_imgs/img_val_lentils.png')

else:
	print('Image class is not known')
	exit()

if __name__ == '__main__':
    patch_size = 6
    num_layers = 5
    subnet_nodes = 512
    patchNR = model.create_NF(num_layers, subnet_nodes, dimension=patch_size**2)

    batch_size = 32
    optimizer_steps = 750000
    optimizer = torch.optim.Adam(patchNR.parameters(), lr = 1e-4)
    im2patch = utils.patch_extractor(patch_size=patch_size)
    for k in tqdm(range(optimizer_steps)):
		#extract patches
        idx = np.random.randint(0,example_img.shape[0])
        patch_example = im2patch(example_img[idx].unsqueeze(0),batch_size)    
        
        #compute loss
        loss = 0
        invs, jac_inv = patchNR(patch_example, rev = True)
        loss +=  torch.mean(0.5 * torch.sum(invs**2, dim=1) - jac_inv)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #validation step
        if k%1000 ==0: 
            with torch.no_grad():
                val_patch = im2patch(val_img,batch_size)
                invs, jac_inv = patchNR(val_patch, rev = True)
                val_loss =  torch.mean(0.5 * torch.sum(invs**2, dim=1) - jac_inv).item()
            print(k)
            print(loss.item())
            print(val_loss)
        #save weights    
        if (k+1) % 50000 == 0:
            it = int((k+1)/1000)
            #torch.save({'net_state_dict': patchNR.state_dict()}, 'patchNR_weights/weights_'+image_class + '_'+str(it) + '.pth')        	
    torch.save({'net_state_dict': patchNR.state_dict()}, 'patchNR_weights/weights_'+image_class + '.pth')
