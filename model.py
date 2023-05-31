# This code belongs to the paper
#
# F. Altekrüger, A. Denker, P. Hagemann, J. Hertrich, P. Maass and G. Steidl (2023).
# PatchNR: Learning from Very Few Images by Patch Normalizing Flow Regularization.
# Inverse Problems, vol. 39, no. 6.
#
# Please cite the paper, if you use the code.

import torch
from torch import nn
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_NF(num_layers, sub_net_size, dimension):
    """
    Creates the patchNR network
    """
    def subnet_fc(c_in, c_out):
        return nn.Sequential(nn.Linear(c_in, sub_net_size), nn.ReLU(),
                             nn.Linear(sub_net_size, sub_net_size), nn.ReLU(),
                             nn.Linear(sub_net_size,  c_out))
    nodes = [Ff.InputNode(dimension, name='input')]
    for k in range(num_layers):
        nodes.append(Ff.Node(nodes[-1],
                          Fm.GLOWCouplingBlock,
                          {'subnet_constructor':subnet_fc, 'clamp':1.6},
                          name=F'coupling_{k}'))
        nodes.append(Ff.Node(nodes[-1],
                          Fm.PermuteRandom,
                          {'seed':(k+1)},
                          name=F'permute_flow_{k}'))
    nodes.append(Ff.OutputNode(nodes[-1], name='output'))

    model = Ff.ReversibleGraphNet(nodes, verbose=False).to(DEVICE)
    return model


