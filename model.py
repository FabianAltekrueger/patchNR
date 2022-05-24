# This code belongs to the paper
#
# F. Alteküger, A. Denker, P. Hagemann, J. Hertrich, P. Maass and G. Steidl.
# PatchNR: Learning from Small Data by Patch Normalizing Flow Regularization
# ArXiv Preprint#xxxx.xxxxx
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

def create_cNF(num_layers, sub_net_size, dimension, dimension_condition):
    """
    Creates the cPatchNR network
    """
    def subnet_fc(c_in, c_out):
        return nn.Sequential(nn.Linear(c_in, sub_net_size), nn.ReLU(),
                             nn.Linear(sub_net_size, sub_net_size), nn.ReLU(),
                             nn.Linear(sub_net_size,  c_out))
    nodes = [Ff.InputNode(dimension, name='input')]
    cond = Ff.ConditionNode(dimension_condition, name='cond')
    for k in range(num_layers):
        nodes.append(Ff.Node(nodes[-1],
                          Fm.GLOWCouplingBlock,
                          {'subnet_constructor':subnet_fc, 'clamp':1.6},
                          conditions = cond,
                          name=F'coupling_{k}'))
        nodes.append(Ff.Node(nodes[-1],
                          Fm.PermuteRandom,
                          {'seed':(k+1)},
                          name=F'permute_flow_{k}'))
    nodes.append(Ff.OutputNode(nodes[-1], name='output'))

    model = Ff.ReversibleGraphNet(nodes+[cond], verbose=False).to(DEVICE)
    return model

