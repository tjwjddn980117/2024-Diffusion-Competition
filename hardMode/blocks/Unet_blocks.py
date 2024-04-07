import torch
from torch import nn
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from ..utils.functions import *

def Upsample(dim, dim_out = None):
    '''
    Inputs:
        dim (int): input dimension.
        dim_out (bool): choose to out with same dim, or different dim.
    
    Outputs:
        nn.Sequential(nn.Upsample -> nn.Conv2d)
        input: [b,c,h,w]
        output:[b,c,2h,2w]
    '''
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    '''
    Inputs:
        dim (int): input dimension.
        dim_out (bool): choose to out with same dim, or different dim. 
    
    Outputs:
        nn.Sequential(Rearrange[b,c,2h,2w]->[b,4c,h,w] -> nn.Conv2d)
        input: [b,c,2h,2w]
        output:[b,4c,h,w]
    '''
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class RMSNorm(nn.Module):
    def __init__(self, dim):
        '''
        It's the kind of Layer Norm. It's more efficient with calculate.
    
        Arguments:
            dim (int): input dimension.
        
        Inputs:
            x (tensor): [B, C, H, W]. 
        
        Outputs:
            x (tensor): [B, C, H, W]. 
        '''
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)
