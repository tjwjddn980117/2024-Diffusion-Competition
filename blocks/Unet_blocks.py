import torch
from torch import nn
import torch.nn.functional as F

from einops import repeat
from ..utils.helpers import *
from einops.layers.torch import Rearrange

class Upsample(nn.Module):
    def __init__(self, dim, dim_out = None,factor = 2):
        '''
        Arguments:
            dim (int): input dimension.
            dim_out (bool): choose to out with same dim, or different dim.
            factor (int): upsampling size. you can think about the size.

        Inputs:
            x (tensor): [B, C, H, W]

        Outputs:
            x (tensor): [B, C, factor*H, factor*W]
        '''
        super(Upsample).__init__()
        self.factor = factor
        self.factor_squared = factor ** 2

        dim_out = default(dim_out, dim)
        conv = nn.Conv2d(dim, dim_out * self.factor_squared, 1)

        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            nn.PixelShuffle(factor)
        )

        self.init_conv_(conv)

    def init_conv_(self, conv):
        '''
        Initing weight with [B, C, H, W]. 
        '''
        o, i, h, w = conv.weight.shape
        conv_weight = torch.empty(o // self.factor_squared, i, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        # In this case, we have the same weight in units of factor_squared.
        conv_weight = repeat(conv_weight, 'o ... -> (o r) ...', r = self.factor_squared)

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        return self.net(x)

def Downsample(dim, dim_out = None, factor = 2):
    '''
    Inputs:
        dim (int): input dimension.
        dim_out (bool): choose to out with same dim, or different dim. 
        factor (int): upsampling size. you can think about the size.
    
    Outputs:
        nn.Sequential(Rearrange[B, C, 2H, 2W]->[B, 4C, H, W] -> nn.Conv2d)
        input: [B, C, 2H, 2W]
        output:[B, 4C, H, W]
    '''
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = factor, p2 = factor),
        nn.Conv2d(dim * (factor ** 2), default(dim_out, dim), 1)
    )

class RMSNorm(nn.Module):
    def __init__(self, dim, scale = True, normalize_dim = 2):
        '''
        It's the kind of Layer Norm. It's more efficient with calculate.
    
        Arguments:
            dim (int): input dimension. 
            scale (bool): decide the scaling. 
            normalize_dim (int): defualt is 2. the dimension that you want to normalize.
        
        Inputs:
           x (tensor): [B, C, H, W]. 
        
        Outputs:
            if scale=True, x (tensor): [B, C, H, W]. 
            if scale=False, x (tensor): [B, C, H, W]. 
        '''
        super(RMSNorm).__init__()
        self.g = nn.Parameter(torch.ones(dim)) if scale else 1

        self.scale = scale
        self.normalize_dim = normalize_dim

    def forward(self, x):
        normalize_dim = self.normalize_dim
        # x.ndim -> the dimension of x 
        # if self.scale: scale = [dim, 1], else: scale = 1
        scale = append_dims(self.g, x.ndim - self.normalize_dim - 1) if self.scale else 1
        # standard is 'normalize_dim'.
        # (x.shape[normalize_dim] ** 0.5) is √(H)
        return F.normalize(x, dim = normalize_dim) * scale * (x.shape[normalize_dim] ** 0.5)
