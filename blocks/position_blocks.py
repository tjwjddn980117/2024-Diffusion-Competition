# sinusoidal positional embeds
import torch
from torch import nn
import math
from einops import rearrange

class LearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb 
        This is for Transformer positioning embedding. 
    
        Aguments:
            dim (int): number of dimension. 
            is_random (bool): select the random. 
        
        Inputs:
            x (tensor): [L]. the lenght of tensor. 
        
        Outputs:
            fouriered (tensor): [L, dim]. 
    
        """
        """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """
        super(LearnedSinusoidalPosEmb).__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered