import torch
from torch import nn
import math
from einops import rearrange
from ..utils.functions import divisible_by

# sinusoidal positional embeds
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta = 10000):
        '''
        for sinusoidal positional embedding.

        Arguments:
            dim (int): number of dimension.
            theta (int): number of theta.

        Inputs:
            x (tensor): [L]. the length of tensor.

        Outputs:
            emb (tensor): [L, emb]. 
        '''
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim, is_random = False):
        """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb 
    
        Aguments:
            dim (int): number of dimension.
            is_random (bool): select the random.
        
        Inputs:
            x (tensor): [L]. the lenght of tensor.
        
        Outputs:
            fouriered (tensor): [L, dim].
    
        """
        """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered