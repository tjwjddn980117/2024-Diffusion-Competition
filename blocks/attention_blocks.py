import torch
from torch import nn, einsum
from einops import rearrange
from einops.layers.torch import Rearrange

from ..utils.helpers import *
from Unet_blocks import RMSNorm

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        '''
        This is the LinearAttention. The LinearAttention is using for down-sampling / up-sampling. 

        Arguments:
            dim (int): the number of dimension.
            heads (int): the number of heads.
            dim_head (int): the number of dimensions of each head. 
        
        Inputs:
            x (tensor): [B, dim, H, W]. 
        
        Outpus:
            x (tensor): [B, dim, H, W]. 
        '''
        super(LinearAttention).__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim, normalize_dim = 1)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim, normalize_dim = 1)
        )

    def forward(self, x):
        residual = x

        b, c, h, w = x.shape

        x = self.norm(x)

        # qkv = ([B, hidden_dim, H, W], [B, hidden_dim, H, W], [B, hidden_dim, H, W])
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        # q, k, v = [B, heads, dim_head, pixels]
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)


        q = q.softmax(dim = -2) # sum of dim_heads are 1. 
        k = k.softmax(dim = -1) # sum of pixels are 1. 

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        # out = [B, heads, dim_head, pixels]
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        # out = [B, hidden_dim, H, W]
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)

        return self.to_out(out) + residual

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32, scale = 8, dropout = 0.):
        '''
        This is the Attention. The Attention is using for Transformer(ViT). 
        
        Arguments:
            dim (int): the number of dimension. 
            heads (int): the number of heads.
            dim_head (int): the number of dimensions of each head. 
            scale (int): parammeter of resizing attention tensor. 
            dropout (int): parammeter of dropouts. 
        
        Inputs:
            x (tensor): [B, pixel(H*W), dim]. 
        
        Outputs:
            x (tensor): [B, pixel(H*W), dim]. 
        '''
        super(Attention).__init__()
        self.scale = scale
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias = False)

        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_out = nn.Linear(hidden_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        # qkv = ([B, pixels, hidden_dim], [B, pixels, hidden_dim], [B, pixels, hidden_dim])
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        # q, k, v = [B, heads, pixels, dim_head]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        q, k = map(l2norm, (q, k))

        q = q * self.q_scale
        k = k * self.k_scale

        # sim = [B, heads, pixels, pixels]
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = sim.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        # out = [B, heads, pixels, dim_head]
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # out = [B, pixels, hidden_dim]
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, cond_dim, mult = 4, dropout = 0.):
        '''
        Arguments:
            dim (int): the number of dimension.
            cond_dim (int): the number of conditional dimension. 
            mult (int): making the 'dim_hidden' with (dim_hidden = dim * mult). 
            dropout (int): parammeter of dropouts. 

        Inputs:
            x (tensor): [B, pixel(H*W), dim]. 
            t (tensor): [L, cond_dim]. 

        Outputs:
            x (tensor): [B, pixel(H*W), dim]. 
        '''
        super(FeedForward).__init__()
        self.norm = RMSNorm(dim, scale = False)
        dim_hidden = dim * mult

        self.to_scale_shift = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, dim_hidden * 2),
            Rearrange('b d -> b 1 d')
        )

        to_scale_shift_linear = self.to_scale_shift[-2]
        nn.init.zeros_(to_scale_shift_linear.weight)
        nn.init.zeros_(to_scale_shift_linear.bias)

        self.proj_in = nn.Sequential(
            nn.Linear(dim, dim_hidden, bias = False),
            nn.SiLU()
        )

        self.proj_out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim, bias = False)
        )

    def forward(self, x, t):
        x = self.norm(x)
        # x = [B, pixels, dim_hidden]. 
        x = self.proj_in(x)

        # scale , shift = [L, 1, dim_hidden].
        scale, shift = self.to_scale_shift(t).chunk(2, dim = -1)
        x = x * (scale + 1) + shift

        return self.proj_out(x)