import torch
from torch import nn
from einops import rearrange, repeat

from Unet_blocks import RMSNorm
from attend import Attend
from functools import partial

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32, num_mem_kv = 4):
        '''
        This is the LinearAttention. The LinearAttention is using for down-sampling / up-sampling. 

        Arguments:
            dim (int): the number of dimension.
            heads (int): the number of heads.
            dim_head (int): the number of dimensions of each head. 
            num_mem_kv (int): Use predefined or learned memory information for attention calculations. 
        
        Inputs:
            x (tensor): [B, dim, H, W]. 
        
        Outputs:
            x (tensor): [B, dim, H, W]. 
        '''
        super(LinearAttention).__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        # qkv = ([B, hidden_dim, H, W], [B, hidden_dim, H, W], [B, hidden_dim, H, W]) 
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        # q, k, v = [B, heads, dim_head, pixels] 
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        # mk, mv = [B, heads, dim_head, num_mem_kv] 
        mk, mv = map(lambda t: repeat(t, 'h c n -> b h c n', b = b), self.mem_kv)
        # k, v = [B, heads, dim_head, pixels + num_mem_kv] 
        k, v = map(partial(torch.cat, dim = -1), ((mk, k), (mv, v)))

        q = q.softmax(dim = -2) # sum of dim_heads are 1. 
        k = k.softmax(dim = -1) # sum of pixels are 1. 

        q = q * self.scale
        
        # context = [B, heads, dim_head, dim_head] 
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        # out = [B, heads, dim_head, pixels] 
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        # out = [B, hidden_dim, H, W] 
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)

        return self.to_out(out)

class Attention(nn.Module):
    def __init__( self, dim, heads = 4, dim_head = 32, num_mem_kv = 4, flash = False):
        super(Attention).__init__()
        '''
        This is the Attention. The Attention is using for Transformer(ViT). 

        Arguments:
            dim (int): the number of dimension.
            heads (int): the number of heads.
            dim_head (int): the number of dimensions of each head. 
            num_mem_kv (int): Use predefined or learned memory information for attention calculations. 
            flash (bool): the selection of using flash. 
        
        Inputs:
            x (tensor): [B, dim, H, W]. 
        
        Outputs:
            x (tensor): [B, dim, H, W]. 
        '''
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.attend = Attend(flash = flash)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        # qkv = ([B, hidden_dim, H, W], [B, hidden_dim, H, W], [B, hidden_dim, H, W]) 
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        # q, k, v = [B, heads, pixels, dim_head] 
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), qkv)

        # mk, mv = [B, heads, num_mem_kv, dim_head] 
        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), self.mem_kv)
        
        # k, v = [B, heads, pixels + num_mem_kv, dim_head] 
        k, v = map(partial(torch.cat, dim = -2), ((mk, k), (mv, v)))

        # out = [B, heads, pixels, dim_head]
        out = self.attend(q, k, v)

        # out = [B, hiddem_dim, H, W]
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)

        return self.to_out(out)