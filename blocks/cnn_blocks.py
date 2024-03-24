from torch import nn
from einops import rearrange
from ..utils.helpers import *

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        '''
        Arguments:
            dim (int): number of dimension.
            dim_out (int): number of out dimension. 
            groups (int): standard with grouping channels.

        Inputs:
            x (tensor): [b, dim, h, w]
            scale_shift (a, b): x(tensor)*(a+1) + b

        Outputs:
            x (tensor): [b, c, h, w]
        '''
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        '''
        Arguments:
            dim (int): number of dimension.
            dim_out (int): number of out dimension. ss
            time_emb_dim (int): if time_emb_dim is exists, mlp is 'SiLU -> Linear(time_emb_dim -> dim_out*2)
            groups (int): standard with grouping channels.

        Inputs:
            x (tensor): [B, dim, H, W]
            time_emb (tensor): [B, time_emb_dim]
        
        Outputs:
            x (tensor): [B, out_dim, H, W]
        '''
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            # time_emb = [B, dim_out * 2]
            time_emb = self.mlp(time_emb)
            # time_emb = [B, dim_out * 2, 1, 1]
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            # scale_shift = ([B, dim_out, 1, 1], [B, dim_out, 1, 1])
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)
