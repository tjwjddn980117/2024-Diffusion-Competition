from torch import nn
from ..blocks.attention_blocks import Attention, FeedForward

class Transformer(nn.Module):
    def __init__(self, dim, time_cond_dim, depth,
                    dim_head = 32, heads = 4, ff_mult = 4, dropout = 0.,):
        '''
        the Transformer blocks. 

        Arguments:
            dim (int): the number of dimension. 
            time_cond_dim (int): the number of conditional dimension. 
            depth (int): the depth of encoding with Attention. 
            dim_head (int): the number of dimensions of each head. 
            heads (int): the number of heads. 
            ff_mult (int): making the 'dim_hidden' with (dim_hidden = dim * mult). 
            dropout (int): parammeter of dropouts. 
        
        Inputs:
            x (tensor): [B, pixel(H*W), dim]. 
            t (tensor): [L, time_cond_dim]. 
            
        Outputs:
            x (tensor): [B, pixel(H*W), dim]. 
        '''

        super(Transformer).__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = dropout),
                FeedForward(dim = dim, mult = ff_mult, cond_dim = time_cond_dim, dropout = dropout)
            ]))

    def forward(self, x, t):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x, t) + x

        return x