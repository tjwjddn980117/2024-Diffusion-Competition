import torch
from torch import nn
from functools import partial

from ..blocks.attention_blocks import Attention, LinearAttention
from ..blocks.CNN_blocks import ResnetBlock
from ..blocks.Unet_blocks import Upsample, Downsample
from ..blocks.positional_blocks import RandomOrLearnedSinusoidalPosEmb, SinusoidalPosEmb
from ..utils.functions import default, cast_tuple, divisible_by

from ..utils.version import __version__

class Unet(nn.Module):
    def __init__(self, 
        dim, 
        init_dim = None,
        out_dim = None,
        dim_mults = (1, 2, 4, 8),
        channels = 3,
        self_condition = False,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        sinusoidal_pos_emb_theta = 10000,
        attn_dim_head = 32,
        attn_heads = 4,
        full_attn = None,    # defaults to full attention only for inner most layer
        flash_attn = False
    ):
        '''
        The model of Unet. 

        Arguments:
            dim (int): the number of dimension. 
            init_dim (int): the dimension of first layer. 
            out_dim (int): the dimension of last layer. 
            dim_mults (iter): the multiples of dimentions with deep layers. 
            channels (int): the input channles. 
            self_condition (bool): if self_conditioin, input_channels = input_channels // 2. 
            resnet_block_groups (int): standard with grouping channels for group normalize. 
            learned_variance (bool): if learned_variance, channels = channels // 2. 
            learned_sinusoidal_cond (bool): SinusoidalPosEmb will Random. 
            random_fourier_features (bool): SinusoidalPosEmb will Random. 
            learned_sinusoidal_dim (int): dimension of learned_sinusoidal_cond. 
            sinusoidal_pos_emb_theta (int): theta of sinusoidal_pos_emb. 
            attn_dim_head (int): dimensioin of each head. 
            attn_heads (int): num of heads. 
            full_attn (bool): # defaults to full attention only for inner most layer
            flash_attn (bool): chose the flash. 
        
        Inputs:
            x (tensor): [B, C, H, W]. 
            time (tensor): [B]. the lenght of tensor. 
            x_self_cond (tensor): [b, c, H, W]. 
        
        Outputs:
            x (tensor): [B, C, H, W]. 
        '''
        super(Unet).__init__()

        # determine dimensions
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        # dims = [init_dim, m*dim_0, m*dim_1, m*dim_2, ...]. 
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        # in_out = [(init_dim, m*dim_0), (m*dim_0, m*dim_1), (m*dim_1, m*dim_2), ...]. 
        # in_out has the same size with dim_mults. 
        in_out = list(zip(dims[:-1], dims[1:]))

        # every 'ResnetBlock' instance's groups should be 'resnet_block_groups'. 
        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings
        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # attention
        if not full_attn:
            # if full_attn is None, full_attn = [ False, False, ... , False, True ] with len(dim_mults). 
            full_attn = (*((False,) * (len(dim_mults) - 1)), True)

        # num_stages are same with len(in_out). 
        num_stages = len(dim_mults)
        full_attn  = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        FullAttention = partial(Attention, flash = flash_attn)

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            # is_last check it's the last layer. 
            is_last = ind >= (num_resolutions - 1)

            # if layer_full_attn is True, then, Attention. Else, LinearAttention. 
            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                attn_klass(dim_in, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = FullAttention(mid_dim, heads = attn_heads[-1], dim_head = attn_dim_head[-1])
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)

            # if layer_full_attn is True, then, Attention. Else, LinearAttention. 
            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                attn_klass(dim_out, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, time, x_self_cond = None):
        # check the posibility of downsampling. 
        assert all([divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x) + x
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x) + x

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)