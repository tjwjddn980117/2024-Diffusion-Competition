import torch
from torch import nn
from functools import partial
from einops import rearrange, pack, unpack
from einops.layers.torch import Rearrange

from ..utils.helpers import exists, default, identity, cast_tuple
from ..blocks.cnn_blocks import ResnetBlock
from ..blocks.position_blocks import LearnedSinusoidalPosEmb
from ..blocks.Unet_blocks import Upsample, Downsample
from ..blocks.attention_blocks import LinearAttention
from transformer import Transformer

class UViT(nn.Module):
    def __init__(self, dim, init_dim = None, out_dim = None, dim_mults = (1, 2, 4, 8),
                 downsample_factor = 2, channels = 3,
                 vit_depth = 6, vit_dropout = 0.2, attn_dim_head = 32, attn_heads = 4, ff_mult = 4,
                 resnet_block_groups = 8, learned_sinusoidal_dim = 16,
                 init_img_transform: callable = None, final_img_itransform: callable = None, patch_size = 1, dual_patchnorm = False):
        '''
        This model is the ViT based with U-net. The basic architecture is Down-Sampling -> Transformer(ViT) -> Up-Sampling. 

        Arguments:
            dim (int): the number of dimension. 
            init_dim (int): the dimension of first layer. 
            out_dim (int): the dimension of last layer. 
            dim_mults (iter): the multiples of dimentions with deep layers. 
            downsample_factor (int): down/up sampling with * factor. if factor is 2, down with /2, up with *2
            channels (int): the input channles. 
            vit_depth (int): the depth of vit. 
            vit_dropout (float): the rate of dropout with vit. 
            attn_dim_head (int): 
            attn_heads (int):
            ff_mult (int): 
            resnet_block_groups (int): standard with grouping channels for group normalize. 
            learned_sinusoidal_dim (int): 
            init_img_transform (function): the function for pre-processing. 
            final_img_itransform (function): the function for post-precessing. 
            patch_size (int): patch for reducing the the input size. 
            dual_patchnorm (bool): normalizing with LayerNorm while doing patch.
        
        Inputs: 
            x (tensor): [B, C, H, W]. 
            time (tensor): [B]. the lenght of tensor. 

        Outputs: 
            x (tensor): [B, C, H, W]. 
        '''
        super(UViT).__init__()

        # for initial dwt transform (or whatever transform researcher wants to try here)
        # just check the shape is same. 
        if exists(init_img_transform) and exists(final_img_itransform):
            init_shape = torch.Size(1, 1, 32, 32)
            # mock_tensor sample from Gaussian Distribution. 
            mock_tensor = torch.randn(init_shape) 
            # init_shape should same with final_img_itransform. 
            assert final_img_itransform(init_img_transform(mock_tensor)).shape == init_shape

        # if img_transform exist, img_transform is img_transform
        # else, img_transform is 'identity', that means the function return themself. 
        self.init_img_transform = default(init_img_transform, identity)
        self.final_img_itransform = default(final_img_itransform, identity)

        input_channels = channels

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        # whether to do initial patching, as alternative to dwt

        self.unpatchify = identity

        # prepare 'needs_patch'

        input_channels = channels * (patch_size ** 2)
        needs_patch = patch_size > 1

        if needs_patch:
            if not dual_patchnorm:
                # the size of image will reduce. 
                # the size will reduce with (H,W) -> (H/patch_size, W/patch_size)
                self.init_conv = nn.Conv2d(channels, init_dim, patch_size, stride = patch_size)
            else:
                # the size will reduce with (H,W) -> (H/patch_size, W/patch_size)
                # the layer with LayerNorm.     
                self.init_conv = nn.Sequential(
                    Rearrange('b c (h p1) (w p2) -> b h w (c p1 p2)', p1 = patch_size, p2 = patch_size),
                    nn.LayerNorm(input_channels),
                    nn.Linear(input_channels, init_dim),
                    nn.LayerNorm(init_dim),
                    Rearrange('b h w c -> b c h w')
                )
            # up-samplling with original size and channels. 
            # (we reduce the H W, and we increase the channels.)
            self.unpatchify = nn.ConvTranspose2d(input_channels, channels, patch_size, stride = patch_size)

        # determine dimensions
            
        # dims = [init_dim, m*dim_0, m*dim_1, m*dim_2, ...]
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        # in_out = [(init_dim, m*dim_0), (m*dim_0, m*dim_1), (m*dim_1, m*dim_2), ...]
        in_out = list(zip(dims[:-1], dims[1:]))

        # every 'ResnetBlock' instance's groups should be 'resnet_block_groups'. 
        resnet_block = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings
        time_dim = dim * 4

        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
        fourier_dim = learned_sinusoidal_dim + 1

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # downsample factors

        # downsample_factor = (downsample_factor(int) for len(dim_mults))
        # ex) downsample_factor=2, len(dim_mults)=4,
        # dwonsample_factor = (2, 2, 2, 2)
        downsample_factor = cast_tuple(downsample_factor, len(dim_mults))
        assert len(downsample_factor) == len(dim_mults)

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # total 4 times for down sampling. 
        for ind, ((dim_in, dim_out), factor) in enumerate(zip(in_out, downsample_factor)):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                resnet_block(dim_in, dim_in, time_emb_dim = time_dim),
                resnet_block(dim_in, dim_in, time_emb_dim = time_dim),
                LinearAttention(dim_in),
                Downsample(dim_in, dim_out, factor = factor)
            ]))

        # down sampling -> mid dim -> up sampling. 
        mid_dim = dims[-1]

        self.vit = Transformer(
            dim = mid_dim,
            time_cond_dim = time_dim,
            depth = vit_depth,
            dim_head = attn_dim_head,
            heads = attn_heads,
            ff_mult = ff_mult,
            dropout = vit_dropout
        )

        for ind, ((dim_in, dim_out), factor) in enumerate(zip(reversed(in_out), reversed(downsample_factor))):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                Upsample(dim_out, dim_in, factor = factor),
                resnet_block(dim_in * 2, dim_in, time_emb_dim = time_dim),
                resnet_block(dim_in * 2, dim_in, time_emb_dim = time_dim),
                LinearAttention(dim_in),
            ]))

        default_out_dim = input_channels
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = resnet_block(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time):
        # x = [B, 3, H, W]
        x = self.init_img_transform(x)
        
        # x = [B, init_dim/dim, H, W]
        x = self.init_conv(x)
        r = x.clone()

        # time = [B]
        t = self.time_mlp(time)
        # t = [B, time_dim]

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = rearrange(x, 'b c h w -> b h w c')
        x, ps = pack([x], 'b * c')

        x = self.vit(x, t)

        x, = unpack(x, ps, 'b * c')
        x = rearrange(x, 'b h w c -> b c h w')

        for upsample, block1, block2, attn in self.ups:
            x = upsample(x)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        x = self.final_conv(x)

        # if, need patch == true, unpatchify make the channel to 3. 
        x = self.unpatchify(x)
        return self.final_img_itransform(x)