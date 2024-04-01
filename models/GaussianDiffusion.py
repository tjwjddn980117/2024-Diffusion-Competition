import torch
from torch import nn
from torch import sqrt
from torch.special import expm1
from torch.cuda.amp import autocast
import tqdm
import torch.nn.functional as F

from einops import repeat, reduce

from UViT import UViT
from ..utils.normalize import *
from ..utils.helpers import exists, default

class GaussianDiffusion(nn.Module):
    def __init__(self, model: UViT, *, image_size, channels = 3, pred_objective = 'v',
                 noise_schedule = logsnr_schedule_cosine, noise_d = None, noise_d_low = None, noise_d_high = None,
                 num_sample_steps = 500, clip_sample_denoised = True, min_snr_loss_weight = True, min_snr_gamma = 5):
        '''
        The Gaussian Diffusion model. 

        Arguments:
            model (nn.Module): architecture with U-net. 
            image_size (int): the size of image. 
            channels (int): the number of channels. 
            pred_objective (str): 'v' means the prediction of v-space, and 'eps' means the prediction of noise. 
            noise_schedule (function): making noise with schedule. 
            noise_d ( ): if noise_d is exist, noise_schedule should be 'logsnr_schedule_shifted'. 
            noise_d_low ( ): if noise_d_low and noise_d_high is exist, noise_schedule should be 'logsnr_schedule_interpolated'. 
            noise_d_high ( ): noise_d_low and noise_d_high is exist, noise_schedule should be 'logsnr_schedule_interpolated'. 
        
        Inputs:
            img (tensor): [B, C, H, W]. 
            noise ( ): to define default noise. 
        
        Returns:
            _ (float): (loss * loss_weight).mean(). It's the value of p_loss. 
        
        '''
        
        super(GaussianDiffusion).__init__()
        # pred_objective is 'v' -> predict v-space.
        # pred_objective is 'eps' -> predict noise. 
        assert pred_objective in {'v', 'eps'}, 'whether to predict v-space (progressive distillation paper) or noise'

        self.model = model

        # image dimensions
        self.channels = channels
        self.image_size = image_size

        # training objective
        self.pred_objective = pred_objective

        # noise schedule
        # if all of 3 parameters are None, then assert. 
        assert not all([*map(exists, (noise_d, noise_d_low, noise_d_high))]), 'you must either set noise_d for shifted schedule, or noise_d_low and noise_d_high for shifted and interpolated schedule'

        # determine shifting or interpolated schedules
        self.log_snr = noise_schedule

        if exists(noise_d):
            self.log_snr = logsnr_schedule_shifted(self.log_snr, image_size, noise_d)

        if exists(noise_d_low) or exists(noise_d_high):
            assert exists(noise_d_low) and exists(noise_d_high), 'both noise_d_low and noise_d_high must be set'

            self.log_snr = logsnr_schedule_interpolated(self.log_snr, image_size, noise_d_low, noise_d_high)

        # sampling
        self.num_sample_steps = num_sample_steps
        self.clip_sample_denoised = clip_sample_denoised

        # loss weight
        self.min_snr_loss_weight = min_snr_loss_weight
        self.min_snr_gamma = min_snr_gamma

    @property
    def device(self):
        '''
        The device properties of the Gaussian class are used to determine which devices (such as CPUs or CUDA devices)
         are currently assigned the model that an instance of that class has. 
        
        using example: 
            # Identifying Devices with Current Gausian Models. \n
            current_device = gaussian_model.device \n
            print(f"The model is on {current_device}") \n

            # Move data to the same device as the model. \n
            input_data = input_data.to(gaussian_model.device)
        '''
        return next(self.model.parameters()).device

    def p_mean_variance(self, x, time, time_next):
        '''
        extract p mean and variance. 
        the p distribution means that denoise the image. 
        the mean and variance is only affected by time information. 
        time gose to ( 1.0 -> 0.0 ). 

        Arguments:
            x (tensor): [B, C, H, W]. random noise. 
            time (tensor_float): [ _ ] tensor about time. The tensor should 'zero dimension'. ex) torch.tensor(0.28)
            time_next (tensor_float): [ _ } tensor about next_time. The tensor should 'zero dimension'. ex) torch.tensor(0.28)
        '''
        log_snr = self.log_snr(time)
        log_snr_next = self.log_snr(time_next)
        # expm1 canculate with (exp(x)-1). so, c should be -(exp(x)-1). 
        # if (log_snr == log_snr_next), then, c == 0. 
        # if (log_snr < log_snr_next), then, c > 0. 
        # if (log_snr > log_snr_next), tehn, c < 0. 
        # c is also tensor. 
        c = -expm1(log_snr - log_snr_next)

        # log_snr and log_snr_next to 0.0 ~ 1.0
        squared_alpha, squared_alpha_next = log_snr.sigmoid(), log_snr_next.sigmoid()
        squared_sigma, squared_sigma_next = (-log_snr).sigmoid(), (-log_snr_next).sigmoid()

        # sqrt with values. 
        alpha, sigma, alpha_next = map(sqrt, (squared_alpha, squared_sigma, squared_alpha_next))

        # batch_log_snr = [B]. 
        batch_log_snr = repeat(log_snr, ' -> b', b = x.shape[0])
        # pred = [B, C, H, W]. 
        pred = self.model(x, batch_log_snr)

        if self.pred_objective == 'v':
            x_start = alpha * x - sigma * pred

        elif self.pred_objective == 'eps':
            x_start = (x - sigma * pred) / alpha

        # re-size x_start with -1. ~ 1.
        x_start.clamp_(-1., 1.)

        model_mean = alpha_next * (x * (1 - c) / alpha + c * x_start)

        posterior_variance = squared_sigma_next * c

        return model_mean, posterior_variance
    
    # sampling related functions
    @torch.no_grad()
    def p_sample(self, x, time, time_next):
        '''
        p_sample do the reparameterize trick. 

        Arguments:
            x (tensor): [B, C, H, W]. random noise. 
            time (int): now time.
            time_next (int): next time. 
        
        Returns:
            reparam_x (tensor): [B, C, H, W]. 
        '''
        batch, *_, device = *x.shape, x.device

        model_mean, model_variance = self.p_mean_variance(x = x, time = time, time_next = time_next)

        # if we can't search next time (the time is end). 
        if time_next == 0:
            return model_mean

        noise = torch.randn_like(x)
        return model_mean + sqrt(model_variance) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape):
        '''
        p_sample_loop doing keep denoising with num_sample_steps. 

        Arguments:
            shape (tuple): the shape would be [batch_size, channels, image_size, image_size]
        
        Reterns:
            img (tensor): [B, C, H, W]. 
        '''
        batch = shape[0]

        # random image sampling. 
        img = torch.randn(shape, device = self.device)
        # Divide equally from 1 to 0 into 'num_sample_steps'
        steps = torch.linspace(1., 0., self.num_sample_steps + 1, device = self.device)

        for i in tqdm(range(self.num_sample_steps), desc = 'sampling loop time step', total = self.num_sample_steps):
            times = steps[i]
            times_next = steps[i + 1]
            img = self.p_sample(img, times, times_next)

        img.clamp_(-1., 1.)
        img = unnormalize_to_zero_to_one(img)
        return img
    
    @torch.no_grad()
    def sample(self, batch_size = 16):
        '''
        sampling the denoising methods. 
        
        Arguments:
            batch_size (int): the number of batch size. 
        
        Returns:
            img (tensor): [B, C, H, W]. 
        '''
        return self.p_sample_loop((batch_size, self.channels, self.image_size, self.image_size))
    
    # training related functions - noise prediction
    @autocast(enabled = False)
    def q_sample(self, x_start, times, noise = None):
        '''
        the function for making noise. 

        Arguments:
            x_start (tensor): [B, C, H, W]. the start x. 
            times (tensor): [B]. random times. 
            noise ( ): to define default noise. 
        
        Returns:
            x_noised (tensor): [B, C, H, W]. the tensor with noise. 
            log_snr ((tensor)): [B]. the value only effected by 'times'. 
        '''
        noise = default(noise, lambda: torch.randn_like(x_start))

        log_snr = self.log_snr(times)

        log_snr_padded = right_pad_dims_to(x_start, log_snr)
        alpha, sigma = sqrt(log_snr_padded.sigmoid()), sqrt((-log_snr_padded).sigmoid())
        x_noised =  x_start * alpha + noise * sigma

        return x_noised, log_snr

    def p_losses(self, x_start, times, noise = None):
        '''
        the loss of de_noise process. 
        q_sample for time is random time. 

        Arguments:
            x_start (tensor): [B, C, H, W]. the start x. 
            times (tensor): [B]. random times. 
            noise ( ): to define default noise. 
        
        Returns:
            _ (float): (loss * loss_weight).mean(). loss = [B]. loss_weight = [B]. 
        '''
        # random noise. 
        noise = default(noise, lambda: torch.randn_like(x_start))

        # q_sample for make random 'log_snr' for random time. and noised with random time 'log_snr'. 
        x, log_snr = self.q_sample(x_start = x_start, times = times, noise = noise)

        # model_out is the prediction of denoise - predict image. timing of (t - 1). 
        model_out = self.model(x, log_snr)

        if self.pred_objective == 'v':
            padded_log_snr = right_pad_dims_to(x, log_snr)
            alpha, sigma = padded_log_snr.sigmoid().sqrt(), (-padded_log_snr).sigmoid().sqrt()
            # it just the target that time of (t - 1). 
            # i don't know what's the reason why it's exactly, but it says it can be. 
            target = alpha * noise - sigma * x_start

        elif self.pred_objective == 'eps':
            target = noise

        loss = F.mse_loss(model_out, target, reduction = 'none')

        loss = reduce(loss, 'b ... -> b', 'mean')
        # loss = [B]

        snr = log_snr.exp()

        maybe_clip_snr = snr.clone()
        if self.min_snr_loss_weight:
            maybe_clip_snr.clamp_(max = self.min_snr_gamma)

        if self.pred_objective == 'v':
            loss_weight = maybe_clip_snr / (snr + 1)
        elif self.pred_objective == 'eps':
            loss_weight = maybe_clip_snr / snr
        # loss_weight = [B]
        
        return (loss * loss_weight).mean()
    
    def forward(self, img, *args, **kwargs):
        '''
        the reason that the forward has only 'p_losses' is the forward only 'noising process'. 
        '''
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'

        img = normalize_to_neg_one_to_one(img)
        # times = [B]. Fill in with random values that follow an even distribution between 0 and 1. 
        times = torch.zeros((img.shape[0],), device = self.device).float().uniform_(0, 1)

        return self.p_losses(img, times, *args, **kwargs)