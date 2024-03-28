from torch import nn
from torch import sqrt
from torch.special import expm1
from torch.cuda.amp import autocast

from einops import repeat

from UViT import UViT
from ..utils.normalize import logsnr_schedule_cosine, logsnr_schedule_shifted, logsnr_schedule_interpolated
from ..utils.helpers import exists

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
        return next(self.model.parameters()).device

    def p_mean_variance(self, x, time, time_next):
        '''
        
        '''
        log_snr = self.log_snr(time)
        log_snr_next = self.log_snr(time_next)
        c = -expm1(log_snr - log_snr_next)

        squared_alpha, squared_alpha_next = log_snr.sigmoid(), log_snr_next.sigmoid()
        squared_sigma, squared_sigma_next = (-log_snr).sigmoid(), (-log_snr_next).sigmoid()

        alpha, sigma, alpha_next = map(sqrt, (squared_alpha, squared_sigma, squared_alpha_next))

        batch_log_snr = repeat(log_snr, ' -> b', b = x.shape[0])
        pred = self.model(x, batch_log_snr)

        if self.pred_objective == 'v':
            x_start = alpha * x - sigma * pred

        elif self.pred_objective == 'eps':
            x_start = (x - sigma * pred) / alpha

        x_start.clamp_(-1., 1.)

        model_mean = alpha_next * (x * (1 - c) / alpha + c * x_start)

        posterior_variance = squared_sigma_next * c

        return model_mean, posterior_variance