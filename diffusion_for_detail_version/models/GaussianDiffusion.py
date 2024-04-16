import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from random import random
from functools import partial
from tqdm.auto import tqdm
from collections import namedtuple

from einops import rearrange, reduce

from ..utils.functions import default, normalize_to_neg_one_to_one, unnormalize_to_zero_to_one, identity
from ..utils.gaussian_functions import extract, cosine_beta_schedule, linear_beta_schedule, sigmoid_beta_schedule

# constants
ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_v',
        beta_schedule = 'sigmoid',
        schedule_fn_kwargs = dict(),
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        offset_noise_strength = 0.,  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
        min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
        min_snr_gamma = 5
    ):
        super(GaussianDiffusion).__init__()
        '''
        The model of Gaussian Diffusion. 

        Arguments:
            model (nn.Module): the based with Unet. 
            *,
            image_size (int): the size of image. 
            timesteps (int): the full timesteps. 
            sampling_timesteps (int): the sampling timesteps. 
            objective (str): the object you want to diffusion. 
            beta_schedule (str): scheduling methods. 
            schedule_fn_kwargs (dict): the more arguments (parameters) of scheduling function. 
            ddim_sampling_eta (float): using from ddim_sampling. 
            auto_normalize (bool): auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False. 
            offset_noise_strength = 0.,  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
            min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
            min_snr_gamma = 5
        
        Inputs: 
            img (tensor): [B, C, H, W]. 
        
        Ouputs:
            loss (float): value of loss. 
        '''
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert not hasattr(model, 'random_or_learned_sinusoidal_cond') or not model.random_or_learned_sinusoidal_cond

        self.model = model

        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        self.image_size = image_size

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        
        # betas = [timesteps]. 
        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        # alphas = [timesteps]. 
        alphas = 1. - betas
        # stacking multiple. 누적곱. alphas_cumprod = [timesteps]. 
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        # stacking multiple with the first number is '1'. alphas_cumprod_prev = [timesteps]. 
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters
        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        # sampling_timesteps should smaller then timesteps. 
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # offset noise strength - in blogpost, they claimed 0.1 was ideal
        self.offset_noise_strength = offset_noise_strength

        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556
        # maybe_clipped_snr = [timesteps]. 
        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False
        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    @property
    def device(self):
        return self.betas.device

    def predict_start_from_noise(self, x_t, t, noise):
        '''
        the function for predict start. (p distribution). 

        Arguments:
            t (tensor): [B] time sequence. 
            x_t (tensor): [B, C, H, W]. x_t noise. 
            noise (tensor): [B, C, H, W]. x_(t-1) noise. 
        
        Returns:
            _ (tensor): [B, C, H, W]. predict the start(x_0), from noise (x_(t-1)). 
        '''
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        '''
        the function for predict noise. (q distribution). 

        Arguments:
            t (tensor): [B] time sequence. 
            x_t (tensor): [B, C, H, W]. x_t noise. 
            x0 (tensor): [B, C, H, W]. x_0 noise. 
        
        Returns:
            _ (tensor): [B, C, H, W]. predict the noise(x_(t-1)), from start (x_0). 
        '''
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        '''
        the function for predict v from real x_start. (p distribution). 

        Arguments:
            t (tensor): [B] time sequence. 
            x_start (tensor): [B, C, H, W]. x_0 img. 
            noise (tensor): [B, C, H, W]. x_(t-1) noise. 
        
        Returns:
            _ (tensor): [B, C, H, W]. predict the v, from start (x_0). 
        '''
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        '''
        the function for predict start. (p distribution). 

        Arguments:
            t (tensor): [B] time sequence. 
            x_t (tensor): [B, C, H, W]. x_t noise. 
            v (tensor): [B, C, H, W]. x_(t-1) v. 
        
        Returns:
            _ (tensor): [B, C, H, W]. predict the start(x_0), from v (x_(t-1)). 
        '''
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        '''
        the function for predict x_t's parameters. (q distribution). 

        Arguments:
            t (tensor): [B] time sequence. 
            x_start (tensor): [B, C, H, W]. x_0 start. 
            x_t (tensor): [B, C, H, W]. x_t noise. 
        
        Returns:
            model_mean (tensor): [B, C, H, W]. parameter of x_t mean. 
            posterior_variance (tensor): [B, C, H, W]. parameter of x_t variance. 
            posterior_log_variance (tensor): [B, C, H, W]. parameter of x_t log_variance. 
        '''
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False):
        '''
        the model predictions for 'pred_noise and x_start'. 
        there are few objective with this function. 
        'pred_noise', 'pred_x0', 'pred_v'. 

        Arguments:
            t (tensor): [B] time sequence. 
            x (tensor): [B, C, H, W]. input noise in 't' sequence. (x_t). 
            x_self_cond (bool): choose to input the self_condition. 
        
        Returns:
            if objective == 'pred_noise': 
                1. input x_t and Unet to x_(t-1). (pred noise). 
                2. prediction x_start from x_(t-1). (x_start). 
            
            if objective == 'pred_x0': 
                1. input x_t and Unet to x_0. (pred x_0). 
                2. prediction x_t from x_0. (pred x_t). 
            
            if objective == 'pred_v':
                1. input x_t and Unet to v (x_(t-1)). (pred v). 
                2. prediction x_start from v. (x_start). 
                3. prediction x_t from x_0 (pred x_t). 
            
            ModelPrediction(pred_noise [B, C, H, W], pred_x_start [B, C, H, W])
        '''
        # model_output = prediction of x_(t-1). 
        model_output = self.model(x, t, x_self_cond)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            # x_start is x_0. 
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            # pred_noise = x_(t-1). 
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_self_cond = None, clip_denoised = True):
        '''
        predict the mean and variance of P.

        Arguments:
            x (tensor): [B, C, H, W]. input noise in 't' sequence. (x_t). 
            t (int): time sequence. 
            x_self_cond (bool): choose to input the self_condition. 
            clip_denoised (bool): choose to denoising. 
        
        Returns:
            model_mean (tensor): [B, 1, 1, 1]. parameter of x_t mean. 
            posterior_variance (tensor): [B, 1, 1, 1]. parameter of x_t variance. 
            posterior_log_variance (tensor): [B, 1, 1, 1]. parameter of x_t log_variance. 
            x_start (tensor): [B, C, H, W]. prediction of x_0. 
        '''
        preds = self.model_predictions(x, t, x_self_cond)
        # x_start = prediction of x_0. 
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.inference_mode()
    def p_sample(self, x, t: int, x_self_cond = None):
        '''
        put x and t and predict(output) the t-1 states. 
        
        Arguments:
            x (tensor): [B, C, H, W]. input noise in 't' sequence. (x_t). 
            t (int): time sequence. 
            x_self_cond (bool): choose to input the self_condition. 
        
        Returns:
            pred_img (tensor): [B, C, H, W]. prediction of image. (x_(t-1)). 
            x_start (tensor): [B, C, H, W]. prediction of x_0. 
        '''
        b, *_, device = *x.shape, self.device
        # batched_times = [B]. fill the same 't' with all batches. 
        batched_times = torch.full((b,), t, device = device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = True)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.inference_mode()
    def p_sample_loop(self, shape, return_all_timesteps = False):
        '''
        P sampling for num_timesteps -> 0. denoising process. 
        Imgs are denoising images with [img(t=timesteps) -> img(t=0)]. 

        Arguments:
            shape (tensor): [B, C, H, W]. 
            return_all_timesteps (bool): choose return all images, or juct last image. 

        Returns:
            ret (tensor/[tensor for timesteps]): [B, C, H, W]. / [[B, C, H, W] ... [B, C, H, W]]. 
        '''
        batch, device = shape[0], self.device

        # random gaussian noise. 
        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            # if not self_condition, x_start will gather with p_sample. 
            self_cond = x_start if self.self_condition else None
            # img_t, t, self_cond_t has input. 
            # x_start is the prediction of x_0. 
            img, x_start = self.p_sample(img, t, self_cond)
            # img_(t-1) has prediced. 
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def ddim_sample(self, shape, return_all_timesteps = False):
        '''
        P sampling for num_timesteps -> 0. denoising process. 
        Imgs are denoising images with [img(t=T-1) -> img(t=0)]. 

        Arguments:
            shape (tensor): [B, C, H, W]. 
            return_all_timesteps (bool): choose return all images, or juct last image. 

        Returns:
            ret (tensor/[tensor for T-1]): [B, C, H, W]. / [[B, C, H, W] ... [B, C, H, W]]. 
        '''
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = True, rederive_pred_noise = True)

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def sample(self, batch_size = 16, return_all_timesteps = False):
        '''
        sampling for denoising from random noise to real image. 

        Arguments:
            batch_size (int): batch size. 
            return_all_timesteps (bool): choose return all images, or juct last image. 
        
        Returns:
            ret (tensor/[tensor for T-1]): [B, C, H, W]. / [[B, C, H, W] ... [B, C, H, W]]. 
        '''
        image_size, channels = self.image_size, self.channels
        # self.sampling_timesteps < timesteps, then ddim_sample. 
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, image_size, image_size), return_all_timesteps = return_all_timesteps)

    @torch.inference_mode()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        '''
        the function for interpolate. 

        Arguments:
            x1 (tensor): [B, C, H, W]. 
            x2 (tensor): [B, C, H, W]. 
        
        '''
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    @autocast(enabled = False)
    def q_sample(self, x_start, t, noise = None):
        '''
        noising sampling with random gaussian sampling. 

        Arguments:
            t (tensor): [B]. 
            x_start (tensor): [B, C, H, W]. 
            noise (tensor): [B, C, H, W]. 
        
        Returns:
            _ (tensor): [B, C, H, W]. 
        '''
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, noise = None, offset_noise_strength = None):
        '''
        calculate p_losses for time is random time simpling 'q_sampling'. 

        Arguments:
            x_start (tensor): [B, C, H, W]. the input image. it will start with 't'. 
            t (tensor): [B]. 
        '''
        b, c, h, w = x_start.shape

        noise = default(noise, lambda: torch.randn_like(x_start))

        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise

        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)

        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device = self.device)
            noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

        # noise sample
        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly
        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(x, t, x_self_cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return self.p_losses(img, t, *args, **kwargs)