import torch
import math

# gaussian diffusion trainer class
def extract(a, t, x_shape):
    '''
    extracting a. the information is 't'. and resize for x_shape. 
    for example, 
        a = tensor([0.327, 0.445, 0.121, 0.311, 0.902]). \n
        t = tensor([1,0,3]). \n
        x_shape.shape = [3, 2]. \n

        then return should be [[0.445], [0.327], [0.311]]. (output.shape = [3, 1]). 

    Arguments:
        a (tensor): [timesteps]. the information of each timesteps. 
        t (tensor): [B]. time for each batch. 
        x_shape (tensor): [B, C, H, W]. 
    
    Returns:
        _ (tensor): [B, 1, 1, 1]. 
    '''
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper.

    Arguments:
        timesteps (int): full time steps. 
    
    Returns:
        _ (tensor): [timesteps]. 
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ

    Arguments:
        timesteps (int): full time steps. 
        s (float): 
    
    Returns:
        _ (tensor): [timesteps]. 
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training

    Arguments:
        timesteps (int): full time steps. 
        start (int): the start num. 
        end (int): the end num. 
        taus (int): 
        clamp_min (float): 
    
    Returns:
        _ (tensor): [timesteps]. 
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)