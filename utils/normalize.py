import math
import torch
from functools import wraps

# normalization functions
def normalize_to_neg_one_to_one(img):
    '''
    If the image is normalized between 0.0 and 1.0, 
    this function converts all pixel values to a range between -1 and 1. 

    Inputs:
        img (tensor): [B, C, H, W] (0.0 ~ 1.0). 
    
    Outputs:
        img (tensor): [B, C, H, W] (-1.0 ~ 1.0). 
    '''
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    '''
    If the input is normalized between -1.0 and 1.0, 
    this function converts all values to a range between 0 and 1. 
    
    Inputs:
        t (tensor): (-1.0 ~ 1.0). 

    Outputs:
        t (tensor): (0.0 ~ 1.0). 
    '''
    return (t + 1) * 0.5

# diffusion helpers

def right_pad_dims_to(x, t):
    '''
    Compare the dimension of x and t. 
    if the dimension of t is bigger/same with x, return t. 
    else, reshape t. 

    ex)
    x.shape = [3,5,4,4], t.shape = [2,2]. 
    then, t.shape [2,2] -> [2,2,1,1]. 

    Inputs:
        x (tensor): x tensor. 
        t (tensor): the dimension should bigger or same than x. 
    
    Outputs:
        if t.ndim >= x.ndim, return t. 
        if t.ndim < x.ndim, return t (with resizing). 
    '''
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

# logsnr schedules and shifting / interpolating decorators
# only cosine for now

def log(t, eps = 1e-20):
    '''
    It is used to prevent this, 
    since the entry of a zero or negative number into the function can return an infinite or NaN value. 

    Inputs:
        t (tensor): input tensor. 
        eps (float): prevent to make input as 0. 
    
    Outputs:
        torch.log(t.clamp(min = eps)). it's safe tensor. 
    '''
    return torch.log(t.clamp(min = eps))

def logsnr_schedule_cosine(t, logsnr_min = -15, logsnr_max = 15):
    '''
    This function computes log SNR using cosine schedules for a given time t. 
    These kinds of schedules can be used in deep learning for training rate scheduling or time-dependent adjustments of other parameters. 

    Inputs:
        t (tensor): [ _ ]. input tensor. The tensor should 'zero dimension'. ex) torch.tensor(0.28). 
        logsnr_min (int): the minimum number of logsnr. 
        logsnr_max (int): the maximum number of logsnr. 
    
    Outputs:
        [ _ ] (tensor): -2 * log(torch.tan(t_min + t * (t_max - t_min))). it's safe tensor. 
    '''
    t_min = math.atan(math.exp(-0.5 * logsnr_max))
    t_max = math.atan(math.exp(-0.5 * logsnr_min))
    return -2 * log(torch.tan(t_min + t * (t_max - t_min)))

def logsnr_schedule_shifted(fn, image_d, noise_d):
    '''
    the function for shifted with time serise. 

    Inputs:
        fn (function): the function for logsnr. 
        imgae_d (int): dimension of image. 
        noise_d (float): noise image. 

    Outputs:
        [ _ ] (tensor): fn(*args, **kwargs) + shift. 
    '''
    shift = 2 * math.log(noise_d / image_d)
    @wraps(fn)
    def inner(*args, **kwargs):
        nonlocal shift
        return fn(*args, **kwargs) + shift
    return inner

def logsnr_schedule_interpolated(fn, image_d, noise_d_low, noise_d_high):
    '''
    the function for logsnr with detail??

    Inputs:
        fn (function): the function for logsnr.
        imgae_d (int): dimension of image. 
        noise_d_low (tensor): noise image with low time. 
        noise_d_high (tensor): noise image with high time. 
    
    Outputs:
        [ _ ] (tensor): t * logsnr_low_fn(t, *args, **kwargs) + (1 - t) * logsnr_high_fn(t, *args, **kwargs). 
    '''
    logsnr_low_fn = logsnr_schedule_shifted(fn, image_d, noise_d_low)
    logsnr_high_fn = logsnr_schedule_shifted(fn, image_d, noise_d_high)

    @wraps(fn)
    def inner(t, *args, **kwargs):
        nonlocal logsnr_low_fn
        nonlocal logsnr_high_fn
        return t * logsnr_low_fn(t, *args, **kwargs) + (1 - t) * logsnr_high_fn(t, *args, **kwargs)

    return inner