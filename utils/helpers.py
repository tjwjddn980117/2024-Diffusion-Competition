from torch import nn
import math
import torch.nn.functional as F
import torchvision

# helpers
def exists(val):
    '''
    the function that check the parameter is exist. 

    Inputs:
        val ( ): input. 
    
    Outputs:
        Outputs:
        if val exists >> return val. 
        if val not exists >> return None. 
    '''
    return val is not None

def identity(t):
    '''
    For alter function. the fuction call it self. 
    
    Inputs:
        t ( ): input. 
    
    Outputs:
        t ( ): input -> output. 
    '''
    return t

def is_lambda(f):
    '''
    check the function is lambda. 
    
    Inputs:
        f ( ): function. 
    
    Outputs:
        
    '''
    return callable(f) and f.__name__ == "<lambda>" 

def default(val, d):
    '''
    choose the default function. 

    Inputs:
        val ( ): exist. 
        d ( ): alter. 
    
    Outputs:
        if val exists >> return val. 
        if val not exists >> return d. 
    '''
    if exists(val):
        return val
    return d() if is_lambda(d) else d

def cast_tuple(t, l = 1):
    '''
    check the 't' is the type of 'tuple'. 
    if the 't' wasn't tuple, return the tuple with t for length. 

    for example, t is 'a' and lenght is 3, then return ('a', 'a', 'a'). 
    '''
    return ((t,) * l) if not isinstance(t, tuple) else t

def append_dims(t, dims):
    '''
    append dims with (1, ). 
    example, torch.Size([2, 3]) >> append_dims(t, 2) >> torch.Size([2, 3, 1, 1]). 

    Inputs:
        t (tensor): Some tensor. 
        dims (int): appending dims. 
    
    Outputs:
        _ (tensor): tensor with appending dims. 
    '''
    shape = t.shape
    return t.reshape(*shape, *((1,) * dims))

def l2norm(t):
    '''
    normalized last layer. 

    Inputs:
        t (tensor) input tensor. 
    
    Outputs:
        return F.normalize(t, dim = -1).
    '''
    return F.normalize(t, dim = -1)

def num_to_groups(num, divisor):
    '''
    Example:
        Input: num = 10, divisor = 3.
        Ouputs: [3, 3, 3, 1].

    Inputs:
        num (int): total nums.
        divisor (int): each num for group.
    
    Outputs:
        arr (arr): devided arr.
    '''
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def cycle(dl):
    '''
    Inputs:
        dl (list): [data1, data2, data3, ...]
    
    Returns:
        data1, data2,... you have to next(cycle(dl)) for work on. 
    '''
    while True:
        for data in dl:
            yield data

def divisible_by(numer, denom):
    '''
    check the number can devide. 

    Inputs:
        numer (int): big number. 
        denom (int): devide number. 
    
    Returns:
        (numer % denom) == 0. 
    '''
    return (numer % denom) == 0

def has_int_squareroot(num):
    '''
    check the 'number' can 'root'. 

    Inputs:
        num (int): check it could be. 
    
    Returns:
        _ (bool): check it could be. 
    '''
    root = math.sqrt(num)
    return root.is_integer()

def convert_image_to_fn(img_type, image):
    '''
    if image type is same with img_type, return image.
    if image type is different with img_type, convert image type with 'img_type'.

    Inputs:
        img_type (List): possible types of images. 
        image (Image): the input images. 
    
    Returns:
        image (Image): the image with convert image type. 
    '''
    if image.mode != img_type:
        return image.convert(img_type)
    return image


def init_img_transform(x):
    '''
    the function for pre-processing. 
    '''
    # resize image to 256x256. 
    resize_transform = torchvision.transforms.Resize((256, 256))
    x = resize_transform(x)
    # normalize the image. 
    normalize_transform = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    x = normalize_transform(x)
    return x

def final_img_itransform(x):
    '''
    the function for post-processing. 
    '''
    # upsampling the image to original size. 
    upsample_transform = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    x = upsample_transform(x)
    return x