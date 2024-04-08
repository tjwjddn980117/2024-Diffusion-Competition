import math

def exists(x):
    '''
    the function that check the parameter is exist. 

    Inputs:
        x ( ): input. 
    
    Outputs:
        Outputs:
        if x exists >> return x. 
        if x not exists >> return None. 
    '''
    return x is not None

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
    return d() if callable(d) else d

def cast_tuple(t, length = 1):
    '''
    check the 't' is the type of 'tuple'. 
    if the 't' wasn't tuple, return the tuple with t for length. 

    for example, t is 'a' and lenght is 3, then return ('a', 'a', 'a'). 

    Inputs:
        t ( ): some instance you want to repeat. 
        lenght (int): repeat times. 
    
    Returns:
        _ (tuple): repeated tuple. 
    '''
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

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

def identity(t, *args, **kwargs):
    '''
    For alter function. the fuction call it self. 
    
    Inputs:
        t ( ): input. 
        and the other parameters. 
    
    Outputs:
        t ( ): input -> output. 
    '''
    return t

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