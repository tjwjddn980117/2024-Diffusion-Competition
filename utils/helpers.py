from torch import nn
import torch.nn.functional as F
import torchvision

# helpers
def exists(val):
    '''
    the function that check the parameter is exist.

    Inputs:
        val ( ): input.
    
    Outputs:
        val ( ): True / False
    '''
    return val is not None

def identity(t):
    '''
    Reinfore the parammeter.
    
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
        val ( ):
        d ( ): 
    
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

    for example, t is 'a' and lenght is 3, then return ('a', 'a', 'a')
    '''
    return ((t,) * l) if not isinstance(t, tuple) else t

def append_dims(t, dims):
    '''
    append dims with (1, )
    example, torch.Size([2, 3]) >> append_dims(t, 2) >> torch.Size([2, 3, 1, 1])

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

def init_img_transform(x):
    '''
    the function for pre-processing. 
    '''
    # 이미지를 256x256 크기로 조정
    resize_transform = torchvision.transforms.Resize((256, 256))
    x = resize_transform(x)
    # 이미지 정규화
    normalize_transform = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    x = normalize_transform(x)
    return x

def final_img_itransform(x):
    '''
    the function for post-processing. 
    '''
    # 이미지를 원본 크기로 업샘플링
    upsample_transform = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    x = upsample_transform(x)
    return x