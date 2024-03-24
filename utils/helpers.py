import torch.nn.functional as F

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
        val / d
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
    '''
    return F.normalize(t, dim = -1)