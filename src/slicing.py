from typing import Tuple
from itertools import repeat

def cut(start: int,
        end: int,
        shift: int)->Tuple:
    """
    Returns a tuple to be used when slicing multidimensional arrays
    """
    return (Ellipsis,)+(slice(start,end),)+(slice(None),)*(shift)

def indices(i: int,
            dim: int)->Tuple:
    """
    Returns a tuple to be used when slicing multidimensional arrays
    """
    return (Ellipsis, i)+ tuple(repeat(slice(None),dim))

def indices2(i: int,
            ndim: int,
            dim: int):
    """
    Returns a tuple to be used when slicing multidimensional arrays
    """
    return indices(i,ndim-1) + (i,) + tuple(repeat(slice(None),dim))

def crop_fv(start,end,dim,ndim,ngh)->Tuple:
    if ngh==None:
        return (Ellipsis,)+(slice(ngh,ngh),)*(ndim-1-dim)+(slice(start,end),)+(slice(ngh,ngh),)*dim
    else:
        return (Ellipsis,)+(slice(ngh,-ngh),)*(ndim-1-dim)+(slice(start,end),)+(slice(ngh,-ngh),)*dim
