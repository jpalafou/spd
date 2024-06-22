import numpy as np
from typing import Tuple
from itertools import repeat
from sd_simulator import SD_Simulator

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
   
def store_interfaces(self: SD_Simulator,
                     M: np.ndarray,
                     dim: str) -> None:
    """
    Stores the values of flux points at the extremes of elements(0,-1)
    These arrays are then used to solve the Riemann problem
    """
    shift=self.ndim+self.dims2[dim]-1
    axis = -(self.dims2[dim]+1)
    self.MR_fp[dim][cut(None,-1,shift)] = M[indices( 0,self.dims2[dim])]
    self.ML_fp[dim][cut(1 ,None,shift)] = M[indices(-1,self.dims2[dim])]

def apply_interfaces(self: SD_Simulator,
                     M: np.ndarray,
                     dim: str):
    """
    Applies the values of flux points at the extremes of elements(0,-1)
    This is done after the Riemann problem at element interfaces has been
    solved. 
    """
    shift=self.ndim+self.dims2[dim]-1
    M[indices( 0,self.dims2[dim])] = self.MR_fp[dim][cut(None,-1,shift)]
    M[indices(-1,self.dims2[dim])] = self.ML_fp[dim][cut(1, None,shift)]

def store_BC(self: SD_Simulator,
             BC_array: np.ndarray,
             M: np.ndarray,
             dim: str) -> None:
    """
    Stores the solution at flux points for the extremes of the domain
    These boundary arrays can then be communicated between domains
    """    
    na=np.newaxis
    idim = self.dims2[dim]
    if self.BC[dim] == "periodic":
        BC_array[0] = M[indices2(-1,self.ndim,idim)]
        BC_array[1] = M[indices2( 0,self.ndim,idim)]
    elif self.BC[dim] == "reflective":
        BC_array[0] = M[indices2( 0,self.ndim,idim)]
        BC_array[1] = M[indices2(-1,self.ndim,idim)]
        BC_array[:,self.dims2[dim]] = -BC_array[:,self.dims2[dim]]
    elif self.BC[dim] == "gradfree":
        BC_array[0] = M[indices2( 0,self.ndim,idim)]
        BC_array[1] = M[indices2(-1,self.ndim,idim)]
    else:
        raise("Undetermined boundary type")
                         
def apply_BC(self: SD_Simulator,
             dim: str) -> None:
    """
    Fills up the missing first column of M_L
    and the missing last column of M_R
    """
    shift=self.ndim+self.dims2[dim]-1
    self.ML_fp[dim][indices( 0,shift)] = self.BC_fp[dim][0]
    self.MR_fp[dim][indices(-1,shift)] = self.BC_fp[dim][1]