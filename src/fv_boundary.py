import numpy as np
from typing import Tuple
from sd_simulator import SD_Simulator
from slicing import cut
from slicing import indices
from slicing import indices2

def store_BC(self: SD_Simulator,
             M: np.ndarray,
             dim: str) -> None:
    """
    Stores the solution of ngh layers in the active region
    """    
    na=np.newaxis
    idim = self.dims2[dim]
    ngh=self.Nghc
    if self.BC[dim] == "periodic":
        self.BC_fv[dim][0] = M[cut(-2*ngh,  -ngh,idim)]
        self.BC_fv[dim][1] = M[cut(   ngh, 2*ngh,idim)]
    else:
        raise("Undetermined boundary type")
                         
def apply_BC(self: SD_Simulator,
             dim: str) -> None:
    """
    Fills ghost cells in M_fv
    """
    ngh=self.Nghc
    idim = self.dims2[dim]
    shift=self.ndim+self.dims2[dim]-1
    self.dm.M_fv[cut(None, ngh,idim)] = self.BC_fv[dim][0]
    self.dm.M_fv[cut(-ngh,None,idim)] = self.BC_fv[dim][1]