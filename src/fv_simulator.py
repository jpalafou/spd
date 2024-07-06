from typing import Callable,Tuple,Union
import sys
import numpy as np
import cupy as cp
from itertools import repeat
from collections import defaultdict
from data_management import CupyLocation
from data_management import GPUDataManager
from simulator import Simulator
import riemann_solver as rs
import muscl

from slicing import cut

class FV_Simulator(Simulator):
    def __init__(
        self,
        riemann_solver_fv: Callable = rs.llf,
        slope_limiter = "minmod",
        *args,
        **kwargs):
        super().__init__(*args, **kwargs)
        self.riemann_solver_fv = riemann_solver_fv
        self.slope_limiter = slope_limiter

    def array_FV(self,n,nvar,dim=None,ngh=0)->np.ndarray:
        shape = [nvar] 
        if self.Z:
            shape += [self.Nz*n+(dim=="z")+2*ngh]
        if self.Y:
            shape += [self.Ny*n+(dim=="y")+2*ngh]
        shape += [self.Nx*n+(dim=="x")+2*ngh]
        return np.ndarray(shape)
    
    def array_FV_BC(self,dim="x")->np.ndarray:
        shape = [2,self.nvar]
        ngh=self.Nghc
        if self.Z:
            shape += [self.Nz*self.nz+2*ngh] if dim!="z" else [ngh]
        if self.Y:
            shape += [self.Ny*self.ny+2*ngh] if dim!="y" else [ngh]
        shape += [self.Nx*self.nx+2*ngh] if dim!="x" else [ngh]
        return np.ndarray(shape)
    
    def fv_arrays(self)->None:
        self.dm.M_fv  = self.array_FV(self.p+1,self.nvar,ngh=self.Nghc)
        self.dm.U_new = self.array_FV(self.p+1,self.nvar)
        for dim in self.dims2:
            #Conservative/Primitive varibles at flux points
            self.dm.__setattr__(f"F_faces_{dim}",self.array_FV(self.p+1,self.nvar,dim=dim))
            self.dm.__setattr__(f"MR_faces_{dim}",self.array_FV(self.p+1,self.nvar,dim=dim))
            self.dm.__setattr__(f"ML_faces_{dim}",self.array_FV(self.p+1,self.nvar,dim=dim))
            self.dm.__setattr__(f"BC_fv_{dim}",self.array_FV_BC(dim=dim))

    def create_dicts_fv(self)->None:
        self.F_faces = defaultdict(list)
        self.MR_faces = defaultdict(list)
        self.ML_faces = defaultdict(list)
        self.BC_fv = defaultdict(list)
        
        for dim in self.dims2:
            self.faces[dim] = self.dm.__getattribute__(f"{dim.upper()}_fp")
            self.centers[dim] = self.dm.__getattribute__(f"{dim.upper()}_cv")
            self.h_fp[dim] = self.dm.__getattribute__(f"d{dim}_fp")
            self.h_cv[dim] = self.dm.__getattribute__(f"d{dim}_cv")
            self.F_faces[dim] = self.dm.__getattribute__(f"F_faces_{dim}")
            self.MR_faces[dim] = self.dm.__getattribute__(f"MR_faces_{dim}")
            self.ML_faces[dim] = self.dm.__getattribute__(f"ML_faces_{dim}")
            self.BC_fv[dim] = self.dm.__getattribute__(f"BC_fv_{dim}")

    def compute_fv_fluxes(self)->None:
        return muscl.compute_second_order_fluxes(self)

    def crop_fv(self,start,end,dim)->Tuple:
        ngh=self.Nghc
        return (Ellipsis,)+(slice(ngh,-ngh),)*(self.ndim-1-dim)+(slice(start,end),)+(slice(ngh,-ngh),)*dim

    def fill_active_region(self, M):
        ngh=self.Nghc
        self.dm.M_fv[(Ellipsis,)+tuple(repeat(slice(ngh,-ngh),self.ndim))] = M

    def fv_apply_fluxes(self,dt):
        dUdt = self.dm.U_cv.copy()*0
        for dim in self.dims2:
            ndim = self.ndim
            ngh = self.ngh[dim]
            shift=self.dims2[dim]
            dx = self.faces[dim][ngh+1:-ngh] - self.faces[dim][ngh:-(ngh+1)]
            dx = dx[(None,)*(ndim-shift)+(slice(None),)+(None,)*(shift)]
            dUdt += (self.F_faces[dim][cut(1,None,shift)]
                             -self.F_faces[dim][cut(None,-1,shift)])/dx
        
        self.dm.U_new -= dUdt*dt

    def fv_update(self):
        self.dm.U_new[...] = self.dm.U_cv
        self.compute_fv_fluxes()
        self.fv_apply_fluxes(self.dm.dt)
        self.dm.U_cv[...] = self.dm.U_new

    def fv_store_BC(self,
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
                         
    def fv_apply_BC(self,
                 dim: str) -> None:
        """
        Fills ghost cells in M_fv
        """
        ngh=self.Nghc
        idim = self.dims2[dim]
        shift=self.ndim+self.dims2[dim]-1
        self.dm.M_fv[cut(None, ngh,idim)] = self.BC_fv[dim][0]
        self.dm.M_fv[cut(-ngh,None,idim)] = self.BC_fv[dim][1]

    def fv_Boundaries(self,
                    M: np.ndarray,
                    dim: str):
        self.fv_store_BC(M,dim)
        #Comms here
        self.fv_apply_BC(dim)