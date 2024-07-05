from typing import Callable,Tuple,Union
import sys
import numpy as np
import cupy as cp
from itertools import repeat
from collections import defaultdict
from data_management import CupyLocation
from data_management import GPUDataManager
from sd_simulator import SD_Simulator
import riemann_solver as rs
import muscl

from slicing import cut

class FV_Simulator(SD_Simulator):
    def __init__(
        self,
        riemann_solver_fv: Callable = rs.llf,
        slope_limiter = "minmod",
        *args,
        **kwargs):
        super().__init__(*args, **kwargs)
        self.riemann_solver_fv = riemann_solver_fv
        self.slope_limiter = slope_limiter

    def array_FV(self,n,dim=None,ngh=0)->np.ndarray:
        shape = [self.nvar] 
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
    
    def fv_arrays(self):
        self.dm.M_fv = self.array_FV(self.p+1,ngh=self.Nghc)
        self.dm.F_faces_x = self.array_FV(self.p+1,dim="x")
        self.dm.MR_faces_x = self.array_FV(self.p+1,dim="x")
        self.dm.ML_faces_x = self.array_FV(self.p+1,dim="x")
        if self.Y:
            self.dm.F_faces_y =  self.array_FV(self.p+1,dim="y")
            self.dm.MR_faces_y = self.array_FV(self.p+1,dim="y")
            self.dm.ML_faces_y = self.array_FV(self.p+1,dim="y")
        if self.Z:
            self.dm.F_faces_z =  self.array_FV(self.p+1,dim="z")
            self.dm.MR_faces_z = self.array_FV(self.p+1,dim="z")
            self.dm.ML_faces_z = self.array_FV(self.p+1,dim="z")
        self.dm.BC_fv_x =  self.array_FV_BC(dim="x")
        self.dm.BC_fv_y =  self.array_FV_BC(dim="y")
        self.dm.BC_fv_z =  self.array_FV_BC(dim="z")

    def create_dicts_fv(self):
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

    def compute_fv_fluxes(self):
        return muscl.compute_second_order_fluxes(self)

    def crop_fv(self,start,end,dim):
        ngh=self.Nghc
        return (Ellipsis,)+(slice(ngh,-ngh),)*(self.ndim-1-dim)+(slice(start,end),)+(slice(ngh,-ngh),)*dim
