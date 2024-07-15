from typing import Callable,Tuple,Union
import sys
import numpy as np
import cupy as cp
from collections import defaultdict

from data_management import CupyLocation
from data_management import GPUDataManager
from initial_conditions_3d import sine_wave
import hydro

class Simulator:
    def __init__(
        self,
        init_fct: Callable = sine_wave,
        p: int =  1, 
        m: int = -1,
        Nx: int = 32,
        Ny: int = 32,
        Nz: int = 32,
        Nghe: int = 1,
        Nghc: int = 2,
        xlim: Tuple = (0,1),
        ylim: Tuple = (0,1),
        zlim: Tuple = (0,1),
        ndim: int = 3,
        gamma: float = 1.4,
        beta: float = 2./3,
        nu: float = 1e-4,
        cfl_coeff: float = 0.8,
        min_c2: float = 1E-10,
        viscosity: bool = False,
        potential: bool = False,
        use_cupy: bool = True,
        BC: Tuple = ("periodic","periodic","periodic"),
    ):
        self.init_fct = init_fct
        if m==-1:
            #By default m=p
            m=p
        self.p = p #Space order
        self.m = m #Time  order
        self.Nx = Nx
        self.Y = ndim>1
        self.Z = ndim>2
        self.Ny = ((1,Ny) [self.Y]) 
        self.Nz = ((1,Nz) [self.Z])
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim
        self.time = 0.0

        self.Nghe = Nghe #Number of ghost element layers
        self.Nghc = Nghc #Number of ghost cell layers
        self.ndim = ndim
        self.gamma=gamma
        self.beta=beta
        self.nu=nu
        self.cfl_coeff = cfl_coeff
        self.min_c2 = min_c2
        self.viscosity = viscosity
        self.potential = potential

        assert len(BC) >= ndim
        self.BC = defaultdict(list)
        self.dims = defaultdict(list)
        self.dims2 = defaultdict(list)
        
        dims = ["x","y","z"]
        for dim in range(ndim):
            self.dims[dim] = dims[dim]
            self.dims2[dims[dim]] = dim
            self.BC[dims[dim]] = BC[dim]
                
        self.dm = GPUDataManager(use_cupy)

        self.nghx = Nghc
        self.nghy = (0,Nghc) [self.Y]
        self.nghz = (0,Nghc) [self.Z]
        
        self.lim = defaultdict(list)
        self.len = defaultdict(list)
        self.h = defaultdict(list)
        self.N = defaultdict(list)
        self.ngh = defaultdict(list)
        self.h_min = 1E10
        for dim in self.dims2:
            self.lim[dim] = self.__getattribute__(f"{dim}lim") 
            self.__setattr__(f"{dim}len",self.lim[dim][1]-self.lim[dim][0])
            self.len[dim] = self.__getattribute__(f"{dim}len")
            self.N[dim] =  self.__getattribute__(f"N{dim}")
            self.__setattr__(f"d{dim}", self.len[dim]/self.N[dim])
            self.h[dim] = self.__getattribute__(f"d{dim}") 
            self.ngh[dim] = self.__getattribute__(f"ngh{dim}") 
            self.h_min = min(self.h_min,self.h[dim])
        self.n_step = 0
        
        nvar=0
        self._d_  = nvar
        nvar+=1
        self._vx_ = nvar
        nvar+=1
        if self.Y: 
            self._vy_ = nvar
            nvar+=1
        else:
            self._vy_ = -1
        if self.Z: 
            self._vz_ = nvar
            nvar+=1
        else:
            self._vz_ = -1
        self._p_  = nvar
        nvar+=1
        assert nvar == 2 + self.ndim
        self.nvar = nvar
        self.vels=np.array([self._vx_,self._vy_,self._vz_])[:self.ndim]

    def shape(self,idim):
        return (None,)*(self.ndim-idim)+(slice(None),)+(None,)*(idim)
    
    def compute_positions(self):
        pass
    
    def compute_mesh_cv(self) -> np.ndarray:
        pass
        
    def post_init(self) -> None:
        pass

    def domain_size(self):
        return [ self.N[dim]*self.n[dim] for dim in self.dims2]

    def regular_faces(self):
        N = self.N
        n = self.n
        lim = self.lim
        return [np.linspace(lim[dim][0],lim[dim][1],N[dim]*n[dim]+1) for dim in self.dims2]

    def regular_centers(self):
        N = self.N
        n = self.n
        lim = self.lim
        return [np.linspace(lim[dim][0],lim[dim][1],N[dim]*n[dim]) for dim in self.dims2]

    def crop(self,M,ngh=1)->np.ndarray:
        return M[(slice(None),)+(slice(ngh,-ngh),)*self.ndim+(Ellipsis,)]

    def compute_primitives(self,U,**kwargs)->np.ndarray:
        return hydro.compute_primitives(
                U,
                self.vels,
                self._p_,
                self.gamma,
                **kwargs)
                
    def compute_conservatives(self,W,**kwargs)->np.ndarray:
        return hydro.compute_conservatives(
                W,
                self.vels,
                self._p_,
                self.gamma,
                **kwargs)
    
    def compute_fluxes(self,F,M,vels,prims)->np.ndarray:
        assert len(vels)==self.ndim
        if prims:
            W = M
        else:
            W = self.compute_primitives(M)
        hydro.compute_fluxes(W,vels,self._p_,self.gamma,F=F)

    def compute_viscous_fluxes(self,M,dMs,vels,prims=False)->np.ndarray:
        assert len(vels)==self.ndim
        if prims:
            W = M
        else:
            W = self.compute_primitives(M)
        return hydro.compute_viscous_fluxes(W,vels,dMs,self._p_,self.nu,self.beta)

    def apply_potential(self,dUdt,U,grad_phi):
        _p_ = self._p_
        for idim in self.dims:
            vel = self.vels[idim]
            dUdt[vel,...] += U[  0]*grad_phi[idim]
            dUdt[_p_,...] += U[vel]*grad_phi[idim]
            
    def compute_dt(self) -> None:
        pass

    

