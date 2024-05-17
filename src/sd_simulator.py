from typing import Callable
from typing import Tuple
import sys
import numpy as np
import cupy as cp

from data_management import CupyLocation
from data_management import GPUDataManager
from polynomials import gauss_legendre_quadrature
from polynomials import solution_points
from polynomials import flux_points
from polynomials import lagrange_matrix
from polynomials import lagrangeprime_matrix
from polynomials import intfromsol_matrix
from polynomials import ader_matrix
from polynomials import quadrature_mean

class SD_Simulator:
    def __init__(
        self,
        p: int =  1, 
        m: int = -1,
        Nx: int = 32,
        Ny: int = 32,
        Nz: int = 32,
        xlim: Tuple = (0,1),
        ylim: Tuple = (0,1),
        zlim: Tuple = (0,1),
        X: bool = True,
        Y: bool = True,
        Z: bool = True,
        use_cupy: bool = True,
    ):
        if m==-1:
            #By default m=p
            m=p
        self.p = p #Space order
        self.m = m #Time  order
        self.Nx = ((1,Nx) [X]) 
        self.Ny = ((1,Ny) [Y]) 
        self.Nz = ((1,Nz) [Z])
        self.dm = GPUDataManager(use_cupy)
        
        self.dimension = X+Y+Z
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim
        self.time = 0.0
        
        self.xlen = xlim[1]-xlim[0]
        self.ylen = ylim[1]-ylim[0]
        self.zlen = zlim[1]-zlim[0]

        self.dx = self.xlen/self.Nx
        self.dy = self.ylen/self.Ny
        self.dz = self.zlen/self.Nz

        self.n_step = 0
        
        nvar=0
        self._d_  = nvar
        nvar+=1
        if X:
            self._vx_ = nvar
            nvar+=1
        if Y: 
            self._vy_ = nvar
            nvar+=1
        if Z: 
            self._vz_ = nvar
            nvar+=1
        self._p_  = nvar
        nvar+=1
        assert nvar == 2 + self.dimension

        self.x, self.w = gauss_legendre_quadrature(0.0, 1.0, p)

        self.x_sp = solution_points(0.0, 1.0, p)
        self.y_sp = (np.ones(1),self.x_sp) [Y]
        self.z_sp = (np.ones(1),self.x_sp) [Z]
        self.x_fp = flux_points(0.0, 1.0, p)
        self.y_fp = (np.ones(1),self.x_fp) [Y]
        self.z_fp = (np.ones(1),self.x_fp) [Z]

        # Lagrange matrices to perform interpolation between basis
        self.dm.sp_to_fp = lagrange_matrix(self.x_fp, self.x_sp)
        self.dm.fp_to_sp = lagrange_matrix(self.x_sp, self.x_fp)
        # Spatial derivative of the flux at sol pts from density at flux pts.
        self.dm.dfp_to_sp = lagrangeprime_matrix(self.x_sp, self.x_fp)
        # Mean values in control volumes from values at sol pts.
        self.dm.sp_to_cv = intfromsol_matrix(self.x_sp, self.x_fp)
        self.dm.fp_to_cv = intfromsol_matrix(self.x_fp, self.x_fp)
        self.dm.cv_to_sp = np.linalg.inv(self.dm.sp_to_cv)
        
        # ADER matrix.
        self.dm.x_tp, self.dm.w_tp = gauss_legendre_quadrature(0.0, 1.0, self.m + 1)
        ader = ader_matrix(self.dm.x_tp, self.dm.w_tp, 1.0)
        self.dm.invader = np.linalg.inv(ader)
        self.dm.invader = np.einsum("p,np->np",self.dm.w_tp,self.dm.invader)
        #number of time slices
        self.nader = self.m+1
        
        na =  np.newaxis
        Nx = Nx+2*X
        Ny = Ny+2*Y
        Nz = Nz+2*Z
        px = (1,p+2) [X]
        py = (1,p+2) [Y]
        pz = (1,p+2) [Z]
        
        self.mesh_cv = np.ndarray((3, Nz, Ny, Nx, pz, py, px))
        self.mesh_cv[0] = xlim[0]+(np.arange(Nx)[na,na,:,na,na,na] + self.x_fp[na,na,na,na,na,:])*(self.xlen+2*self.dx)/Nx-self.dx
        self.mesh_cv[1] = ylim[0]+(np.arange(Ny)[na,:,na,na,na,na] + self.y_fp[na,na,na,na,:,na])*(self.ylen+2*self.dy)/Ny-self.dy
        self.mesh_cv[2] = zlim[0]+(np.arange(Nz)[:,na,na,na,na,na] + self.z_fp[na,na,na,:,na,na])*(self.zlen+2*self.dz)/Nz-self.dz
        
        X_sp = xlim[0]+(np.arange(self.Nx)[:,na] + self.x_sp[na,:])*(self.xlen)/(self.Nx)
        Y_sp = ylim[0]+(np.arange(self.Ny)[:,na] + self.y_sp[na,:])*(self.ylen)/(self.Ny)
        Z_sp = zlim[0]+(np.arange(self.Nz)[:,na] + self.z_sp[na,:])*(self.zlen)/(self.Nz)
        
        self.dm.X_sp = X_sp.reshape(self.Nx,(1,p+1) [X])
        self.dm.Y_sp = Y_sp.reshape(self.Ny,(1,p+1) [Y])
        self.dm.Z_sp = Z_sp.reshape(self.Nz,(1,p+1) [Z])