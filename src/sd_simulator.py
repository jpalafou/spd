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
from initial_conditions_3d import sine_wave

class SD_Simulator:
    def __init__(
        self,
        init_fct: Callable = sine_wave,
        p: int =  1, 
        m: int = -1,
        Nx: int = 32,
        Ny: int = 32,
        Nz: int = 32,
        Nghe: int = 1,
        xlim: Tuple = (0,1),
        ylim: Tuple = (0,1),
        zlim: Tuple = (0,1),
        X: bool = True,
        Y: bool = True,
        Z: bool = True,
        use_cupy: bool = True,
    ):
        self.init_fct = init_fct
        if m==-1:
            #By default m=p
            m=p
        self.p = p #Space order
        self.m = m #Time  order
        self.Nx = ((1,Nx) [X]) 
        self.Ny = ((1,Ny) [Y]) 
        self.Nz = ((1,Nz) [Z])
        self.Nghe = Nghe #Number of ghost element layers
        self.X = X
        self.Y = Y
        self.Z = Z

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
        self.nvar = nvar

        self.x, self.w = gauss_legendre_quadrature(0.0, 1.0, p)

        self.x_sp = solution_points(0.0, 1.0, p)
        self.y_sp = (np.ones(1)/2,self.x_sp) [Y]
        self.z_sp = (np.ones(1)/2,self.x_sp) [Z]
        self.x_fp = flux_points(0.0, 1.0, p)
        self.y_fp = (np.ones(1)/2,self.x_fp) [Y]
        self.z_fp = (np.ones(1)/2,self.x_fp) [Z]

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
        Nx = self.Nx+2*self.Nghe*X
        Ny = self.Ny+2*self.Nghe*Y
        Nz = self.Nz+2*self.Nghe*Z
  
        px = (1,p+2) [X]
        py = (1,p+2) [Y]
        pz = (1,p+2) [Z]
        
        self.mesh_cv = np.ndarray((3, Nz, Ny, Nx, pz, py, px))
        self.mesh_cv[0] = xlim[0]+(np.arange(Nx)[na,na,:,na,na,na] + self.x_fp[na,na,na,na,na,:])*(self.xlen+2*self.Nghe*self.dx)/Nx-self.dx
        self.mesh_cv[1] = ylim[0]+(np.arange(Ny)[na,:,na,na,na,na] + self.y_fp[na,na,na,na,:,na])*(self.ylen+2*self.Nghe*self.dy)/Ny-self.dy
        self.mesh_cv[2] = zlim[0]+(np.arange(Nz)[:,na,na,na,na,na] + self.z_fp[na,na,na,:,na,na])*(self.zlen+2*self.Nghe*self.dz)/Nz-self.dz
        
        X_sp = xlim[0]+(np.arange(self.Nx)[:,na] + self.x_sp[na,:])*(self.xlen)/(self.Nx)
        Y_sp = ylim[0]+(np.arange(self.Ny)[:,na] + self.y_sp[na,:])*(self.ylen)/(self.Ny)
        Z_sp = zlim[0]+(np.arange(self.Nz)[:,na] + self.z_sp[na,:])*(self.zlen)/(self.Nz)
        
        self.dm.X_sp = X_sp.reshape(self.Nx,(1,p+1) [X])
        self.dm.Y_sp = Y_sp.reshape(self.Ny,(1,p+1) [Y])
        self.dm.Z_sp = Z_sp.reshape(self.Nz,(1,p+1) [Z])

        self.post_init()

    def domain_size(self):
        Nx = self.Nx*(self.nx)
        Ny = self.Ny*(self.ny)
        Nz = self.Ny*(self.nz)
        return Nx,Ny,Nz

    def regular_faces(self):
        Nx,Ny,Nz = self.domain_size(self)
        x=np.linspace(0,self.xlen,Nx+1)
        y=np.linspace(0,self.ylen,Ny+1)
        z=np.linspace(0,self.zlen,Nz+1)
        return x,y,z

    def regular_centers(self):
        Nx,Ny,Nz = self.domain_size(self)
        x=np.linspace(0,self.xlen,Nx)
        y=np.linspace(0,self.ylen,Ny)
        z=np.linspace(0,self.zlen,Nz)
        return x,y,z
    
    def transpose_to_fv(self,M):
        #nvar,Nz,Ny,Nx,nz,ny,nx
        #nvar,Nznz,Nyny,Nxnx
        assert M.ndim == 7
        return np.transpose(M,(0,1,4,2,5,3,6)).reshape(M.shape[0],M.shape[1]*M.shape[4],M.shape[2]*M.shape[5],M.shape[3]*M.shape[6])   

    def array(self,px,py,pz,ngh=0) -> np.ndarray:
        return np.ndarray((
            self.nvar,
            self.Nz+2*ngh*self.Z,
            self.Ny+2*ngh*self.Y,
            self.Nx+2*ngh*self.X,
            px,
            py,
            pz))
        
    def array_sp(self,**kwargs):
        p=self.p
        return self.array(
            (1,p+1) [self.X],
            (1,p+1) [self.Y],
            (1,p+1) [self.Z],
            **kwargs)

    def array_fp(self,dims="xyz",**kwargs):
        p=self.p
        return self.array(
            (1,p+1+("x" in dims)) [self.X],
            (1,p+1+("y" in dims)) [self.Y],
            (1,p+1+("z" in dims)) [self.Z],
            **kwargs)

    def post_init(self) -> None:
        na = np.newaxis
        nvar = self.nvar
        ngh = self.Nghe
        X = self.X
        Y = self.Y
        Z = self.Z
        # This arrays contain Nghe layers of ghost elements
        W_gh = self.array_sp(ngh=ngh)
        for var in range(nvar):
            W_gh[var] = quadrature_mean(self.mesh_cv, self.init_fct, self.dimension, var)

        self.W_init_cv = self.crop(W_gh)

    def crop_x(self,M):
        ngh = self.Nghe
        if self.X:
            return M[:,:,:,ngh:-ngh,...]
        else:
            return M
        
    def crop_y(self,M):
        ngh = self.Nghe
        if self.X:
            return M[:,:,ngh:-ngh,...]
        else:
            return M
        
    def crop_z(self,M):
        ngh = self.Nghe
        if self.Z:
            return M[:,ngh:-ngh,...]
        else:
            return M

    def crop(self,M):
        return(self.crop_z(self.crop_y(self.crop_x(M))))
                    

        