from typing import Callable,Tuple,Union
import sys
import numpy as np
from collections import defaultdict
from sd_simulator import SD_Simulator
from fv_simulator import FV_Simulator
from data_management import CupyLocation
from polynomials import gauss_legendre_quadrature
from polynomials import ader_matrix
import sd_boundary as bc
import riemann_solver as rs
from trouble_detection import detect_troubles
from timeit import default_timer as timer
from slicing import cut, indices, indices2

class SDADER_Simulator(SD_Simulator,FV_Simulator):
    def __init__(self,
                 update = "SD",
                 FB = False,
                 tolerance = 1e-5,
                 PAD=True,
                 min_rho = 1e-10,
                 max_rho = 1e10,
                 min_P = 1e-10,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.update = update
        self.FB = FB
        self.tolerance = tolerance
        self.PAD = True
        self.min_rho = min_rho
        self.max_rho = max_rho
        self.min_P = min_P

        # ADER matrix.
        self.dm.x_tp, self.dm.w_tp = gauss_legendre_quadrature(0.0, 1.0, self.m + 1)
        ader = ader_matrix(self.dm.x_tp, self.dm.w_tp, 1.0)
        self.dm.invader = np.linalg.inv(ader)
        self.dm.invader = np.einsum("p,np->np",self.dm.w_tp,self.dm.invader)
        #number of time slices
        self.nader = self.m+1

        self.ader_arrays()
        self.compute_positions()
        if update=="FV":
            self.fv_arrays()
            if FB:
                self.fb_arrays()

    def compute_positions(self):
        na = np.newaxis
        X_sp = self.xlim[0]+(np.arange(self.Nx)[:,na] + self.x_sp[na,:])*(self.xlen)/(self.Nx)
        Y_sp = self.ylim[0]+(np.arange(self.Ny)[:,na] + self.y_sp[na,:])*(self.ylen)/(self.Ny)
        Z_sp = self.zlim[0]+(np.arange(self.Nz)[:,na] + self.z_sp[na,:])*(self.zlen)/(self.Nz)
        
        self.dm.X_sp = X_sp.reshape(self.Nx,self.nx)
        self.dm.Y_sp = Y_sp.reshape(self.Ny,self.ny)
        self.dm.Z_sp = Z_sp.reshape(self.Nz,self.nz)
        # 1-D array storing the position of interfaces
        self.dm.X_fp = np.ndarray((self.Nx * self.nx + self.nghx*2+1))
        self.dm.Y_fp = np.ndarray((self.Ny * self.ny + self.nghy*2+1))
        self.dm.Z_fp = np.ndarray((self.Nz * self.nz + self.nghz*2+1))
        self.faces = defaultdict(list)
        self.faces["x"] = self.dm.X_fp
        self.faces["y"] = self.dm.Y_fp
        self.faces["z"] = self.dm.Z_fp
        for dim in self.dims2:
            ngh = self.ngh[dim]
            self.faces[dim][ngh :-ngh] = (
            self.len[dim]/self.N[dim]*np.hstack(
            (np.arange(self.N[dim]).repeat(self.n[dim]) + 
             np.tile(self.fp[dim][:-1], self.N[dim]), self.N[dim]))
            )
            self.faces[dim][0:ngh] = -self.faces[dim][ngh+1:2*ngh+1][::-1]
            self.faces[dim][-ngh:] = self.faces[dim][-(ngh+1)] + self.faces[dim][ngh+1:2*ngh+1]
        
        self.dm.X_cv = 0.5*(self.dm.X_fp[1:]+self.dm.X_fp[:-1])
        self.dm.Y_cv = 0.5*(self.dm.Y_fp[1:]+self.dm.Y_fp[:-1])
        self.dm.Z_cv = 0.5*(self.dm.Z_fp[1:]+self.dm.Z_fp[:-1])
        self.centers = defaultdict(list)
        self.centers["x"] = self.dm.X_cv
        self.centers["y"] = self.dm.Y_cv
        self.centers["z"] = self.dm.Z_cv

        self.dm.dx_fp = (self.dm.X_fp[1:]-self.dm.X_fp[:-1])[self.shape(0)]
        self.dm.dx_cv = (self.dm.X_cv[1:]-self.dm.X_cv[:-1])[self.shape(0)]
        self.dm.dy_fp = (self.dm.Y_fp[1:]-self.dm.Y_fp[:-1])[self.shape(1)]
        self.dm.dy_cv = (self.dm.Y_cv[1:]-self.dm.Y_cv[:-1])[self.shape(1)]
        self.dm.dz_fp = (self.dm.Z_fp[1:]-self.dm.Z_fp[:-1])[self.shape(2)]
        self.dm.dz_cv = (self.dm.Z_cv[1:]-self.dm.Z_cv[:-1])[self.shape(2)]

        self.h_fp = defaultdict(list)
        self.h_cv = defaultdict(list)
        self.h_fp["x"] = self.dm.dx_fp
        self.h_cv["x"] = self.dm.dx_cv
        self.h_fp["y"] = self.dm.dy_fp
        self.h_cv["y"] = self.dm.dy_cv
        self.h_fp["z"] = self.dm.dz_fp
        self.h_cv["z"] = self.dm.dz_cv

    def ader_arrays(self):
        """
        Allocate arrays to be used in the ADER time integration
        """
        self.dm.U_ader_sp = self.array_sp(ader=True)
        for dim in self.dims2:
            #Conservative/Primitive varibles at flux points
            self.dm.__setattr__(f"M_ader_fp_{dim}",self.array_fp(dims=dim,ader=True))
            #Conservative fluxes at flux points
            self.dm.__setattr__(f"F_ader_fp_{dim}",self.array_fp(dims=dim,ader=True))
            #Arrays to Solve Riemann problem at the interface between elements
            self.dm.__setattr__(f"ML_fp_{dim}",self.array_RS(dim=dim,ader=True))
            self.dm.__setattr__(f"MR_fp_{dim}",self.array_RS(dim=dim,ader=True))
            #Arrays to communicate boundary values
            self.dm.__setattr__(f"BC_fp_{dim}",self.array_BC(dim=dim,ader=True))

    def fb_arrays(self):
        """
        Allocate arrays to be used in trouble detection
        """
        self.dm.troubles  = self.array_FV(self.p+1,1)
        for dim in self.dims2:
            #Conservative/Primitive varibles at flux points
            self.dm.__setattr__(f"affected_faces_{dim}",self.array_FV(self.p+1,1,dim=dim))

    def create_dicts(self):
        """
        Creates dictonaries for the arrays used in the ADER time
        integration. It enables writting generic functions for all
        dimensions.
        """
        names = ["M_ader_fp","F_ader_fp","MR_fp","ML_fp","BC_fp"]
        for name in names:
            self.__setattr__(name,{})
            for dim in self.dims2:
                self.__getattribute__(name)[dim] = self.dm.__getattribute__(f"{name}_{dim}")

        if self.update=="FV":
            self.create_dicts_fv()

    def ader_string(self)->str:
        """
        Returns a string to be used in the
        einsum performed to compute the ADER update.
        The string length depends on the dimensions
        """
        if self.ndim==3:
            return "zyxkji"
        elif self.ndim==2:
            return "yxji"
        else:
            return "xi"

    def ader_dudt(self):
        dUdt = self.compute_sp_from_dfp_x()
        if self.Y:
            dUdt += self.compute_sp_from_dfp_y()
        if self.Z:
            dUdt += self.compute_sp_from_dfp_z()
        return dUdt

    def ader_predictor(self,prims: bool = False) -> None:
        na = self.dm.xp.newaxis
        # W -> primivite variables
        # U -> conservative variables
        # Structure of arrays:
        #   U_sp: (nvar,Nz,Ny,Nx,pz,py,px)
        #   U_ader_sp: (nader,nvar,Nz,Ny,Nx,pz,py,px)

        # 1) Initialize u_ader_sp to u_sp, at all ADER time substeps.
        self.dm.U_ader_sp[...] = self.dm.U_sp[:,na, ...]

        # 2) ADER scheme (Picard iteration).
        # nader: number of time slices
        # m+1: order and number of iterations
        for ader_iter in range(self.m + 1):
            if prims:
                # Primitive variables
                M = self.compute_primitives(self.dm.U_ader_sp)   
            else:
                # Otherwise conservative variables
                M = self.dm.U_ader_sp
            # Once M hosts the correct set of variables,
            # we can interpolate to faces, and solve    
            self.solve_faces(M,ader_iter,prims=prims)

            if ader_iter < self.m:
                # 2c) Compute new iteration value.
                # Axes labels:
                #   u: conservative variables
                #   n: ADER substeps, next
                #   p: ADER substeps, prev

                #Let's store dUdt first
                s = self.ader_string()
                self.dm.U_ader_sp[...] = np.einsum(f"np,up{s}->un{s}",self.dm.invader,
                                                    self.ader_dudt())*self.dm.dt
                #Update
                # U_new = U_old - dUdt
                self.dm.U_ader_sp[...] = self.dm.U_sp[:,na] - self.dm.U_ader_sp

    def ader_update(self):
        na = self.dm.xp.newaxis
        # dUdt = (dFxdx +dFydy + S)dt 
        s = self.ader_string()
        dUdt = (np.einsum(f"t,ut{s}->u{s}",self.dm.w_tp,self.ader_dudt())*self.dm.dt)

        # U_new = U_old - dUdt
        self.dm.U_sp -= dUdt

        # Compute primitive variables at solution points from updated solution    
        self.dm.W_sp =  self.compute_primitives(self.dm.U_sp)

    def solve_faces(self, M, ader_iter, prims=False)->None:
        na=np.newaxis
        # Interpolate M(U or W) to flux points
        # Then compute fluxes at flux points
        for key in self.dims:
            dim = self.dims[key]
            vels = np.roll(self.vels,-key)
            self.M_ader_fp[dim][...] = self.compute_fp_from_sp(M,dim,ader=True)
            self.compute_fluxes(self.F_ader_fp[dim], self.M_ader_fp[dim],vels,prims)
            bc.store_interfaces(self,self.M_ader_fp[dim],dim)

            bc.store_BC(self,self.BC_fp[dim],self.M_ader_fp[dim],dim)
            #Here would go the BC comms between different domains
            bc.apply_BC(self,dim)

            self.riemann_solver_sd(self.ML_fp[dim], self.MR_fp[dim], vels, self._p_, self.gamma, self.min_c2, prims)
            bc.apply_interfaces(self,self.F_ader_fp[dim],dim)

    ####################
    ## Finite volume
    ####################
    def switch_to_finite_volume(self):
        #Change to Finite Volume scheme
        self.dm.U_cv[...] = self.compute_cv_from_sp(self.dm.U_sp)
        self.dm.W_cv[...] = self.compute_primitives(self.dm.U_cv)

        self.dm.U_cv = self.transpose_to_fv(self.dm.U_cv)
        self.dm.W_cv = self.transpose_to_fv(self.dm.W_cv)
        
        for dim in self.dims2:
            self.F_ader_fp[dim][...] = self.integrate_faces(self.F_ader_fp[dim],dim)

    def switch_to_high_order(self):
        #Change back to High-Order scheme
        self.dm.U_cv = self.transpose_to_sd(self.dm.U_cv)
        self.dm.W_cv = self.transpose_to_sd(self.dm.W_cv)
        self.dm.U_sp[...] = self.compute_sp_from_cv(self.dm.U_cv)
        self.dm.W_sp[...] = self.compute_primitives(self.dm.U_sp)

    def store_high_order_fluxes(self,i_ader):
        ndim=self.ndim
        dims  = [(0,1,2),(0,1,3,2,4),(0,1,4,2,5,3,6)]
        dims2 = [(0),(0,1,2),(0,1,3,2,4)]
        shapes = [[self.nvar,self.Nx*self.nx],
                  [self.nvar,self.Ny*self.ny,self.Nx*self.nx],
                  [self.nvar,self.Nz*self.nz,self.Ny*self.ny,self.Nx*self.nx]]
        for dim in self.dims2:
            shift=self.dims2[dim]
            shape=shapes[ndim-1].copy()
            self.F_faces[dim][cut(None,-1,shift)] = np.transpose(
                self.F_ader_fp[dim][:,i_ader][cut(None,-1,shift)],dims[ndim-1]
                ).reshape(shape)
            shape.pop(ndim-shift)
            self.F_faces[dim][indices(-1,shift)] = np.transpose(
                self.F_ader_fp[dim][:,i_ader][indices2(-1,ndim,shift)],dims2[ndim-1]).reshape(shape)
    
    def correct_fluxes(self):
        for dim in self.dims2:
            theta = self.dm.__getattribute__(f"affected_faces_{dim}")
            self.F_faces[dim] = theta*self.MR_faces[dim] + (1-theta)*self.F_faces[dim]

    def fv_update(self):
        self.switch_to_finite_volume()
        self.dm.U_new[...] = self.dm.U_cv
        for i_ader in range(self.nader):
            dt = self.dm.dt*self.dm.w_tp[i_ader]
            self.store_high_order_fluxes(i_ader)
            if self.FB:
                detect_troubles(self)
                self.compute_fv_fluxes()
                self.correct_fluxes()
            self.fv_apply_fluxes(dt)
            self.dm.U_cv[...] = self.dm.U_new
        self.switch_to_high_order()

    ####################
    ## Update functions
    ####################
    def perform_update(self) -> bool:
        self.n_step += 1
        self.ader_predictor()
        if self.update=="SD":
            self.ader_update()
        else:
            self.fv_update()
        self.time += self.dm.dt
        return True

    def init_sim(self):
        self.dm.switch_to(CupyLocation.device)
        self.create_dicts()
        self.execution_time = -timer() 

    def end_sim(self):
        self.dm.switch_to(CupyLocation.host)
        self.execution_time += timer() 
        self.create_dicts()
        self.dm.U_cv[...] = self.compute_cv_from_sp(self.dm.U_sp)
        self.dm.W_cv[...] = self.compute_cv_from_sp(self.dm.W_sp)

    def perform_iterations(self, n_step: int) -> None:
        self.init_sim()
        for i in range(n_step):
            self.compute_dt()
            self.perform_update()
        self.end_sim()
     
    def perform_time_evolution(self, t_end: float, nsteps=0) -> None:
        self.init_sim()
        while(self.time < t_end):
            if not self.n_step % 100:
                print(f"Time step #{self.n_step} (t = {self.time})",end="\r")
            self.compute_dt()   
            if(self.time + self.dm.dt >= t_end):
                self.dm.dt = t_end-self.time
            if(self.dm.dt < 1E-14):
                print(f"dt={self.dm.dt}")
                break
            self.status = self.perform_update()
        self.end_sim()          