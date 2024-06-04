from typing import Callable,Tuple,Union
import sys
import numpy as np
from collections import defaultdict
from sd_simulator import SD_Simulator
from data_management import CupyLocation
import sd_boundary as bc
from initial_conditions import sine_wave
import riemann_solver as rs

class SDADER_Simulator(SD_Simulator):
    def __init__(
        self,
        init_fct: Callable = sine_wave(),
        p: int =  1, 
        m: int = -1,
        Nx: int = 32,
        Ny: int = 32,
        Nz: int = 32,
        Nghe: int = 1,
        xlim: Tuple = (0,1),
        ylim: Tuple = (0,1),
        zlim: Tuple = (0,1),
        ndim: int = 3,
        gamma: float = 1.4,
        cfl_coeff: float = 0.8,
        min_c2: float = 1E-10,
        use_cupy: bool = True,
        BC: Tuple = ("periodic","periodic","periodic"),
        riemann_solver_sd: Callable = rs.llf,):
        SD_Simulator.__init__(self)
        self.ader_arrays()

    def ader_arrays(self):
        """
        Allocate arrays to be used in the ADER time integration
        """
        self.dm.U_ader_sp = self.array_sp(ader=True)
        #Conservative/Primitive varibles at flux points
        #Conservative fluxes at flux points
        self.dm.M_ader_fp_x = self.array_fp(dims="x",ader=True)
        self.dm.F_ader_fp_x = self.array_fp(dims="x",ader=True)
        if self.Y:
            self.dm.M_ader_fp_y = self.array_fp(dims="y",ader=True)
            self.dm.F_ader_fp_y = self.array_fp(dims="y",ader=True)
        if self.Z:
            self.dm.M_ader_fp_z = self.array_fp(dims="z",ader=True)
            self.dm.F_ader_fp_z = self.array_fp(dims="z",ader=True)

        #Arrays to Solve Riemann problem at the interface between
        #elements
        self.dm.ML_fp_x = self.array_RS(dim="x",ader=True)
        self.dm.MR_fp_x = self.array_RS(dim="x",ader=True)
        self.dm.BC_fp_x = self.array_BC(dim="x",ader=True)
        if self.Y:
            self.dm.ML_fp_y = self.array_RS(dim="y",ader=True)
            self.dm.MR_fp_y = self.array_RS(dim="y",ader=True)
            self.dm.BC_fp_y = self.array_BC(dim="y",ader=True)
        if self.Z:
            self.dm.ML_fp_z = self.array_RS(dim="z",ader=True)
            self.dm.MR_fp_z = self.array_RS(dim="z",ader=True)
            self.dm.BC_fp_z = self.array_BC(dim="z",ader=True)

    def create_dicts(self):
        self.M_ader_fp = defaultdict(list)
        self.F_ader_fp = defaultdict(list)
        self.MR_fp = defaultdict(list)
        self.ML_fp = defaultdict(list)
        self.BC_fp = defaultdict(list)

        self.M_ader_fp["x"] = self.dm.M_ader_fp_x
        self.F_ader_fp["x"] = self.dm.F_ader_fp_x
        if self.Y:
            self.M_ader_fp["y"] = self.dm.M_ader_fp_y
            self.F_ader_fp["y"] = self.dm.F_ader_fp_y
        if self.Z:
            self.M_ader_fp["z"] = self.dm.M_ader_fp_z
            self.F_ader_fp["z"] = self.dm.F_ader_fp_z

        #Arrays to Solve Riemann problem at the interface between
        #elements
        self.MR_fp["x"] = self.dm.MR_fp_x
        self.ML_fp["x"] = self.dm.ML_fp_x
        self.BC_fp["x"] = self.dm.BC_fp_x
        if self.Y:
            self.MR_fp["y"] = self.dm.MR_fp_y
            self.ML_fp["y"] = self.dm.ML_fp_y
            self.BC_fp["y"] = self.dm.BC_fp_y
        if self.Z:
            self.MR_fp["z"] = self.dm.MR_fp_z
            self.ML_fp["z"] = self.dm.ML_fp_z
            self.BC_fp["z"] = self.dm.BC_fp_z    

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

    def perform_update(self) -> bool:
        self.n_step += 1
        na = self.dm.xp.newaxis
        self.ader_predictor()
        self.ader_update()
        self.time += self.dm.dt
        return True
    
    def perform_iterations(self, n_step: int) -> None:
        self.dm.switch_to(CupyLocation.device)
        self.create_dicts(self)
        for i in range(n_step):
            self.compute_dt()
            self.perform_update()
        self.dm.switch_to(CupyLocation.host)
        self.create_dicts(self)
        self.dm.U_cv[...] = self.compute_cv_from_sp(self.dm.U_sp)
        self.dm.W_cv[...] = self.compute_cv_from_sp(self.dm.W_sp)
     
    def perform_time_evolution(self, t_end: float, nsteps=0) -> None:
        self.dm.switch_to(CupyLocation.device)
        self.create_dicts()
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
        self.dm.switch_to(CupyLocation.host)
        self.create_dicts()
        self.dm.U_cv[...] = self.compute_cv_from_sp(self.dm.U_sp)
        self.dm.W_cv[...] = self.compute_cv_from_sp(self.dm.W_sp)