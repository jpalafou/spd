import numpy as np
import sd_boundary as bc

def ader_string(dim):
    if dim==3:
        return "zyxkji"
    elif dim==2:
        return "yxji"
    else:
        return "xi"
    
def ader_arrays(self: "SD_Simulator"):
    """
    Allocate arrays to be used in the ADER time integration
    """
    nader=self.nader
    nvar=self.nvar
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
    if self.Y:
        self.dm.ML_fp_y = self.array_RS(dim="y",ader=True)
        self.dm.MR_fp_y = self.array_RS(dim="y",ader=True)
    if self.Z:
        self.dm.ML_fp_z = self.array_RS(dim="z",ader=True)
        self.dm.MR_fp_z = self.array_RS(dim="z",ader=True)
        
    #Arrays to store and impose boundary conditions
    self.dm.BC_fp_x = self.array_BC(dim="x",ader=True)
    self.dm.BC_fp_y = self.array_BC(dim="y",ader=True)
    self.dm.BC_fp_z = self.array_BC(dim="z",ader=True)

def ader_predictor(self: "SD_Simulator",prims=False) -> None:
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
        solve_faces(self,M,ader_iter)
            
        if ader_iter < self.m:
            # 2c) Compute new iteration value.
            # Axes labels:
            #   u: conservative variables
            #   n: ADER substeps, next
            #   p: ADER substeps, prev
            #   r, s: sol pts
            #   x: sol pts on untouched dimension
            #   f: flux pts
            #   b, c: cells
            #   z: cells on untouched dimension
            
            #Let's store dUdt first
            s = ader_string(self.ndim)
            self.dm.U_ader_sp[...] += np.einsum(f"np,up{s}->un{s}",self.dm.invader,
                                                 self.compute_sp_from_dfp_x()+
                                                 self.compute_sp_from_dfp_y()+
                                                 self.compute_sp_from_dfp_z())*self.dm.dt
            #Update
            # U_new = U_old - dUdt
            self.dm.U_ader_sp[...] = self.dm.U_sp[:,na] - self.dm.U_ader_sp
                    
        
def ader_update(self: "SD_Simulator"):
    na = self.dm.xp.newaxis
    # dUdt = (dFxdx +dFydy + S)dt 
    s = ader_string(self.ndim)
    dUdt = (np.einsum(f"t,ut{s}->u{s}",self.dm.w_tp,
                            self.compute_sp_from_dfp_x()+
                            self.compute_sp_from_dfp_y()+
                            self.compute_sp_from_dfp_z())*self.dm.dt)
        
    # U_new = U_old - dUdt
    self.dm.U_sp[...] -= dUdt
    
    # Compute primitive variables at solution points from updated solution    
    self.dm.W_sp =  self.compute_primitives(self.dm.U_sp)

def solve_faces(self: "SD_Simulator", M, ader_iter, prims=False)->None:
    na=np.newaxis
    # Interpolate M(U or W) to flux points
    # Then compute fluxes at flux points
    vx = self._vx_
    vy = self._vy_
    vz = self._vz_
    self.dm.M_ader_fp_x[...] = self.compute_fp_from_sp(M,"x",ader=True)
    self.compute_fluxes(self.dm.F_ader_fp_x, self.dm.M_ader_fp_x,vx,vy,vz,prims)
    if self.Y:
        self.dm.M_ader_fp_y[...] = self.compute_fp_from_sp(M,"y",ader=True)
        v1,v2 = ((vz,vx),(vx,vz)) [self.Z]
        self.compute_fluxes(self.dm.F_ader_fp_y, self.dm.M_ader_fp_y,vy,v1,v2,prims)
    if self.Z:
        self.dm.M_ader_fp_z[...] = self.compute_fp_from_sp(M,"z",ader=True)
        self.compute_fluxes(self.dm.F_ader_fp_z, self.dm.M_ader_fp_z,vz,vx,vy,prims)

    bc.store_BC(self,self.dm.BC_fp_x,self.dm.M_ader_fp_x,"x")
    bc.store_interfaces(self,self.dm.M_ader_fp_x,"x")
    #Here would go the BC comms between different domains
    bc.apply_BC(self,"x")
    return
