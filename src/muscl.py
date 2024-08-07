from typing import Callable,Tuple,Union
import sys
import numpy as np
import cupy as cp
from itertools import repeat
from simulator import Simulator

from slicing import cut
from slicing import crop_fv

def minmod(SlopeL: np.ndarray,SlopeR: np.ndarray)->np.ndarray:
    """
    args: 
        SlopeL/R: Solution vector with Left/Right slopes
    returns:
        Slopes: Limited slopes
    """
    #First compute ratio between slopes SlopeR/SlopeL
    #Then limit the ratio to be lower than 1
    #Finally, limit the ratio to be positive and multiply
    #  by SlopeL to get the limited slope at the cell center
    #We use "where" instead of "maximum/minimum" as it doesn't
    # propagte the NaNs caused when SlopeL=0
    ratio = SlopeR/SlopeL
    ratio = np.where(ratio<1,ratio,1)
    return np.where(ratio>0,ratio,0)*SlopeL

def moncen(dU_L: np.ndarray,
           dU_R: np.ndarray,
           dx_L: np.ndarray,
           dx_R: np.ndarray,
           dx_M: np.ndarray)->np.ndarray:
    """
    args: 
        dU_L/R: Solution vector with Left/Right slopes
        dx_L/R: vector of cell sizes (distance between cell centers)
        dx_M:   vector of cell sizes (distance between flux points)
    returns:
        Slopes: Limited slopes
    """
    dU_C = (dx_L*dU_L + dx_R*dU_R)/(dx_L+dx_R)
    slope = np.minimum(np.abs(2*dU_L*dx_L/dx_M),np.abs(2*dU_R*dx_R/dx_M))
    slope = np.sign(dU_C)*np.minimum(slope,np.abs(dU_C))
    return np.where(dU_L*dU_R>=0,slope,0)     

def compute_slopes(M: np.ndarray,
                   h_cv: np.ndarray,
                   h_fp: np.ndarray,
                   idim: int,
                   slope_limiter: str,
                   gradient: bool=False):
    """
    args: 
        M:      Solution vector (conservatives/primitives)
        h_cv:   vector of cell sizes (distance between cell centers)
        h_fp:   vector of cell sizes (distance between flux points)
        idim:   index of dimension
    returns:
        S:      Slopes of M
        dMh:    Gradient of M
    """
    dM = (M[cut(1,None,idim)] - M[cut(None,-1,idim)])/h_cv
    if slope_limiter == "minmod":
        dMh = minmod(dM[cut(None,-1,idim)],dM[cut(1,None,idim)])

    elif slope_limiter == "moncen":
        dMh = moncen(dM[cut(None,-1,idim)],
                  dM[cut(1,None,idim)],
                  h_cv[cut(None,-1,idim)],
                  h_cv[cut(1,None,idim)],
                  h_fp[cut(1,-1,idim)])
    if gradient:
        return dMh
    else:
        #Slope*h/2
        return 0.5*dMh*h_fp[cut(1,-1,idim)] 
    
def MUSCL_fluxes(self: Simulator,
                 F: dict,
                 dt: float,
                 prims=True)->None:
    """
    args: 
        self:   Simulator object
        dt:     timestep
        prims:  Wheter values at faces are primitives
                or conservatives
    overwrites:
        self.MR_faces[dim]:   Values interpolated to the right
        self.ML_faces[dim]:   Values interpolated to the left
        self.F_faces_FB[dim]: Fluxes given by the Riemann solver
    """
    for dim in self.dims2:
        shift=self.dims2[dim]
        
        S = self.compute_slopes(self.dm.M_fv,shift)    
        
        self.MR_faces[dim][...] = self.interpolate_R(self.dm.M_fv,S,shift)
        self.ML_faces[dim][...] = self.interpolate_L(self.dm.M_fv,S,shift)
        self.solve_riemann_problem(dim,F[dim],prims)
    
def compute_prediction(W: np.ndarray,
                       dWs: np.ndarray,
                       dtW: np.ndarray,
                       vels: np.array,
                       ndim: int,
                       gamma: float,
                       _d_: int,
                       _p_: int,
                       WB: bool,
                       isothermal: bool,
                       )->None:
    """
    args: 
        W:      Solution vector with primitive variables
        dWs:    Solution vector with slopes 
    overwrites:
        dtW:  Solution vector with predictions 
    """
    dtW[...] = 0
    for idim in range(ndim):
        vel = vels[idim]
        dW = dWs[idim]
        dtW[_d_] -= (W[vel]*dW[_d_] +       W[_d_]*dW[vel])
        dtW[_p_] -= (W[vel]*dW[_p_] + gamma*W[_p_]*dW[vel])
        dtW[vel] -= (W[vel]*dW[vel]+ dW[_p_]/W[_d_])
        if WB:
            dW = dWs[idim+ndim]
            dtW[_d_] -= (W[vel]*dW[_d_]) 
            dtW[_p_] -= (W[vel]*dW[_p_])
        for vel2 in vels[1:]:
            dtW[vel2] -= W[vel]*dW[vel2]
    dtW[_p_] *= 0 if isothermal else 0

def MUSCL_Hancock_fluxes(self: Simulator,
                         F: dict,
                         dt: float,
                         prims=True)->None:
    """
    args: 
        self:   Simulator object
        dt:     timestep
        prims:  Wheter values at faces are primitives
                or conservatives
    overwrites:
        self.MR_faces[dim]:   Values interpolated to the right
        self.ML_faces[dim]:   Values interpolated to the left
        self.F_faces_FB[dim]: Fluxes given by the Riemann solver
    """
    dMhs={}
    S={}
    crop = lambda start,end,idim : crop_fv(start,end,idim,self.ndim,1)
    for dim in self.dims2:
        idim=self.dims2[dim]
        dMh = self.compute_slopes(self.dm.M_fv,idim,gradient=True)
        S[idim] = 0.5*dMh*self.h_fp[dim][cut(1,-1,idim)]
        dMhs[idim] = dMh[crop(None,None,idim)]
        if self.WB:
            dMhs[idim+self.ndim] = self.compute_slopes(self.dm.M_eq_fv,idim,gradient=True)[crop(None,None,idim)]
    if self.WB:
        self.dm.M_fv += self.dm.M_eq_fv                    
    self.compute_prediction(self.dm.M_fv[crop(1,-1,0)],dMhs)
    if self.WB:
        if self.potential:
            drho = ((self.dm.M_fv[0]-self.dm.M_eq_fv[0])/self.dm.M_fv[0])[crop(1,-1,0)]
            for vel in self.vels:
                self.dm.dtM[vel][crop(1,-1,0)] += drho[crop(1,-1,0)]*self.dm.grad_phi_fv[vel-1]
        #We move back to the perturbation
        self.dm.M_fv -= self.dm.M_eq_fv
    self.dm.M_fv[crop(1,-1,0)] += 0.5*self.dm.dtM*dt
    
    for dim in self.dims2:
        idim=self.dims2[dim]
        self.MR_faces[dim][...] = self.interpolate_R(self.dm.M_fv,S[idim],idim)
        self.ML_faces[dim][...] = self.interpolate_L(self.dm.M_fv,S[idim],idim)
        self.solve_riemann_problem(dim,F[dim],prims)
    
def compute_viscosity(self: Simulator,
                      F: dict,)->None:
    """
    args: 
        self:   Simulator object
    overwrites:
        self.F_faces_FB[dim]: Adds viscous fluxes
    """
    ngh=self.Nghc
    dW={}
    for dim in self.dims2:
        idim = self.dims2[dim]
        #Make a choice of values (here left)
        M = self.ML_faces[dim]
        h = self.h_fp[dim][cut(ngh,-ngh,idim)]
        #Compute gradient in dim at cell centers
        dW[idim] = (M[cut( 1,None,idim)]-M[cut(None,-1,idim)])/h
    dW_f = {}
    for dim in self.dims2:
        shift = self.dims2[dim]
        vels = np.roll(self.vels,-shift)
        #Interpolate gradients(all) to faces at dim
        for idim in self.dims:
            self.fill_active_region(dW[idim])
            if self.BC[dim][0] == "periodic":
                self.fv_Boundaries(self.dm.M_fv,dim)    
            S = self.compute_slopes(self.dm.M_fv,shift)
            #Counter the previous choice of values (now right)
            dW_f[idim] = self.interpolate_R(self.dm.M_fv,S,shift)
        #Add viscous flux
        F[dim][...] -= self.compute_viscous_fluxes(self.ML_faces[dim],dW_f,vels,prims=True)
       
def compute_second_order_fluxes(self: Simulator,
                                F: dict,
                                dt: float,
                                **kwargs)->None:
    """
    args: 
        self:   Simulator object
        F:      Dictionary with references to Flux arrays
        dt:     timestep
    overwrites:
        self.F_faces_FB[dim]: Second order fluxes
    """
    if self.predictor:
        MUSCL_Hancock_fluxes(self,F,dt,**kwargs)
    else:
        MUSCL_fluxes(self,F,dt,**kwargs)
    if self.viscosity:
        compute_viscosity(self,F)
        