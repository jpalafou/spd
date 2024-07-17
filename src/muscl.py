from typing import Callable,Tuple,Union
import sys
import numpy as np
import cupy as cp
from itertools import repeat
from simulator import Simulator

from slicing import cut
from slicing import crop_fv

def minmod(SlopeL,SlopeR):
    #First compute ratio between slopes SlopeR/SlopeL
    #Then limit the ratio to be lower than 1
    #Finally, limit the ratio be positive and multiply by SlopeL to get the limited slope at the cell center
    #We use where instead of maximum/minimum as it doesn't propagte the NaNs caused when SlopeL=0
    ratio = SlopeR/SlopeL
    ratio = np.where(ratio<1,ratio,1)
    return np.where(ratio>0,ratio,0)*SlopeL

def moncen(dU_L,dU_R,dx_L,dx_R,dx_M):
    #Compute central slope
    dU_C = (dx_L*dU_L + dx_R*dU_R)/(dx_L+dx_R)
    slope = np.minimum(np.abs(2*dU_L*dx_L/dx_M),np.abs(2*dU_R*dx_R/dx_M))
    slope = np.sign(dU_C)*np.minimum(slope,np.abs(dU_C))
    return np.where(dU_L*dU_R>=0,slope,0)     

def compute_slopes(self: Simulator, M, dim, gradient=False):
    """
    args: 
        M: ndarray
        dim: int
    out:
        S: ndarray = Slopes of M
        dMh: ndarray = Gradient of M
    """
    h_cv = self.h_cv[self.dims[dim]]
    h_fp = self.h_fp[self.dims[dim]]
    dM = (M[cut(1,None,dim)] - M[cut(None,-1,dim)])/h_cv
    if self.slope_limiter == "minmod":
        dMh = minmod(dM[cut(None,-1,dim)],dM[cut(1,None,dim)])

    elif self.slope_limiter == "moncen":
        dMh = moncen(dM[cut(None,-1,dim)],
                  dM[cut(1,None,dim)],
                  h_cv[cut(None,-1,dim)],
                  h_cv[cut(1,None,dim)],
                  h_fp[cut(1,-1,dim)])
    if gradient:
        return dMh
    else:
        return 0.5*dMh*h_fp[cut(1,-1,dim)] #Slope*h/2

def interpolate_R(self: Simulator, M, S, idim):
    #UR = U - SlopeC*h/2
    ngh=self.Nghc
    crop = lambda start,end,idim : crop_fv(start,end,idim,self.ndim,ngh)
    return M[crop(ngh,-1,idim)] - S[crop( 1,None,idim)]

def interpolate_L(self: Simulator, M, S, idim):
    #UL = U + SlopeC*h/2
    ngh=self.Nghc
    crop = lambda start,end,idim : crop_fv(start,end,idim,self.ndim,ngh)
    return M[crop(1,-ngh,idim)] + S[crop(None,-1,idim)]        

def solve_riemann_problem(self: Simulator,dim: str, prims: bool):
    idim=self.dims2[dim]
    vels = np.roll(self.vels,-idim)
    if self.WB:
        #Move to solution at interfaces
        M_eq_faces = self.dm.__getattribute__(f"M_eq_faces_{dim}")
        self.MR_faces[dim][...] += M_eq_faces
        self.ML_faces[dim][...] += M_eq_faces
    self.F_faces_FB[dim] = self.riemann_solver_fv(self.ML_faces[dim], self.MR_faces[dim], vels, self._p_, self.gamma, self.min_c2, prims)
    if self.WB:
        #We compute the perturbation over the flux for conservative variables
        self.F_faces_FB[dim] -= self.dm.__getattribute__(f"F_eq_faces_{dim}")
    
                
def MUSCL_fluxes(self: Simulator, dt: float, prims=True):
    ngh = self.Nghc
    self.dm.M_fv[...]  = 0
    #Copy W_cv to active region of M_fv
    self.fill_active_region(self.dm.W_cv)
    for dim in self.dims2:
        self.fv_Boundaries(self.dm.M_fv,dim)
    for dim in self.dims2:
        shift=self.dims2[dim]
        vels = np.roll(self.vels,-shift)
        
        S = compute_slopes(self,self.dm.M_fv,shift)    
        
        self.MR_faces[dim][...] = interpolate_R(self,self.dm.M_fv,S,shift)
        self.ML_faces[dim][...] = interpolate_L(self,self.dm.M_fv,S,shift)
        solve_riemann_problem(self,dim,prims)
    
def compute_prediction(self: Simulator, W, dWs):
    gamma=self.gamma
    _d_ = self._d_
    _p_ = self._p_
    self.dm.dtM[...] = 0
    for idim in self.dims:
        vel = self.vels[idim]
        dW = dWs[idim]
        self.dm.dtM[_d_] -= (W[vel]*dW[_d_] +       W[_d_]*dW[vel])
        self.dm.dtM[_p_] -= (W[vel]*dW[_p_] + gamma*W[_p_]*dW[vel])
        self.dm.dtM[vel] -= (W[vel]*dW[vel]+ dW[_p_]/W[_d_])
        if self.WB:
            dW = dWs[idim+self.ndim]
            self.dm.dtM[_d_] -= (W[vel]*dW[_d_]) 
            self.dm.dtM[_p_] -= (W[vel]*dW[_p_])
        for vel2 in self.vels[1:]:
            self.dm.dtM[vel2] -= W[vel]*dW[vel2]   

def MUSCL_Hancock_fluxes(self: Simulator, dt: float, prims=True):
    ngh = self.Nghc
    dMhs={}
    S={}
    self.dm.M_fv[...]  = 0
    crop = lambda start,end,idim : crop_fv(start,end,idim,self.ndim,1)
    #Copy W_cv to active region of M_fv
    self.fill_active_region(self.dm.W_cv)
    for dim in self.dims2:
        self.fv_Boundaries(self.dm.M_fv,dim)

    for dim in self.dims2:
        idim=self.dims2[dim]
        dMh = compute_slopes(self,self.dm.M_fv,idim,gradient=True)
        S[idim] = 0.5*dMh*self.h_fp[dim][cut(1,-1,idim)]
        dMhs[idim] = dMh[crop(None,None,idim)]
        if self.WB:
            dMhs[idim+self.ndim] = compute_slopes(self,self.dm.M_eq_fv,idim,gradient=True)[crop(None,None,idim)]
    if self.WB:
        self.dm.M_fv += self.dm.M_eq_fv                    
    compute_prediction(self,self.dm.M_fv[crop(1,-1,0)],dMhs)
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
        vels = np.roll(self.vels,-idim)
        self.MR_faces[dim][...] = interpolate_R(self,self.dm.M_fv,S[idim],idim)
        self.ML_faces[dim][...] = interpolate_L(self,self.dm.M_fv,S[idim],idim)
        solve_riemann_problem(self,dim,prims)
    
def compute_viscosity(self: Simulator):
    ngh=self.Nghc
    dW={}
    #for dim in self.dims2:
    #    idim = self.dims2[dim]
    #    ML =  self.dm.M_fv[cut(None,-2,idim)]
    #    MR =  self.dm.M_fv[cut( 2,None,idim)]
    #    centers =  self.centers[dim]
    #    h = centers[cut( 2,None,idim)]-centers[cut(None,-2,idim)]
    #    dW[idim] = (MR-ML)/h
    for dim in self.dims2:
        idim = self.dims2[dim]
        #Make a choice of values (here left)
        M = self.ML_faces[dim]
        h = self.h_fp[dim][cut(ngh,-ngh,idim)]
        dW[idim] = (M[cut( 1,None,idim)]-M[cut(None,-1,idim)])/h
    dW_f = {}
    for dim in self.dims2:
        shift = self.dims2[dim]
        vels = np.roll(self.vels,-shift)
        for idim in self.dims:
            self.fill_active_region(dW[idim])
            if self.BC[dim] == "periodic":
                self.fv_Boundaries(self.dm.M_fv,dim)    
            S = compute_slopes(self,self.dm.M_fv,shift)
            #Counter the previous choice of values (now right)
            dW_f[idim] = interpolate_R(self,self.dm.M_fv,S,shift)
        #Add viscous flux
        self.F_faces_FB[dim][...] -= self.compute_viscous_fluxes(self.ML_faces[dim],dW_f,vels,prims=True)
       
def compute_second_order_fluxes(self: Simulator,
                                dt: float,
                                **kwargs):
    if self.predictor:
        MUSCL_Hancock_fluxes(self,dt,**kwargs)
    else:
        MUSCL_fluxes(self,dt,**kwargs)
    if self.viscosity:
        compute_viscosity(self)
        