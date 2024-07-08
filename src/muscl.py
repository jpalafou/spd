from typing import Callable,Tuple,Union
import sys
import numpy as np
import cupy as cp
from itertools import repeat
from simulator import Simulator

from slicing import cut

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

def compute_slopes(self: Simulator, dU, dim):
        if self.slope_limiter == "minmod":
            return minmod(dU[cut(None,-1,dim)],dU[cut(1,None,dim)])

        elif self.slope_limiter == "moncen":
            h_cv = self.h_cv[self.dims[dim]]
            h_fp = self.h_fp[self.dims[dim]]
            return moncen(dU[cut(None,-1,dim)],
                      dU[cut(1,None,dim)],
                      h_cv[cut(None,-1,dim)],
                      h_cv[cut(1,None,dim)],
                      h_fp[cut(1,-1,dim)])
                
def MUSCL_fluxes(self: Simulator, dt: float, prims=True):
    ngh = self.Nghc
    self.dm.M_fv[...]  = 0
    #Copy W_cv to active region of M_fv
    self.fill_active_region(self.compute_primitives(self.dm.U_cv))

    for dim in self.dims2:
        shift=self.dims2[dim]
        vels = np.roll(self.vels,-shift)
        h_cv = self.h_cv[dim]
        h_fp = self.h_fp[dim]

        self.fv_Boundaries(self.dm.M_fv,dim)
        
        dM = (self.dm.M_fv[cut(1,None,shift)] - self.dm.M_fv[cut(None,-1,shift)])/h_cv
        dMh = compute_slopes(self,dM,shift)    
        S = 0.5*dMh*h_fp[cut(1,-1,shift)] #Slope*h/2  
        #UR = U - SlopeC*h/2, UL = U + SlopeC*h/2
        self.MR_faces[dim][...] = self.dm.M_fv[self.crop_fv(ngh,-1,shift,ngh)] - S[self.crop_fv( 1,None,shift,ngh)]
        self.ML_faces[dim][...] = self.dm.M_fv[self.crop_fv(1,-ngh,shift,ngh)] + S[self.crop_fv(None,-1,shift,ngh)] 
        self.riemann_solver_fv(self.ML_faces[dim], self.MR_faces[dim], vels, self._p_, self.gamma, self.min_c2, prims)
        
def compute_prediction(self: Simulator, U, dUs):
    gamma=self.gamma
    _d_ = self._d_
    _p_ = self._p_
    self.dm.dtM[...] = 0
    for idim in self.dims:
        vel = self.vels[idim]
        dU = dUs[idim]
        self.dm.dtM[_d_] -= (U[vel]*dU[_d_] +       U[_d_]*dU[vel])
        self.dm.dtM[_p_] -= (U[vel]*dU[_p_] + gamma*U[_p_]*dU[vel])
        self.dm.dtM[vel] -= (U[vel]*dU[vel]+ dU[_p_]/U[_d_])
        for vel2 in self.vels[1:]:
            self.dm.dtM[vel2] -= U[vel]*dU[vel2]   

def MUSCL_Hancock_fluxes(self: Simulator, dt: float, prims=True):
    ngh = self.Nghc
    self.dMh={}
    self.S={}
    self.dm.M_fv[...]  = 0
    #Copy W_cv to active region of M_fv
    self.fill_active_region(self.compute_primitives(self.dm.U_cv))
    for dim in self.dims2:
        idim=self.dims2[dim]
        h_cv = self.h_cv[dim]
        h_fp = self.h_fp[dim]

        self.fv_Boundaries(self.dm.M_fv,dim)
        
        dM = (self.dm.M_fv[cut(1,None,idim)] - self.dm.M_fv[cut(None,-1,idim)])/h_cv
        dMh = compute_slopes(self,dM,idim)
        #Slope*h/2    
        S = 0.5*dMh*h_fp[cut(1,-1,idim)]
        self.dMh[idim] = dMh[self.crop_fv(None,None,idim,1)]
        self.S[idim] = S 
    compute_prediction(self,self.dm.M_fv[self.crop_fv(1,-1,0,1)],self.dMh)
    self.dm.M_fv[self.crop_fv(1,-1,0,1)] += 0.5*self.dm.dtM*dt
    for dim in self.dims2:
        idim=self.dims2[dim]
        vels = np.roll(self.vels,-idim)
        #UR = U - SlopeC*h/2, UL = U + SlopeC*h/2
        self.MR_faces[dim][...] = self.dm.M_fv[self.crop_fv(ngh,-1,idim,ngh)] - self.S[idim][self.crop_fv( 1,None,idim,ngh)]
        self.ML_faces[dim][...] = self.dm.M_fv[self.crop_fv(1,-ngh,idim,ngh)] + self.S[idim][self.crop_fv(None,-1,idim,ngh)] 
        self.riemann_solver_fv(self.ML_faces[dim], self.MR_faces[dim], vels, self._p_, self.gamma, self.min_c2, prims)

def compute_second_order_fluxes(self: Simulator,
                                dt: float,
                                **kwargs):
    if self.predictor:
        MUSCL_Hancock_fluxes(self,dt,**kwargs)
    else:
        MUSCL_fluxes(self,dt,**kwargs)
        