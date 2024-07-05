from typing import Callable,Tuple,Union
import sys
import numpy as np
import cupy as cp
from itertools import repeat
from sd_simulator import SD_Simulator
import fv_boundary as bc

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

def compute_slopes(self: SD_Simulator, dU, dim):
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

def compute_second_order_fluxes(self: SD_Simulator, prims=True):
    ngh = self.Nghc
    self.dm.M_fv[...]  = 0
    #Copy W_cv to active region of M_fv
    self.dm.M_fv[(Ellipsis,)+tuple(repeat(slice(ngh,-ngh),self.ndim))] = self.dm.W_cv

    for dim in self.dims2:
        shift=self.dims2[dim]
        vels = np.roll(self.vels,-shift)
        h_cv = self.h_cv[dim]
        h_fp = self.h_fp[dim]
        bc.store_BC(self,self.dm.M_fv,dim)
        #Comms here
        bc.apply_BC(self,dim)
        dM = (self.dm.M_fv[cut(1,None,shift)] - self.dm.M_fv[cut(None,-1,shift)])/h_cv
        dMh = compute_slopes(self,dM,shift)    
        S = 0.5*dMh*h_fp[cut(1,-1,shift)] #Slope*h/2  

        #UR = U - SlopeC*h/2, UL = U + SlopeC*h/2
        self.MR_faces[dim][...] = self.dm.M_fv[self.crop_fv(ngh,-1,shift)] - S[self.crop_fv( 1,None,shift)]
        self.ML_faces[dim][...] = self.dm.M_fv[self.crop_fv(1,-ngh,shift)] + S[self.crop_fv(None,-1,shift)] 
        self.riemann_solver_fv(self.ML_faces[dim], self.MR_faces[dim], vels, self._p_, self.gamma, self.min_c2, prims)  
        self.F_faces[dim] = self.MR_faces[dim]