from typing import Union

import numpy as np
import cupy as cp

def compute_cs2(p,rho,gamma,min_c2):
        c2 = gamma*p/rho
        #np.maximum propagates NaNs, so we use np.where
        c2 = np.where(c2>min_c2,c2,min_c2)
        return c2

def compute_cs(p,rho,gamma,min_c2):
        return np.sqrt(compute_cs2(p,rho,gamma,min_c2))

def compute_fluxes(self,M,vels,prims=True)->np.ndarray:
    F = M.copy()
    self.compute_fluxes(F,M,vels,prims=prims)
    return F

def compute_dt(self: "SD_Simulator") -> None:
    W = self.dm.W_cv
    c_s = compute_cs(W[self._p_],W[self._d_],self.gamma,self.min_c2)
    c = np.abs(W[self._vx_])+c_s
    if self.Y:
        c += np.abs(W[self._vy_])+c_s
    if self.Z:
        c += np.abs(W[self._vz_])+c_s
    c_max = np.max(c)
    self.dm.dt = self.cfl_coeff*min(self.dx,min(self.dy,self.dz))/c_max/(self.p + 1)