from typing import Union

import numpy as np
import cupy as cp
from sd_simulator import SD_Simulator

def compute_cs2(p,rho,gamma,min_c2):
        c2 = gamma*p/rho
        #np.maximum propagates NaNs, so we use np.where
        c2 = np.where(c2>min_c2,c2,min_c2)
        return c2

def compute_cs(p,rho,gamma,min_c2):
        return np.sqrt(compute_cs2(p,rho,gamma,min_c2))

def compute_dt(self: SD_Simulator) -> None:
    W = self.dm.W_cv
    c_s = compute_cs(W[self._p_],W[self._d_],self.gamma,self.min_c2)
    c_x = (0,np.abs(W[self._vx_])+c_s) [self.X]
    c_y = (0,np.abs(W[self._vy_])+c_s) [self.Y]
    c_z = (0,np.abs(W[self._vz_])+c_s) [self.Z]
    c_max = np.max(c_x+c_y+c_z)
    self.dm.dt = self.cfl_coeff*min(self.dx,min(self.dy,self.dz))/c_max/(self.p + 1)