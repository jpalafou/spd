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

def compute_primitives(U,vels,_p_,gamma)->np.ndarray:
        W = U.copy()
        K = W[0].copy()*0
        for vel in vels:
            W[vel] = U[vel]/U[0]
            K += W[vel]**2
        K *= 0.5*U[0]
        W[_p_] = (gamma-1)*(U[_p_]-K)
        return W
                
def compute_conservatives(W,vels,_p_,gamma)->np.ndarray:
        U = W.copy()
        K = W[0].copy()*0
        for vel in vels:
            U[vel] = W[vel]*U[0]
            K += W[vel]**2
        K  *= 0.5*U[0]
        U[_p_] = W[_p_]/(gamma-1)+K
        return U

def compute_fluxes_from_primitives(F,W,vels,_p_,gamma)->np.ndarray:
        K = W[0].copy()*0
        v1=vels[0]
        for v in vels[::-1]:
            #Iterate over inverted array of vels
            #so that the last value of m correspond to the 
            #normal component
            m = W[0]*W[v]
            K += m*W[v]
            F[v,...] = m*W[v1]
        E = W[_p_]/(gamma-1) + 0.5*K
        F[0  ,...] = m
        F[v1,...] = m*W[v1] + W[_p_]
        F[_p_,...] = W[v1]*(E + W[_p_])

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