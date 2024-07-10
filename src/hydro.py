from typing import Union

import numpy as np
import cupy as cp

def compute_cs2(
        P: np.ndarray,
        rho: np.ndarray,
        gamma: float,
        min_c2: float)->np.ndarray:
    """
    INPUT:
    P: Array of Pressure values
    rho: Array of density values
    gamma: Adiabatic index
    min_c2: Minimum value allowed for the square of the sound speed
    OUTPUT:
    Cs^2: Sound speed square
    """
    c2 = gamma*P/rho
    #np.maximum propagates NaNs, so we use np.where
    c2 = np.where(c2>min_c2,c2,min_c2)
    return c2

def compute_cs(
        P: np.ndarray,
        rho: np.ndarray,
        gamma: float,
        min_c2: float)->np.ndarray:
    """
    INPUT:
    P: Array of Pressure values
    rho: Array of density values
    gamma: Adiabatic index
    min_c2: Minimum value allowed for the square of the sound speed
    OUTPUT:
    Cs: Array of sound speed values
    """
    return np.sqrt(compute_cs2(P,rho,gamma,min_c2))

def compute_primitives(
        U: np.ndarray,
        vels: np.array,
        _p_: int,
        gamma: float,
        W=None)->np.ndarray:
    """
    INPUT:
    U: Solution array of conseved variables
    vels: array containing the indices of velocity components [vx,vy,vz]
    in the Solution array. The size of this array has to match the number of dimensions
    _p_: index of pressure/energy in the Solution array
    gamma: Adiabatic index
    OUTPUT:
    W: Solution array of primitive variables
    """
    if type(W)==type(None):
        W = U.copy()
    assert W.shape == U.shape
    K = W[0].copy()*0
    for vel in vels:
        W[vel] = U[vel]/U[0]
        K += W[vel]**2
    K *= 0.5*U[0]
    W[_p_] = (gamma-1)*(U[_p_]-K)
    return W
                
def compute_conservatives(
        W: np.ndarray,
        vels: np.array,
        _p_: int,
        gamma: float,
        U=None)->np.ndarray:
    """
    INPUT:
    W: Solution array of primitive variables
    vels: array containing the indices of velocity components [vx,vy,vz]
    in the Solution array. The size of this array has to match the number of dimensions
    _p_: index of pressure/energy in the Solution array
    gamma: Adiabatic index
    OUTPUT:
    U: Solution array of conserved variables
    """
    if type(U)==type(None):
        U = W.copy()
    assert U.shape == W.shape
    K = W[0].copy()*0
    for vel in vels:
        U[vel] = W[vel]*U[0]
        K += W[vel]**2
    K  *= 0.5*U[0]
    U[_p_] = W[_p_]/(gamma-1)+K
    return U

def compute_fluxes(
        W: np.ndarray,
        vels: np.array,
        _p_: int,
        gamma: float,
        F=None)->np.ndarray:
    """
    INPUT:
    W: Solution array of primitive variables
    vels: array containing the indices of velocity components [vx,vy,vz]
    in the Solution array. The size of this array has to match the number of dimensions
    _p_: index of pressure/energy in the Solution array
    gamma: Adiabatic index
    OUTPUT:
    F: Solution array of fluxes for the conserved variables
    """
    if type(F)==type(None):
        F = W.copy()
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
    return F

def compute_viscous_fluxes(
        W: np.ndarray,
        vels: np.array,
        dUs: dict,
        _e_: int,
        nu: float,
        beta: float,
        F=None)->np.ndarray:
    if type(F)==type(None):
        F = W.copy()
    F[...] = 0
    #index of normal component
    v1  = vels[0]
    #Gradient in normal dimension
    dU1 = dUs[v1-1]
    #Flux is normal dimension
    F[v1] = 2*dU1[v1] - beta*dU1[v1]
    #Energy flux
    F[_e_] = W[v1]*F[v1]
    for vel in vels[1:]:
        idim = vel-1
        dU = dUs[idim]
        F[v1]  -= beta*dU[vel]
        F[vel]  = (dU1[vel]+dU[v1])
        F[_e_] += W[vel]*F[vel]
    return F*W[0]*nu