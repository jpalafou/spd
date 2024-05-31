import numpy as np
import hydro

def llf(
    M_L: np.ndarray,
    M_R: np.ndarray,
    vels: np.array,
    _p_: int,
    gamma: float,
    min_c2: float,
    prims: bool,
    *args,
    **kwargs,
) -> None:
    """
    LLF Riemann Solver
    M_L/M_R = Primitive variables or Conservative variables
    v_1 velocity normal to the interface
    v_2 velocity parallel to the interface
    Output:
    The resulting fluxes are written in M_R
    """
    #Density index
    _d_=0 
    if prims:
        W_L = M_L
        W_R = M_R
        U_L = hydro.compute_conservatives(W_L,vels,_p_,gamma)
        U_R = hydro.compute_conservatives(W_R,vels,_p_,gamma)
    else:
        U_L = M_L
        U_R = M_R
        W_L = hydro.compute_primitives(U_L,vels,_p_,gamma)
        W_R = hydro.compute_primitives(U_R,vels,_p_,gamma)
    
    v_1 = vels[0]
    F_L = hydro.compute_fluxes(W_L,vels,_p_,gamma)
    F_R = hydro.compute_fluxes(W_R,vels,_p_,gamma)
    
    c_L = hydro.compute_cs(W_L[_p_],W_L[_d_],gamma,min_c2) + np.abs(W_L[v_1])
    c_R = hydro.compute_cs(W_R[_p_],W_R[_d_],gamma,min_c2) + np.abs(W_R[v_1])
    
    c_max = np.where(c_L>c_R,c_L,c_R)[np.newaxis,...]
        
    M_R[...] = M_L[...] = 0.5*(F_R+F_L)-0.5*c_max*(U_R-U_L)

