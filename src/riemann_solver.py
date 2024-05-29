import numpy as np
import hydro

def llf(
    s: "simulator",
    M_L: np.ndarray,
    M_R: np.ndarray,
    vels: np.array,
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
    if prims:
        W_L = M_L
        W_R = M_R
        U_L = s.compute_conservatives(W_L)
        U_R = s.compute_conservatives(W_R)
    else:
        U_L = M_L
        U_R = M_R
        W_L = s.compute_primitives(U_L)
        W_R = s.compute_primitives(U_R)
    
    v_1 = vels[0]
    F_L = hydro.compute_fluxes(s,W_L,vels)
    F_R = hydro.compute_fluxes(s,W_R,vels)
    
    c_L = hydro.compute_cs(W_L[s._p_],W_L[s._d_],s.gamma,s.min_c2) + np.abs(W_L[v_1])
    c_R = hydro.compute_cs(W_R[s._p_],W_R[s._d_],s.gamma,s.min_c2) + np.abs(W_R[v_1])
    
    c_max = np.where(c_L>c_R,c_L,c_R)[np.newaxis,...]
        
    M_R[...] = M_L[...] = 0.5*(F_R+F_L)-0.5*c_max*(U_R-U_L)

