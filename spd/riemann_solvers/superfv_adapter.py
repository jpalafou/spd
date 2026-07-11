"""Thin adapter from SPD's Riemann-solver signature to SuperFV's solvers."""

from functools import lru_cache

import numpy as np


_SUPERFV_DIM_FROM_VEL = {1: "x", 2: "y", 3: "z"}


def uses_superfv_riemann(equations: str, solver_name: str) -> bool:
    return equations == "hydro" and solver_name in ("hllc", "llf")


@lru_cache(maxsize=None)
def _superfv_riemann_kernel(solver_name, npassive):
    from superfv.riemann_solvers import RiemannSolver, solve_riemann_problem
    from superfv.tools.variable_index_map import VariableIndexMap

    var_idx_map = {
        "rho": 0,
        "vx": 1,
        "vy": 2,
        "vz": 3,
        "P": 4,
        "mx": 1,
        "my": 2,
        "mz": 3,
        "E": 4,
    }
    group_var_map = {
        "v": ["vx", "vy", "vz"],
        "m": ["mx", "my", "mz"],
        "primitives": ["rho", "v", "P"],
        "conservatives": ["rho", "m", "E"],
    }
    if npassive == 1:
        var_idx_map["dye"] = 5
        group_var_map["passives"] = ["dye"]
    idx = VariableIndexMap(var_idx_map, group_var_map=group_var_map)
    solver = {
        "hllc": RiemannSolver.HLLC,
        "llf": RiemannSolver.LLF,
    }[solver_name]
    return solver, solve_riemann_problem, idx


def superfv_riemann(
    scheme,
    M_L: np.ndarray,
    M_R: np.ndarray,
    F: np.ndarray,
    vels: np.array,
    _p_: int,
    gamma: float,
    min_c2: float,
    prims: bool,
    solver_name: str,
    call_timer: bool = True,
    **kwargs,
) -> np.ndarray:
    del _p_, min_c2

    npassive = kwargs.get("npassive", scheme.npassive)
    if npassive not in (0, 1):
        raise NotImplementedError(
            "SuperFV Riemann adapter supports at most one passive dye."
        )
    if npassive == 1 and scheme.passives != ["dye"]:
        raise NotImplementedError(
            "SuperFV Riemann adapter only supports the passive variable 'dye'."
        )

    riemann_solver, solve_riemann_problem, idx = _superfv_riemann_kernel(
        solver_name, npassive
    )
    dim = _SUPERFV_DIM_FROM_VEL[int(vels[0])]
    W_L = M_L if prims else scheme.compute_primitives(M_L)
    W_R = M_R if prims else scheme.compute_primitives(M_R)
    if W_R is F:
        W_R = W_R.copy()
    F[...] = 0.0

    call_timer and scheme._start_subtimer("riemann_solver")
    solve_riemann_problem(W_L, W_R, F, riemann_solver, dim, idx, gamma)
    call_timer and scheme._stop_subtimer("riemann_solver")

    return F
