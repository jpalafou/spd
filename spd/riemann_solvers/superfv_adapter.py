"""Adapt SuperFV's hydrodynamic Riemann solvers to SPD's calling interface."""

from contextlib import nullcontext
from functools import lru_cache

from superfv.riemann_solvers import solve_riemann_problem
from superfv.tools.variable_index_map import VariableIndexMap

from spd import hydro


@lru_cache(maxsize=None)
def _variable_index_map(npassive):
    variables = {
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
    groups = {"v": ["vx", "vy", "vz"], "m": ["mx", "my", "mz"]}
    if npassive:
        passives = [f"passive{i}" for i in range(npassive)]
        variables.update({name: 5 + i for i, name in enumerate(passives)})
        groups["passives"] = passives
    return VariableIndexMap(variables, group_var_map=groups)


def superfv_riemann_solver(name):
    def solve(M_L, M_R, F, vels, _p_, gamma, min_c2, prims, npassive=0,
              timer=None, call_timer=True, **kwargs):
        # SuperFV computes its own sound speeds, without SPD's min_c2 floor.
        if prims:
            W_L, W_R = M_L, M_R
        else:
            with timer._time("primitive_conservative") if timer is not None else nullcontext():
                W_L = hydro.compute_primitives(
                    M_L, vels, _p_, gamma, npassive=npassive, **kwargs
                )
            with timer._time("primitive_conservative") if timer is not None else nullcontext():
                W_R = hydro.compute_primitives(
                    M_R, vels, _p_, gamma, npassive=npassive, **kwargs
                )

        # SD reuses the right interface state as the output buffer. SuperFV's
        # NumPy kernels read the primitive states while writing the fluxes.
        if W_L is F:
            W_L = W_L.copy()
        if W_R is F:
            W_R = W_R.copy()

        nvar = 5 + npassive
        with timer._time("riemann_solver", call_timer) if timer is not None else nullcontext():
            solve_riemann_problem(
                W_L[:nvar],
                W_R[:nvar],
                F[:nvar],
                name,
                "xyz"[int(vels[0]) - 1],
                _variable_index_map(npassive),
                gamma,
            )
        # Temperature, when present, is a diagnostic field with no advective flux.
        F[nvar:] = 0.0
        return F

    return solve
