"""Check SPD's interface against SuperFV, including SD's in-place output."""

import numpy as np
import pytest
from superfv.riemann_solvers import solve_riemann_problem
from superfv.tools.variable_index_map import VariableIndexMap

from spd import hydro
from spd.riemann_solvers.riemann_solver_1D import Riemann_solver_1D


@pytest.mark.parametrize("name", ["hllc", "llf"])
@pytest.mark.parametrize("dim", "xyz")
@pytest.mark.parametrize("prims", [False, True])
@pytest.mark.parametrize("npassive", [0, 2])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_superfv_flux(name, dim, prims, npassive, inplace, backend):
    xp = np if backend == "numpy" else pytest.importorskip("cupy")
    rng = np.random.default_rng(42)
    wl = xp.asarray(rng.uniform(0.5, 2.0, (5 + npassive, 2, 3, 4)))
    wr = xp.asarray(rng.uniform(0.5, 2.0, wl.shape))
    wl[1:4] -= 1.25
    wr[1:4] -= 1.25
    # Interface buffers can be noncontiguous.
    wl, wr = wl[..., ::2], wr[..., ::2]
    vels = np.roll([1, 2, 3], -"xyz".index(dim))
    gamma = 1.4
    variables = {"rho": 0, "P": 4, "E": 4}
    for i, axis in enumerate("xyz", 1):
        variables[f"v{axis}"] = variables[f"m{axis}"] = i
    passives = [f"tracer{i}" for i in range(npassive)]
    variables.update({v: 5 + i for i, v in enumerate(passives)})
    idx = VariableIndexMap(
        variables,
        group_var_map={"passives": passives} if passives else None,
    )
    expected = xp.empty_like(wl)
    solve_riemann_problem(wl, wr, expected, name, dim, idx, gamma)

    ml = wl.copy() if prims else hydro.compute_conservatives(
        wl, vels, 4, gamma, npassive=npassive
    )
    mr = wr.copy() if prims else hydro.compute_conservatives(
        wr, vels, 4, gamma, npassive=npassive
    )
    left_before, right_before = ml.copy(), mr.copy()
    flux = mr if inplace else xp.empty_like(mr)
    result = Riemann_solver_1D(name, "hydro").solver(
        ml, mr, flux, vels, 4, gamma, 1e-10, prims, npassive=npassive,
    )
    assert result is flux
    to_numpy = np.asarray if backend == "numpy" else xp.asnumpy
    np.testing.assert_allclose(
        to_numpy(flux), to_numpy(expected), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_array_equal(to_numpy(ml), to_numpy(left_before))
    if not inplace:
        np.testing.assert_array_equal(to_numpy(mr), to_numpy(right_before))


@pytest.mark.parametrize("name", ["hllc", "llf"])
def test_uniform_state_with_temperature(name):
    # rho=2, vx=3, P=4: mass flux=6, momentum flux=22, energy flux=69.
    state = np.array([[2.0], [3.0], [0.0], [0.0], [4.0], [2.0]])
    flux = state.copy()
    Riemann_solver_1D(name, "hydro").solver(
        state, flux, flux, np.array([1, 2, 3]), 4, 1.4, 1e-10, True,
        thdiffusion=True, _t_=5,
    )
    np.testing.assert_allclose(flux[:, 0], [6, 22, 0, 0, 69, 0])
