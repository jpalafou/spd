"""Fallback timestep bounds must use physical SD subcell widths."""

import numpy as np
import pytest

from spd.sdfb_simulator import SPD_Simulator


def uniform_flow(mesh, var):
    return np.full_like(mesh[0], [1.0, 0.5, 0.25, 0.0, 1.0][var])


@pytest.mark.parametrize("p", [3, 7])
@pytest.mark.parametrize("nu", [0.0, 1.0])
@pytest.mark.parametrize("time_integrator", ["rk3", "ader"])
def test_fallback_subcell_timestep(p, nu, time_integrator):
    sim = SPD_Simulator(
        init_fct=uniform_flow, N=(3, 2), xlim=(0, 2), p=p,
        viscosity=nu > 0, nu=nu, cfl_coeff=0.6,
        time_integrator=time_integrator, use_cupy=False,
    )
    h = min(np.min(np.diff(sim.fp[d])) * sim.h[d] for d in sim.dims)

    def expected_dt(vx):
        advective = h / (2 * np.sqrt(1.4) + vx + 0.25)
        diffusive = 0.25 * h**2 / nu if nu else np.inf
        return 0.6 * min(advective, diffusive)

    dt = sim.dt
    sim.ho_scheme.compute_dt()
    assert dt == pytest.approx(min(sim.ho_scheme.dt, expected_dt(0.5)))
    # Recompute after a change in the state, rather than retaining the initial dt.
    sim.dm.W_cv[1] = 1e4
    sim.compute_dt()
    assert sim.dt == pytest.approx(expected_dt(1e4))
    assert sim.ho_scheme.dt == sim.dt
    assert sim.lo_scheme.dt == sim.dt


def test_unlimited_sd_retains_its_timestep():
    sim = SPD_Simulator(
        FB=False, init_fct=uniform_flow, N=(3, 2), p=3,
        time_integrator="rk3", cfl_coeff=0.6, use_cupy=False,
    )
    sim.compute_dt()
    expected = 0.6 * (1 / 3) / 4 / (2 * np.sqrt(1.4) + 0.75)
    assert sim.dt == pytest.approx(expected)
