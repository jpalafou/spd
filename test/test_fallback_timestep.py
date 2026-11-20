"""Advection and diffusion numbers use average subcell widths with fallback."""

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
        viscosity=nu > 0, nu=nu, cfl_coeff=0.6, cfl_diffusion=0.01,
        time_integrator=time_integrator, use_cupy=False,
    )
    hx, hy = (sim.h[d] / (p + 1) for d in "xy")

    def expected_dt(vx):
        advective = 0.6 / ((np.sqrt(1.4) + vx) / hx + (np.sqrt(1.4) + 0.25) / hy)
        diffusive = 0.01 / (nu * (hx**-2 + hy**-2)) if nu else np.inf
        return min(advective, diffusive)

    dt = sim.dt
    sim.ho_scheme.compute_dt()
    assert dt == pytest.approx(min(sim.ho_scheme.dt, expected_dt(0.5)))
    sim.lo_scheme.compute_dt()
    assert sim.lo_scheme.dt == pytest.approx(expected_dt(0.5))
    # Recompute after a change in the state, rather than retaining the initial dt.
    sim.dm.W_cv[1] = 1e4
    sim.compute_dt()
    assert sim.dt == pytest.approx(expected_dt(1e4))
    assert sim.ho_scheme.dt == sim.dt
    assert sim.lo_scheme.dt == sim.dt


@pytest.mark.parametrize("p", [3, 7])
@pytest.mark.parametrize("time_integrator", ["rk3", "ader"])
@pytest.mark.parametrize("cfl_coeff", [0.3, 0.6])
def test_advection_cfl_independent_of_fallback(p, time_integrator, cfl_coeff):
    expected = cfl_coeff / ((p + 1) * (3 * (np.sqrt(1.4) + 0.5) + 2 * (np.sqrt(1.4) + 0.25)))
    for FB in (False, True):
        sim = SPD_Simulator(
            FB=FB, init_fct=uniform_flow, N=(3, 2), p=p,
            time_integrator=time_integrator, cfl_coeff=cfl_coeff, use_cupy=False,
        )
        assert sim.dt == pytest.approx(expected)
