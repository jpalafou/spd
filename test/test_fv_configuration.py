"""The SPD driver's FV mode must respect FV numerical settings."""

import numpy as np
import pytest

from spd.sdfb_simulator import SPD_Simulator


def uniform_flow(mesh, var):
    return np.full_like(mesh[0], [1.0, 0.5, 0.25, 0.0, 1.0][var])


@pytest.mark.parametrize("p", [0, 1, 3])
def test_fv_cfl_independent_of_sd_degree(p):
    sim = SPD_Simulator(
        scheme="FV", init_fct=uniform_flow, N=(4, 3), p=p,
        time_integrator="rk2", cfl_coeff=0.6, use_cupy=False,
    )
    sim.compute_dt()
    expected = 0.6 * 0.25 / (2 * np.sqrt(1.4) + 0.75)
    assert sim.dt == pytest.approx(expected)
