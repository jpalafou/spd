"""Periodic shear flow should decay under SD viscosity."""

import numpy as np
import pytest

from spd.spectral_difference.sd_simulator import SD_Simulator


def shear_flow(mesh, var):
    if var in (0, 4):
        return np.ones_like(mesh[0])
    if var == 2:
        return 0.1 * np.sin(2 * np.pi * mesh[0])
    return np.zeros_like(mesh[0])


@pytest.mark.parametrize("time_integrator", ["ader"])
def test_periodic_viscous_shear(time_integrator):
    sim = SD_Simulator(
        init_fct=shear_flow, N=(4, 3), p=2, viscosity=True, nu=0.01,
        time_integrator=time_integrator, cfl_coeff=0.1, use_cupy=False,
        verbose=False,
    )
    initial = sim.scheme.compute_cv_from_sp(sim.dm.U_sp)
    sim.perform_iterations(2)
    assert np.isfinite(sim.dm.U_cv).all()
    assert np.linalg.norm(sim.dm.U_cv[2]) < np.linalg.norm(initial[2])
    volumes = np.diff(sim.fp["y"])[:, None] * np.diff(sim.fp["x"])[None, :]
    axes = tuple(range(1, initial.ndim))
    np.testing.assert_allclose(
        (sim.dm.U_cv * volumes).sum(axis=axes),
        (initial * volumes).sum(axis=axes), atol=1e-12,
    )
