"""SD initialization must represent the same conserved state in both layouts."""

import numpy as np
import pytest

from spd import initial_conditions as ics
from spd.sdfb_simulator import SPD_Simulator


@pytest.mark.parametrize("p", [3, 7])
@pytest.mark.parametrize("scheme", ["SD", "SDFB"])
def test_dmr_initial_state_and_zero_timestep(p, scheme):
    sim = SPD_Simulator(
        p=p, N=(8, 2), xlim=(0., 4.), ylim=(0., 1.),
        init_fct=ics.double_mach_reflection(gamma=1.4),
        BC=(("doublemach", "doublemach"),) * 2,
        scheme=scheme, time_integrator="rk3", use_cupy=False,
    )
    initial = sim.dm.U_cv.copy()
    projected = sim.compute_cv_from_sp(sim.dm.U_sp)
    np.testing.assert_allclose(projected, initial, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(
        sim.compute_cv_from_sp(sim.dm.W_sp), sim.dm.W_cv, atol=1e-12, rtol=1e-12,
    )

    sim.dt = sim.ho_scheme.dt = sim.lo_scheme.dt = 0.
    with np.errstate(divide="ignore", invalid="ignore"):
        sim.perform_update()
    np.testing.assert_allclose(sim.dm.U_cv, initial, atol=1e-12, rtol=1e-12)


def linear_velocity(mesh, var):
    if var == 1:
        return mesh[0]
    return np.full_like(mesh[0], 1. if var in (0, 4) else 0.)


@pytest.mark.parametrize("p", [3, 7])
def test_linear_velocity_energy_averages(p):
    sim = SPD_Simulator(
        p=p, N=(2,), init_fct=linear_velocity, use_cupy=False,
    )
    # rho = P = 1, vx = x: the nodal and averaged kinetic energies differ.
    internal = 1 / (sim.gamma - 1)
    np.testing.assert_allclose(
        sim.dm.U_sp[4], internal + 0.5 * sim.dm.X_sp**2, atol=1e-12, rtol=1e-12,
    )
    faces = sim.faces["x"][sim.Nghc:-sim.Nghc]
    centers = 0.5 * (faces[:-1] + faces[1:])
    expected = internal + 0.5 * centers**2 + np.diff(faces)**2 / 24
    np.testing.assert_allclose(sim.dm.U_cv[4].ravel(), expected, atol=1e-12, rtol=1e-12)
