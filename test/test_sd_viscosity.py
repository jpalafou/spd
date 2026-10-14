"""Periodic shear flow should decay under SD viscosity."""

import numpy as np
import pytest

from spd.spectral_difference.sd_simulator import SD_Simulator
from spd.spectral_difference import sd_boundary as bc


def shear_flow(mesh, var):
    if var in (0, 4):
        return np.ones_like(mesh[0])
    if var == 2:
        return 0.1 * np.sin(2 * np.pi * mesh[0])
    return np.zeros_like(mesh[0])


@pytest.mark.parametrize("time_integrator", ["ader", "rk2"])
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


@pytest.mark.parametrize("p", [3, 7])
@pytest.mark.parametrize("ndim, normal", [(2, 0), (2, 1), (3, 2)])
@pytest.mark.parametrize("ader", [False, True])
def test_reflecting_wall_stress(p, ndim, normal, ader):
    boundaries = [("periodic", "periodic")] * ndim
    boundaries[normal] = ("reflective", "reflective")
    sim = SD_Simulator(
        N=(2,) * ndim, p=p, time_integrator="ader" if ader else "rk3",
        viscosity=True, nu=1., use_cupy=False, BC=tuple(boundaries),
    )
    sc = sim.scheme
    shape = [1] * (2 * ndim)
    shape[ndim - 1 - normal] = 2
    shape[2 * ndim - 1 - normal] = p + 1
    coordinate = getattr(sim.dm, "XYZ"[normal] + "_sp").reshape(shape)
    W = np.zeros_like(sim.dm.U_sp)
    W[[0, 4]] = 1
    W[normal + 1] = coordinate * (1 - coordinate)
    U = sim.compute_conservatives(W)
    if ader:
        U = U[:, None]
    for dim in sc.dims:
        sc.M_fp[dim][...] = sc.compute_fp_from_sp(U, dim, ader=ader)
        sc.F_fp[dim][...] = 0
    sc.add_nabla_terms(ader=ader)
    momentum_flux = sc.F_fp["xyz"[normal]][normal + 1]
    for side, expected in [(0, -4 / 3), (-1, 4 / 3)]:
        wall = [slice(None)] * momentum_flux.ndim
        wall[int(ader) + ndim - 1 - normal] = side
        wall[int(ader) + 2 * ndim - 1 - normal] = side
        np.testing.assert_allclose(momentum_flux[tuple(wall)], expected, atol=1e-12)


@pytest.mark.parametrize("normal", [0, 1, 2])
@pytest.mark.parametrize("derivative", [0, 1, 2])
def test_gradient_reflection_parity(normal, derivative):
    boundaries = [("periodic", "periodic")] * 3
    boundaries[normal] = ("reflective", "reflective")
    sim = SD_Simulator(
        N=(2, 2, 2), p=1, time_integrator="rk3", BC=tuple(boundaries),
        viscosity=True, thdiffusion=True, passives=["dye"], use_cupy=False,
    )
    dim = "xyz"[normal]
    sc = sim.scheme
    # Unit gradients make the even/odd symmetry visible in every field,
    # including density, pressure, dye, and diagnostic temperature.
    bc.Boundaries_sd(sc, np.ones_like(sc.M_fp[dim]), dim, gradient_dim="xyz"[derivative])
    expected = np.full(sim.nvar, -1. if derivative == normal else 1.)
    expected[normal + 1] *= -1
    for side in (0, 1):
        trace = sc.BC_fp[dim][side]
        np.testing.assert_array_equal(
            trace, np.broadcast_to(expected.reshape((-1,) + (1,) * (trace.ndim - 1)), trace.shape),
        )
