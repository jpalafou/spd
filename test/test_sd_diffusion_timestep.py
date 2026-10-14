"""Check the smallest-subcell timestep against periodic SD diffusion."""

from math import factorial

import numpy as np
import pytest

from spd.spectral_difference.sd_simulator import SD_Simulator


def constant(mesh, var):
    return np.ones_like(mesh[0]) if var in (0, 4) else np.zeros_like(mesh[0])


@pytest.fixture(scope="module", params=[3, 7])
def diffusion_spectrum(request):
    p = request.param
    sim = SD_Simulator(
        N=(2,), xlim=(0, 2), p=p, time_integrator="rk3", init_fct=constant,
        viscosity=True, nu=1., passives=["dye"], use_cupy=False,
    )
    sc = sim.scheme
    n = 2 * (p + 1)
    operator = np.empty((n, n))
    for j in range(n):
        U = sim.dm.U_sp.copy()
        U[5] = 0
        U[5].flat[j] = 1
        sc.M_fp["x"][...] = sc.compute_fp_from_sp(U, "x", ader=False)
        sc.F_fp["x"][...] = 0
        sc.add_nabla_terms(ader=False)
        operator[:, j] = -sc.compute_dudt(U)[5].ravel()
    return p, np.linalg.eigvals(operator)


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("integrator", ["rk1", "rk2", "rk3", "rk4", "ader"])
def test_diffusion_amplification(diffusion_spectrum, ndim, integrator):
    p, eigenvalues = diffusion_spectrum
    sim = SD_Simulator(
        N=(2,) * ndim, p=p, time_integrator=integrator, init_fct=constant,
        viscosity=True, nu=1., cfl_coeff=0.8, use_cupy=False,
    )
    rates = np.array([0.])
    for d in sim.dims:
        rates = np.add.outer(rates, eigenvalues / sim.h[d]**2).ravel()
    order = p + 1 if integrator == "ader" else int(integrator[2:])
    coefficients = [1 / factorial(j) for j in range(order + 1)]
    amplification = np.polynomial.polynomial.polyval(sim.dt * rates, coefficients)
    assert np.max(np.abs(amplification)) <= 1 + 1e-11


@pytest.mark.parametrize("chi", [0., 100.])
def test_smallest_physical_subcell(diffusion_spectrum, chi):
    p, _ = diffusion_spectrum
    sim = SD_Simulator(
        N=(3, 2), xlim=(0, 2), p=p, time_integrator="rk3", init_fct=constant,
        viscosity=True, nu=1., thdiffusion=chi > 0, chi=chi, use_cupy=False,
    )
    h_min = min(np.min(np.diff(sim.fp[d])) * sim.h[d] for d in sim.dims)
    diffusivity = max(4 / 3, (sim.gamma - 1) * chi)
    expected = sim.cfl_coeff * h_min**2 / (2 * sim.ndim * diffusivity)
    assert sim.dt == pytest.approx(expected)
