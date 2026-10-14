"""Reduced-dimensional viscous stresses use zero inactive derivatives."""

import numpy as np
import pytest

from spd import hydro


@pytest.mark.parametrize("ndim, normal", [(1, 0), (2, 0), (2, 1)])
def test_inactive_viscous_derivatives(ndim, normal):
    rng = np.random.default_rng(42)
    W = rng.uniform(0.5, 2.0, (5, 8))
    gradients = {d: rng.normal(size=W.shape) for d in range(ndim)}
    full_gradients = {
        d: gradients[d] if d in gradients else np.zeros_like(W) for d in range(3)
    }
    vels = np.roll([1, 2, 3], -normal)
    expected = hydro.compute_viscous_fluxes(W, vels, full_gradients, 4, 0.1, 2 / 3)
    actual = hydro.compute_viscous_fluxes(W, vels, gradients, 4, 0.1, 2 / 3)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("normal", [0, 1, 2])
def test_viscous_energy_is_stress_work(ndim, normal):
    rng = np.random.default_rng(42)
    W = rng.uniform(0.5, 2., (7, 8))
    # Include all three directions, setting inactive derivatives to zero.
    gradients = {
        d: rng.normal(size=W.shape) if d < ndim else np.zeros_like(W)
        for d in range(3)
    }
    F = hydro.compute_viscous_fluxes(
        W, np.roll([1, 2, 3], -normal), gradients, 4, 0.1, 2 / 3, npassive=2,
    )
    np.testing.assert_allclose(F[4], np.sum(W[1:4] * F[1:4], axis=0))
    np.testing.assert_allclose(F[5:], 0.1 * W[0] * gradients[normal][5:])
