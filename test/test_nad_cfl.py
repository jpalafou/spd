"""CFL-scaled NAD tolerances follow the candidate update, including ADER weights."""

import numpy as np
import pytest

from spd.fallback import trouble_detection
from spd.sdfb_simulator import SPD_Simulator


def uniform_flow(mesh, var):
    # The inactive z velocity must not contribute to the characteristic speed.
    return np.full_like(mesh[0], [1.0, -0.5, 0.25, 10.0, 1.0][var])


@pytest.mark.parametrize("nad", ["", "delta"])
@pytest.mark.parametrize("integrator", ["rk3", "ader"])
@pytest.mark.parametrize("courant", [None, 0.2, 0.8, 10.0])
@pytest.mark.parametrize("use_cupy", [False, True])
@pytest.mark.parametrize("N", [(3,), (3, 2)])
def test_nad_cfl_candidate_tolerance(
    monkeypatch, nad, integrator, courant, use_cupy, N,
):
    if use_cupy:
        cp = pytest.importorskip("cupy")
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("No CUDA device")

    sim = SPD_Simulator(
        init_fct=uniform_flow, N=N, xlim=(0, 2), p=3,
        time_integrator=integrator, NAD=nad, tolerance=0.01,
        SED=False, use_cupy=use_cupy,
        **({"NAD_cfl": True} if courant is not None else {}),
    )
    widths = [
        np.tile(np.diff(sim.fp[d]) * sim.h[d], sim.N[d]) for d in sim.dims
    ]
    h = widths[0] if sim.ndim == 1 else np.maximum(widths[0][None, :], widths[1][:, None])
    h = h[None, ...]
    speed = sim.ndim * np.sqrt(1.4) + sum([0.5, 0.25][:sim.ndim])
    dt = (0.2 if courant is None else courant) * h.min() / speed
    sim.dt = sim.ho_scheme.dt = sim.lo_scheme.dt = dt
    sim.init_sim()
    initial = sim.dm.W_cv.copy()
    tolerances = []
    nad_check = trouble_detection.nad_check

    def record_check(W, lo, hi, alpha, tolerance, delta_mode):
        tolerances.append(cp.asnumpy(tolerance) if use_cupy else tolerance)
        assert delta_mode == (nad == "delta")
        # Exercise both bounds with overshoots accepted by the old tolerance
        # but rejected when the multiplier tightens it sufficiently.
        xp = cp if use_cupy else np
        lower = xp.broadcast_to(xp.array([-2.0, 1.0]).reshape(2, *([1] * sim.ndim)), W.shape)
        upper = xp.broadcast_to(xp.array([1.0, 3.0]).reshape(2, *([1] * sim.ndim)), W.shape)
        scale = upper - lower if delta_mode else xp.abs(lower)
        candidate = lower - 0.005 * scale
        flags = nad_check(candidate, lower, upper, None, tolerance, delta_mode)
        assert bool(xp.all(flags == (tolerance < 0.005)))
        scale = upper - lower if delta_mode else xp.abs(upper)
        candidate = upper + 0.005 * scale
        flags = nad_check(candidate, lower, upper, None, tolerance, delta_mode)
        assert bool(xp.all(flags == (tolerance < 0.005)))
        return nad_check(W, lo, hi, alpha, tolerance, delta_mode)

    monkeypatch.setattr(trouble_detection, "nad_check", record_check)
    sim.perform_update()
    if integrator == "ader":
        weights = sim.lo_scheme.dm.w_tp
        weights = cp.asnumpy(weights) if use_cupy else weights
    else:
        weights = np.ones(3)
    expected = (
        np.full_like(weights, 0.01) if courant is None
        else np.stack([
            0.01 * np.minimum(1.0, courant * weight * h.min() / h)
            for weight in weights
        ])
    )
    np.testing.assert_allclose(tolerances, expected, rtol=1e-12)
    final = sim.dm.W_cv
    if use_cupy:
        initial, final = cp.asnumpy(initial), cp.asnumpy(final)
    np.testing.assert_allclose(final, initial, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("nad", ["", "delta"])
def test_nad_cfl_rejects_well_balanced(nad):
    with pytest.raises(NotImplementedError, match="NAD_cfl.*WB=True"):
        SPD_Simulator(NAD=nad, NAD_cfl=True, WB=True, init=False, use_cupy=False)
    # Well-balanced schemes still accept the default, unscaled tolerance.
    sim = SPD_Simulator(NAD=nad, WB=True, init=False, use_cupy=False)
    assert not sim.lo_scheme.NAD_cfl


def test_nad_cfl_degree_zero_keeps_zero_tolerance(monkeypatch):
    sim = SPD_Simulator(
        init_fct=uniform_flow, N=(3,), p=0, time_integrator="rk1",
        NAD_cfl=True, use_cupy=False,
    )
    tolerances = []
    nad_check = trouble_detection.nad_check

    def record_check(W, lo, hi, alpha, tolerance, delta_mode):
        tolerances.append(tolerance)
        return nad_check(W, lo, hi, alpha, tolerance, delta_mode)

    monkeypatch.setattr(trouble_detection, "nad_check", record_check)
    sim.perform_iterations(1)
    assert tolerances == [0.0]
