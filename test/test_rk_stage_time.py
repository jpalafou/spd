"""Moving DMR boundaries must follow the RK stage times in every scheme."""

import numpy as np
import pytest

from spd import initial_conditions as ics
from spd.sdfb_simulator import SPD_Simulator


@pytest.mark.parametrize("scheme", ["SD", "SDFB", "FV"])
@pytest.mark.parametrize("integrator", ["rk1", "rk2", "rk3", "rk4"])
def test_dmr_boundary_stage_times(scheme, integrator, monkeypatch):
    sim = SPD_Simulator(
        p=3, N=(8, 2), xlim=(0., 4.), ylim=(0., 1.),
        init_fct=ics.double_mach_reflection(gamma=1.4),
        BC=(("doublemach", "doublemach"),) * 2,
        scheme=scheme, time_integrator=integrator, cfl_coeff=0.1, use_cupy=False,
    )
    ho = sim.ho_scheme
    fv = sim.lo_scheme if scheme == "SDFB" else ho
    angle = np.pi / 3
    shock0 = 1 / 6 + 1 / np.tan(angle)
    x = ho.dm.X_sp if scheme != "FV" else fv.centers["x"]
    # Arrange a top-boundary point to cross the shock during this timestep.
    dt = sim.dt
    t0 = (x[x > shock0].min() - shock0) * np.sin(angle) / 10 - 0.75 * dt
    sim.time = t0
    times, sd_traces, fv_traces = [], [], []
    compute_update = sim.scheme.compute_update

    def record_boundaries(U, **kwargs):
        rhs = compute_update(U, **kwargs)
        times.append(sim.time)
        if scheme != "FV":
            sd_traces.append(ho.BC_fp["y"][1][0].copy())
        if scheme != "SD":
            fv_traces.append(fv.BC_fp["y"][1][0].copy())
        return rhs

    monkeypatch.setattr(sim.scheme, "compute_update", record_boundaries)
    with np.errstate(divide="ignore", invalid="ignore"):
        sim.perform_update()
    expected_times = t0 + dt * sim.integrator.c
    np.testing.assert_allclose(times, expected_times, atol=1e-15, rtol=0)
    assert sim.time == pytest.approx(t0 + dt)
    for i, t in enumerate(expected_times):
        shock_x = 10 * t / np.sin(angle) + 1 / 6
        if scheme != "FV":
            expected = np.where(ho.dm.X_sp < shock_x + 1 / np.tan(angle), 8., 1.4)
            np.testing.assert_array_equal(sd_traces[i], expected)
        if scheme != "SD":
            y = fv.centers["y"][-sim.Nghc:, None]
            expected = np.where(
                fv.centers["x"] < shock_x + y / np.tan(angle), 8., 1.4,
            )
            np.testing.assert_array_equal(fv_traces[i], expected)


def test_stage_time_restored_on_rhs_failure(monkeypatch):
    sim = SPD_Simulator(N=(2,), p=1, time_integrator="rk3", use_cupy=False)
    sim.time = 0.3

    def fail_at_second_stage(U, **kwargs):
        if kwargs["c_l"] == 1:
            raise RuntimeError("RHS failed")
        return np.zeros_like(U)

    monkeypatch.setattr(sim.scheme, "compute_update", fail_at_second_stage)
    with pytest.raises(RuntimeError, match="RHS failed"):
        sim.integrator.update(sim.scheme)
    assert sim.time == 0.3
