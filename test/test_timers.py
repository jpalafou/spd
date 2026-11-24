"""Profiling categories and call counts match the jonathan branch."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from superfv.tools import step_history

from spd.sdfb_simulator import SPD_Simulator
from spd.simulator import Simulator


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("viscosity", [False, True])
def test_rk_timers_and_numerical_equivalence(fallback, viscosity):
    results = []
    for profile in (False, True):
        sim = SPD_Simulator(
            N=(4, 4), p=3, time_integrator="rk3", use_cupy=False,
            scheme="SDFB" if fallback else "SD", viscosity=viscosity,
            nu=1e-5, profile=profile, verbose=False,
        )
        sim.perform_iterations(2)
        results.append(sim.dm.U_cv.copy())
        assert len(sim.trouble_counts) == 6 * fallback
        assert len(sim.step_times) == 2
        assert len(sim.limiting_fractions) == 2
        assert all(0 <= f <= 1 for f in sim.limiting_fractions)
        if not fallback:
            assert sim.limiting_fractions == [0.0, 0.0]
        assert sim.step_times[-1] == sim.time
        # Reference counts from two RK3 steps on jonathan, including cleanup.
        expected = dict(
            take_step=2, compute_dt=2, update_unew=2, f=6, compute_dudt=6,
            primitive_conservative=50 if viscosity else 38,
            boundary_conditions=48 if viscosity else 12,
            einsum=62 if viscosity else 26, transpose=36 if fallback else 0,
            riemann_solver=12,
            **{cat: 6 if fallback else 0 for cat in (
                "mood_loop", "candidate_solution", "detect_troubles",
                "fallback_fluxes", "assign_fluxes",
            )},
        )
        assert sim.timer_categories == list(expected)
        for category, count in expected.items():
            timer = sim.timer[category]
            assert timer.n_calls == (count if profile or category == "take_step" else 0)
            assert not timer.is_timing
        sim.init_sim()
        assert all(sim.timer[cat].n_calls == 0 for cat in expected)
    np.testing.assert_array_equal(*results)


@pytest.mark.parametrize("profile", [False, True])
def test_cuda_sync_and_profile_gating(monkeypatch, profile):
    synchronize = Mock()
    device = SimpleNamespace(synchronize=synchronize)
    monkeypatch.setattr(step_history, "CUPY_AVAILABLE", True)
    monkeypatch.setattr(
        step_history, "cp",
        SimpleNamespace(cuda=SimpleNamespace(Device=lambda: device)), raising=False,
    )
    sim = Simulator(use_cupy=True, profile=profile, init=False)
    for category in sim.timer_categories:
        with sim._time(category):
            pass
        assert sim.timer[category].n_calls == int(profile or category == "take_step")
    calls = len(sim.timer_categories) if profile else 1
    assert synchronize.call_count == 2 * calls
    with pytest.raises(RuntimeError), sim._time("take_step"):
        raise RuntimeError("interrupted step")
    assert not sim.timer["take_step"].is_timing
    assert synchronize.call_count == 2 * (calls + 1)


def test_output_round_trips_histories(tmp_path):
    sim = SPD_Simulator(
        N=(2,), p=1, time_integrator="rk3", use_cupy=False,
        scheme="SDFB", verbose=False, folder=str(tmp_path),
    )
    sim.trouble_counts = [1, 2, 3]
    sim.step_times = [0.1]
    sim.limiting_fractions = [0.125]
    sim.output()
    sim.trouble_counts = []
    sim.step_times = []
    sim.limiting_fractions = []
    sim.load_output()
    assert sim.trouble_counts == [1, 2, 3]
    assert sim.step_times == [0.1]
    assert sim.limiting_fractions == [0.125]
    (tmp_path / "limiting_fractions.npy").unlink()
    sim.load_output()
    assert np.isnan(sim.limiting_fractions[0])


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("time_integrator", ["rk3", "ader"])
def test_limiting_fraction_weights_volume_and_stages(monkeypatch, ndim, time_integrator):
    sim = SPD_Simulator(
        N=(2,) * ndim, p=3, time_integrator=time_integrator, use_cupy=False,
        SED=False, PAD=False, blending=True, verbose=False,
        xlim=(0, 2), ylim=(0, 3), zlim=(0, 4),
    )
    calls = 0

    def flag_first_subcell(W_new, *args):
        nonlocal calls
        trouble = np.zeros_like(W_new)
        if calls == 0:
            trouble[(0,) * (ndim + 1)] = 1
        calls += 1
        return trouble

    monkeypatch.setattr("spd.fallback.trouble_detection.nad_check", flag_first_subcell)
    weights = sim.ho_scheme.dm.w_tp if sim.ader else sim.integrator.b
    # For p=3, the first flux interval on [0, 1] has this exact width.
    expected = ((1 - np.sqrt(3 / 5)) / 4) ** ndim * weights[0]
    sim.perform_iterations(1)
    assert sim.trouble_counts == [1] + [0] * (calls - 1)
    assert sim.limiting_fractions == pytest.approx([expected])
    assert len(sim.limiting_fractions) == len(sim.step_times)
