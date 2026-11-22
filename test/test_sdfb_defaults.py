"""SD fallback uses the same numerical defaults as SuperFV's MOOD setup."""

import numpy as np
import pytest
from superfv import HydroSolver

from spd.fallback.trouble_detection import pad_check
from spd.sdfb_simulator import SPD_Simulator


@pytest.mark.parametrize("thermal", [False, True])
def test_rk3_defaults_match_superfv(thermal):
    fv = HydroSolver(p=3, use_MOOD=True, positivity_guard=False, nx=8)
    sd = SPD_Simulator(
        N=(2,), p=3, time_integrator="rk3", passives=["dye"],
        thdiffusion=thermal, init=False, use_cupy=False,
    )
    mood = fv.params.fv_scheme.mood_params
    fallback = sd.lo_scheme
    assert fallback.slope_limiter.limiter == mood.fallback_cascade[0].muscl_params.MUSCL_limiter
    assert fallback.blending == mood.blend_troubles
    assert fallback.tolerance == mood.NAD_params.rtol
    assert fallback.SED == mood.NAD_params.SED_params.use_SED
    assert fallback.NAD_cfl == mood.NAD_params.scale_by_cfl
    assert fallback.min_rho == fv.params.hydro.rho_min
    assert fallback.min_P == fv.params.hydro.P_min
    assert np.isinf(fallback.max_rho)
    assert sd.nu == fv.params.hydro.nu
    assert sd.chi == fv.params.hydro.Chi
    assert fallback.limiting_variables == list(range(6))
    # PAD accepts states above SuperFV's floors without an arbitrary density cap.
    flags = pad_check(
        np.zeros(2), np.array([1e11, 1.]), np.array([1e-11, 1.]),
        fallback.min_rho, fallback.max_rho, fallback.min_P,
    )
    np.testing.assert_array_equal(flags, 0)


def constant(mesh, var):
    return np.full_like(mesh[0], 1. if var in (0, 4) else 0.)


@pytest.mark.parametrize("variable", [1, 2, 3, 5])
def test_default_nad_detects_velocity_and_passive_overshoots(variable):
    sd = SPD_Simulator(
        N=(2,), p=3, time_integrator="rk3", init_fct=constant,
        passives=["dye"], SED=False, use_cupy=False,
    )
    sd.ho_scheme.switch_to_finite_volume(integrate_faces=False)
    fallback = sd.lo_scheme
    fallback.working_arrays()
    candidate = fallback.W_cv.copy()
    candidate[variable, 3] = 0.1
    fallback.dm.U_new[...] = sd.compute_conservatives(candidate)
    fallback.detect_troubles(sd.dt)
    assert fallback.dm.troubles[0, 3] == 1
    assert fallback.dm.troubles.sum() == 1
    assert fallback.dm.theta.max() == 1
    assert np.count_nonzero(fallback.dm.theta) == 1
