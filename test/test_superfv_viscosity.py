"""MUSCL-Hancock with optional SuperFV diffusion should agree to roundoff."""

import numpy as np
import pytest
from superfv import HydroSolver

from spd.finite_volume.fv_simulator import FV_Simulator
from spd.finite_volume.superfv_viscosity import add_superfv_diffusion
from spd.sdfb_simulator import SPD_Simulator


@pytest.mark.parametrize("N", [(8,), (8, 6), (8, 6, 4)])
@pytest.mark.parametrize("limiter", ["minmod", "moncen"])
@pytest.mark.parametrize("solver", ["hllc", "llf"])
@pytest.mark.parametrize("driver", ["fv", "spd"])
@pytest.mark.parametrize("backend", ["numpy", "cupy"])
@pytest.mark.parametrize("thermal", [False, True])
def test_viscous_muscl_hancock_matches_superfv(
    N, limiter, solver, driver, backend, thermal,
):
    xp = np if backend == "numpy" else pytest.importorskip("cupy")
    cls = FV_Simulator if driver == "fv" else SPD_Simulator
    sd = cls(
        N=N, p=1, time_integrator="rk1", viscosity=True, nu=0.03,
        scheme="MUSCL-Hancock" if driver == "fv" else "FV",
        thdiffusion=thermal, chi=0.02,
        superfv_viscosity=True, slope_limiter=limiter,
        riemann_solver_fv=solver, cfl_coeff=0.3, use_cupy=backend == "cupy",
    )
    fv = HydroSolver(
        nx=N[0], ny=N[1] if len(N) > 1 else 1, nz=N[2] if len(N) > 2 else 1,
        p=1, use_MUSCL=True, use_SED=False, MUSCL_limiter=limiter,
        riemann_solver=solver, nu=0.03, CFL=0.3, cupy=backend == "cupy",
        Chi=0.02 if thermal else 0.0,
    )
    sd.init_sim()
    # Use identical cell averages, independent of the IC quadrature conventions.
    rng = np.random.default_rng(42)
    W = xp.asarray(rng.uniform(0.8, 1.2, sd.dm.W_cv.shape))
    W[1:4] -= 1.0
    sd.compute_conservatives(W, U=sd.dm.U_cv)
    sd.compute_primitives(sd.dm.U_cv, W=sd.dm.W_cv)
    axes = (0, *range(len(N), 0, -1))

    def superfv_layout(U):
        # SPD also carries a diagnostic temperature row when conduction is on.
        out = U[:fv.arrays["u"].shape[0]].transpose(axes)
        return out.reshape((*out.shape, *(1,) * (3 - len(N))))

    fv.arrays["u"][...] = superfv_layout(sd.dm.U_cv)
    # The codes use different CFL estimates on anisotropic meshes. Prescribe
    # the same dt to compare their spatial operators and Hancock updates.
    dt = 1e-3
    for _ in range(5):
        fv.arrays["u"][...] += dt * fv.f(fv.t, fv.arrays["u"], dt, hancock=True)
        fv.t += dt
        sd.dt = sd.scheme.dt = dt
        sd.perform_update()
        to_numpy = np.asarray if backend == "numpy" else xp.asnumpy
        np.testing.assert_allclose(
            to_numpy(superfv_layout(sd.dm.U_cv)), to_numpy(fv.arrays["u"]),
            rtol=8 * np.finfo(float).eps, atol=8 * np.finfo(float).eps,
        )


def test_passive_diffusion():
    sim = FV_Simulator(
        N=(8,), time_integrator="rk1", viscosity=True, nu=0.03,
        superfv_viscosity=True, passives=["dye", "tracer"], use_cupy=False,
    )
    sim.init_sim()
    sim.dm.W_cv[...] = 0.0
    sim.dm.W_cv[[0, 4]] = 1.0
    sim.dm.W_cv[5] = (np.arange(8) + 0.5) / 8
    sim.dm.W_cv[6] = 2 * sim.dm.W_cv[5]
    flux = {"x": np.zeros_like(sim.scheme.F_fp["x"])}
    add_superfv_diffusion(sim.scheme, flux)
    np.testing.assert_array_equal(flux["x"][:5], 0.0)
    np.testing.assert_allclose(flux["x"][5, 1:-1], -0.03)
    np.testing.assert_allclose(flux["x"][6, 1:-1], -0.06)


@pytest.mark.parametrize("scheme", ["SD", "SDFB", "FVFB"])
def test_rejects_non_pure_fv(scheme):
    with pytest.raises(ValueError, match="pure FV"):
        SPD_Simulator(scheme=scheme, superfv_viscosity=True, time_integrator="rk1")


@pytest.mark.parametrize("scheme, integrator", [
    ("MUSCL", "rk1"), ("MUSCL-Hancock", "rk2"), ("MUSCL-Hancock", "ader"),
])
def test_rejects_non_hancock_updates(scheme, integrator):
    with pytest.raises(ValueError, match="MUSCL-Hancock"):
        FV_Simulator(scheme=scheme, time_integrator=integrator, superfv_viscosity=True)
