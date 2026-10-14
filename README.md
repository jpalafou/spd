# spd
Spectral Difference method with a-posteriori limiting.

Hydrodynamic HLLC and LLF fluxes use SuperFV's NumPy/CuPy kernels. For a
local checkout of SuperFV alongside this repository, install both with:

```bash
python -m pip install -e ../superfv -e .
```

These kernels use SuperFV's sound-speed treatment rather than SPD's `min_c2` floor.

For viscous MUSCL-Hancock comparisons on a uniform FV mesh, opt in to
SuperFV's second-order diffusion fluxes:

```python
from spd.sdfb_simulator import SPD_Simulator

sim = SPD_Simulator(
    scheme="FV", time_integrator="rk1", superfv_viscosity=True,
    viscosity=True, nu=1e-3, slope_limiter="moncen", riemann_solver_fv="hllc",
)
```

The option is also available on `FV_Simulator` with `scheme="MUSCL-Hancock"`
and `time_integrator="rk1"`. It is disabled by default and rejects SD,
fallback blending, and multistage time integration. Diffusion uses the
cell-centered state before the Hancock predictor. Match initial cell averages,
boundary conditions, limiter, Riemann solver, and timesteps when comparing codes.
Enabled thermal diffusion follows the same path; passive diffusivity remains `nu`.

## Documentation

Build Sphinx docs from repository root:

```bash
python -m pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```
