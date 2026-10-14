# spd
Spectral Difference method with a-posteriori limiting.

Hydrodynamic HLLC and LLF fluxes use SuperFV's NumPy/CuPy kernels. For a
local checkout of SuperFV alongside this repository, install both with:

```bash
python -m pip install -e ../superfv -e .
```

These kernels use SuperFV's sound-speed treatment rather than SPD's `min_c2` floor.

## Documentation

Build Sphinx docs from repository root:

```bash
python -m pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```
