"""Layout adapter for SuperFV's second-order diffusion fluxes."""

from superfv.finite_volume_driver import add_viscuous_fluxes

from spd.riemann_solvers.superfv_adapter import _variable_index_map


def add_superfv_diffusion(scheme, F):
    xp = scheme.dm.xp
    ngh = scheme.Nghc
    dims = tuple(scheme.dims)
    axes = (0, *range(scheme.ndim, 0, -1))

    # Refill the workspace: Hancock modified it with the half-step predictor,
    # whereas SuperFV evaluates diffusion from the original cell averages.
    W = scheme.compute_primitives(scheme._full_solution()) if scheme.WB else scheme.W_cv
    scheme.dm.M[...] = 0
    scheme.fill_active_region(W)
    scheme.Boundaries(scheme.dm.M)

    # SPD stores (var, z, y, x); SuperFV uses contiguous (var, x, y, z).
    w = scheme.dm.M.transpose(axes)
    w = xp.ascontiguousarray(w.reshape((*w.shape, *(1,) * (3 - scheme.ndim))))
    idx = _variable_index_map(0)
    spacing = [scheme.len[d] / scheme.N[d] if d in dims else 1.0 for d in "xyz"]
    nu = scheme.nu if scheme.viscosity else 0.0
    chi = scheme.chi if scheme.thdiffusion else 0.0

    for dim in dims:
        shape = [*w.shape, 1]
        shape[1 + scheme.dims[dim]] = scheme.N[dim] + 1
        flux = xp.zeros(shape)
        add_viscuous_fluxes(
            w, flux, idx, dim, dims, ngh, *spacing, 1, False, nu, chi, 0.0,
        )
        # SPD diffuses each passive concentration with the same nu as momentum.
        for i in range(scheme.npassive):
            dye_idx = idx.copy()
            dye_idx.add_var("dye", 5 + i)
            add_viscuous_fluxes(
                w, flux, dye_idx, dim, dims, ngh, *spacing, 1, False, 0.0, 0.0, nu,
            )
        interior = (slice(None),) + tuple(
            slice(ngh, -ngh) if d in dims and d != dim else slice(None)
            for d in "xyz"
        ) + (0,)
        values = flux[interior].reshape((scheme.nvar, *F[dim].shape[:0:-1]))
        F[dim][...] += values.transpose(axes)
