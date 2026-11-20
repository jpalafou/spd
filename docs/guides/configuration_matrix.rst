Configuration Matrix
====================

The framework combines three mostly independent axes:

- Spatial discretization: SD or FV
- Time integration: ADER or RK (including induction RK)
- Equations set: Hydro, Induction, or MHD

Common patterns
---------------

- ``spectral_difference.sd_simulator.SD_Simulator``
  - High-order SD runs with ADER (default) or RK.
- ``finite_volume.fv_simulator.FV_Simulator``
  - FV runs with MUSCL/MUSCL-Hancock options and RK/ADER integration flow.
- ``sdfb_simulator.SPD_Simulator``
  - SD primary plus FV fallback blending in troubled regions.
- ``induction.induction_simulator.InductionSimulator``
  - Induction-focused simulator selecting SD/FV induction schemes.
- ``MHD.mhd_spd_simulator.MHD_SPD_Simulator``
  - MHD-coupled simulator using the same modular architecture.

Key constructor knobs
---------------------

- ``scheme`` / ``scheme_fb``: choose SD/FV and fallback-enabled variants.
- ``time_integrator``: choose ``ader`` or ``rkN`` family.
- ``soe``: select equations path (hydro/induction/mhd where applicable).
- ``N``, ``p``, ``m``: mesh and order controls.
- ``BC``: per-dimension boundary-condition tuples.
- ``NAD_cfl=True`` on ``SPD_Simulator``: multiply the tolerance for either
  ``NAD=""`` or ``NAD="delta"`` by ``min(1, v_max * dt / h)``. Disabled
  by default. The speed is evaluated in each cell as
  ``v_max = max(abs(vx), abs(vy), abs(vz)) + c_s``. ``h`` is each cell's
  maximum physical width over active directions, and ``dt`` is the candidate
  update's timestep (``dt * w_tp[i]`` for each ADER contribution).
  Enabling this option with ``WB=True`` raises ``NotImplementedError``.

Advection and diffusion numbers
------------------------------

Both numbers use the directional average subcell width ``h_ave``: element width
divided by ``p + 1`` for SD, or cell width for pure FV, including with fallback.
For the one-dimensional scalar advection-diffusion stability analysis,
``C = abs(a) * dt / h_ave`` and ``D = nu * dt / h_ave**2``.

In hydrodynamics, the advective speed is ``abs(v) + c_s``. Multiple dimensions
use ``C = dt * max(sum((abs(v_d) + c_s) / h_ave_d))`` and
``D = diffusivity * dt * sum(1 / h_ave_d**2)`` over active directions.
The diffusion coefficient is ``nu`` for viscosity and passive transport, or
``(gamma - 1) * chi`` for thermal diffusion; the largest active coefficient
controls the diffusion bound.

``cfl_coeff`` limits C and ``cfl_diffusion`` independently limits D, matching
SuperFV's ``CFL`` and ``CFL_diffusion`` arguments. They default to ``0.8`` and
``0.2``. The timestep satisfies both bounds and may be clipped to reach a target
time, so the achieved numbers can be smaller than the requested limits.
The stable C and D depend on the spatial scheme, degree, physics, and integrator;
the defaults do not encode the degree-dependent SD stability boundaries.

Previously, SD with fallback normalized ``cfl_coeff`` by the smallest subcell
width. To retain the same inviscid timestep, multiply an old coefficient by
``h_smallest / h_ave``. For ``p=7``, an old value of ``2.4`` becomes approximately
``0.489`` on an isotropic mesh. Diffusion is now controlled by ``cfl_diffusion``
in average-width units, rather than by ``cfl_coeff`` and a smallest-width cap.

Operational note
----------------

Fallback mode is intended to preserve robustness while keeping high-order fluxes
where smoothness criteria pass. Trouble detection and blending are implemented in
``schemes.fallback.FallbackScheme`` and ``trouble_detection`` helpers.
