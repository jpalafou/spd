import numpy as np

# ----------------------------------------------------------------------
# Double Mach Reflection (Woodward & Colella 1984)
# ----------------------------------------------------------------------
# A Mach-10 shock in gamma=1.4 air meets a reflecting wall.  The shock
# starts at x = DMR_XC on the bottom boundary, inclined at DMR_ANGLE to
# the wall.  These constants are shared by the initial condition and the
# "doublemach" boundary conditions.
DMR_XC = 1.0 / 6.0           # shock foot on the bottom wall
DMR_ANGLE = np.pi / 3.0      # 60 degrees from the x-axis
DMR_SHOCK_SPEED = 10.0       # shock speed factor used for the moving trace


def dmr_post_shock(gamma=1.4):
    """Post-shock primitive state behind the Mach-10 shock."""
    return dict(
        rho=8.0,
        vx=8.25 * np.cos(np.pi / 6),   # 8.25 * cos(30 deg) ~ 7.1447
        vy=-8.25 * np.sin(np.pi / 6),  # -8.25 * sin(30 deg) = -4.125
        P=116.5,
    )


def dmr_ambient(gamma=1.4):
    """Undisturbed (ahead-of-shock) primitive state."""
    return dict(rho=1.4, vx=0.0, vy=0.0, P=1.0)


def dmr_shock_x(t, y, xc=DMR_XC, angle=DMR_ANGLE, speed=DMR_SHOCK_SPEED):
    """x-position of the (tilted, moving) shock front at height ``y``/time ``t``."""
    return speed * t / np.sin(angle) + xc + y / np.tan(angle)


def double_mach_reflection(
    xy: np.ndarray, case: int, gamma=1.4
) -> np.ndarray:
    """Initial condition for the Double Mach Reflection problem.

    Behind the initial shock line ``x < xc + y/tan(angle)`` the flow is in
    the post-shock state; ahead of it the flow is at rest (ambient).
    Recommended domain: ``[0, 4] x [0, 1]``.
    """
    x = xy[0]
    y = xy[1]
    behind = x < (DMR_XC + y / np.tan(DMR_ANGLE))
    ps = dmr_post_shock(gamma)
    am = dmr_ambient(gamma)
    if case == 0:
        return np.where(behind, ps["rho"], am["rho"])
    elif case == 1:
        return np.where(behind, ps["vx"], am["vx"])
    elif case == 2:
        return np.where(behind, ps["vy"], am["vy"])
    elif case == 4:
        return np.where(behind, ps["P"], am["P"])
    else:
        return np.zeros(x.shape)


def step_function(xy: np.ndarray,case: int, vx=1, vy=1, P=1):
    x=xy[0]
    y=xy[1]
    if case==0:
        #density
        return np.where(np.fabs(x-0.5)<0.25,
                        np.where(np.fabs(y-0.5)<0.25,2,1),1)
    elif case==1:
        #vx
        return vx*np.ones(x.shape)
    elif case==2:
        #vy
        return vy*np.ones(x.shape)
    elif case==4:
        #Pressure
        return P*np.ones(x.shape)
    else:
        return np.zeros(x.shape)
    
def sine_wave(xy: np.ndarray,case: int, A=0.125, vx=1, vy=1, P=1):
    x=xy[0]
    y=xy[1]
    if case==0:
        #density
        return 1.0+A*(np.sin(2*np.pi*(x+y)))
    elif case==1:
        #vx
        return vx*np.ones(x.shape)
    elif case==2:
        #vy
        return vy*np.ones(x.shape)
    elif case==3:
        #Pressure
        return P*np.ones(x.shape)
    else:
        return np.zeros(x.shape)


def decaying_isotropic_turbulence(
    xy: np.ndarray,
    case: int,
    M: float = 10.0,
    slope: float = -4.0,
    seed: int = None,
    solenoidal: bool = True,
    fine_factor: int = 1,
    seed_fine: int = None,
    xlim: tuple = (0.0, 1.0),
    ylim: tuple = (0.0, 1.0),
) -> np.ndarray:
    x = xy[0]
    y = xy[1]

    if case == 0 or case == 4:
        return np.ones_like(x, dtype=float)
    if case not in (1, 2, 3):
        return np.zeros(x.shape)
    if case == 3:
        return np.zeros_like(x, dtype=float)

    if x.ndim >= 6:
        n_y_total, n_x_total, n_y_sub, n_x_sub = x.shape[:4]
        tail0 = (0,) * (x.ndim - 4)
        x_line = x[(0, slice(None), 0, 0) + tail0]
        y_line = y[(slice(None), 0, 0, 0) + tail0]
    else:
        n_y_total, n_x_total = x.shape[:2]
        n_y_sub = n_x_sub = 1
        tail0 = (0,) * (x.ndim - 2)
        x_line = x[(0, slice(None)) + tail0]
        y_line = y[(slice(None), 0) + tail0]

    active_x = (x_line >= xlim[0]) & (x_line < xlim[1])
    active_y = (y_line >= ylim[0]) & (y_line < ylim[1])
    if not np.any(active_x):
        active_x = np.ones(n_x_total, dtype=bool)
    if not np.any(active_y):
        active_y = np.ones(n_y_total, dtype=bool)

    n_x = int(np.count_nonzero(active_x)) * n_x_sub
    n_y = int(np.count_nonzero(active_y)) * n_y_sub
    offset_x = int(np.argmax(active_x)) * n_x_sub
    offset_y = int(np.argmax(active_y)) * n_y_sub

    if fine_factor < 1 or n_x % fine_factor != 0 or n_y % fine_factor != 0:
        raise ValueError("fine_factor must be a positive divisor of the number of cells.")

    hx = (xlim[1] - xlim[0]) / n_x if n_x > 1 else 1.0
    hy = (ylim[1] - ylim[0]) / n_y if n_y > 1 else 1.0

    def spectral_velocities(shape, spacings, this_seed):
        rng = np.random.RandomState(None if this_seed is None else int(this_seed))
        kx = np.fft.fftfreq(shape[0], d=spacings[0])
        ky = np.fft.fftfreq(shape[1], d=spacings[1])
        KX = kx[:, None]
        KY = ky[None, :]
        K2 = KX * KX + KY * KY
        K = np.sqrt(K2)

        env = np.zeros_like(K)
        nonzero = K > 0.0
        if np.any(nonzero):
            env[nonzero] = (K[nonzero] / K[nonzero].min()) ** ((slope - 1.0) / 2.0)

        Vx = np.fft.fftn(rng.standard_normal(shape)) * env
        Vy = np.fft.fftn(rng.standard_normal(shape)) * env

        if solenoidal:
            invK2 = np.divide(1.0, K2, out=np.zeros_like(K2), where=K2 > 0.0)
            kdotV = KX * Vx + KY * Vy
            Vx -= KX * kdotV * invK2
            Vy -= KY * kdotV * invK2

        Vx[0, 0] = 0.0
        Vy[0, 0] = 0.0
        return Vx, Vy

    if fine_factor > 1:
        seed_fine = seed if seed_fine is None else seed_fine
        coarse_shape = (n_x // fine_factor, n_y // fine_factor)
        Vx_c, Vy_c = spectral_velocities(coarse_shape, (hx * fine_factor, hy * fine_factor), seed)
        vx_c = np.fft.ifftn(Vx_c).real
        vy_c = np.fft.ifftn(Vy_c).real

        ones = np.ones((fine_factor, fine_factor), dtype=float)
        Vx = np.fft.fftn(np.kron(vx_c, ones))
        Vy = np.fft.fftn(np.kron(vy_c, ones))

        Vx_f, Vy_f = spectral_velocities((n_x, n_y), (hx, hy), seed_fine)
        kx = np.fft.fftfreq(n_x, d=hx)
        ky = np.fft.fftfreq(n_y, d=hy)
        K = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
        high_k_mask = K > 1.0 / (2.0 * max(hx, hy) * fine_factor)
        Vx += Vx_f * high_k_mask
        Vy += Vy_f * high_k_mask
    else:
        Vx, Vy = spectral_velocities((n_x, n_y), (hx, hy), seed)

    vx = np.fft.ifftn(Vx).real
    vy = np.fft.ifftn(Vy).real
    u_rms = float(np.sqrt(np.mean(vx * vx + vy * vy)))
    if u_rms > 0.0:
        vx *= M / u_rms
        vy *= M / u_rms

    if x.ndim >= 6:
        y_idx = (
            np.arange(n_y_total)[:, None, None, None] * n_y_sub
            + np.arange(n_y_sub)[None, None, :, None]
            - offset_y
        ) % n_y
        x_idx = (
            np.arange(n_x_total)[None, :, None, None] * n_x_sub
            + np.arange(n_x_sub)[None, None, None, :]
            - offset_x
        ) % n_x
        out = (vx if case == 1 else vy)[x_idx, y_idx]
        return np.broadcast_to(out[(Ellipsis,) + (None,) * (x.ndim - 4)], x.shape)

    y_idx = (np.arange(n_y_total)[:, None] - offset_y) % n_y
    x_idx = (np.arange(n_x_total)[None, :] - offset_x) % n_x
    out = (vx if case == 1 else vy)[x_idx, y_idx]
    return np.broadcast_to(out[(Ellipsis,) + (None,) * (x.ndim - 2)], x.shape)

def RTI(
    xy: np.ndarray,
    case: int,
    P0=1.0,
    gamma=5 / 3,
    g=-1.0,
    rho1=2.0,
    rho2=1.0,
    yc=0.5,
) -> np.ndarray:
    """Rayleigh-Taylor instability (single-mode, hydrostatic background).

    A heavy fluid ``rho1`` sits on top of a light fluid ``rho2`` at the
    interface ``y = yc`` in a constant downward gravitational field, with a
    small single-mode vertical velocity perturbation seeded at the interface.

    The gravitational potential is returned for ``case == -1`` so the same
    function can be passed as ``init_fct`` with ``potential=True``.
    """
    x = xy[0]
    y = xy[1]
    if case == 0:
        # density
        return np.where(y > yc, rho2, rho1)
    elif case == 1:
        # vx
        return np.zeros(x.shape)
    elif case == 2:
        # vy: single-mode perturbation scaled by the local sound speed
        dv = np.sqrt(gamma * (P0 + rho1 * yc - P0 + 1) / rho1)
        return -0.025 * dv * np.cos(8 * np.pi * x)
    elif case == 4:
        # Pressure (hydrostatic equilibrium: dP/dy = rho * g, with g = -1)
        return np.where(
            y > yc, P0 + rho2 * y + (rho1 - rho2) * yc, P0 + rho1 * y
        )
    elif case == -1:
        # Gravitational potential phi (acceleration g_y = -dphi/dy = g)
        return g * y
    else:
        return np.zeros(x.shape)


def KH_instability(
    xy: np.ndarray,
    case: int,
    density_jump=1.0,
    a=0.05,
    sigma=0.2,
    u_flow=1.0,
    A=0.01,
    P0=10.0,
    z1=0.5,
    z2=1.5,
) -> np.ndarray:
    x = xy[0]
    z = xy[1]
    tanh1 = np.tanh((z - z1) / a)
    tanh2 = np.tanh((z - z2) / a)
    if case == 0:
        return 1.0 + 0.5 * density_jump * (tanh1 - tanh2)
    elif case == 1:
        return u_flow * (tanh1 - tanh2 - 1.0)
    elif case == 2:
        return A * np.sin(2 * np.pi * x) * (
            np.exp(-((z - z1) ** 2) / sigma**2)
            + np.exp(-((z - z2) ** 2) / sigma**2)
        )
    elif case == 4:
        return P0 * np.ones(z.shape)
    elif case == 5:
        return 0.5 * (tanh2 - tanh1 + 2.0)
    else:
        return np.zeros(x.shape)


def gresho_vortex(
    xy: np.ndarray, case: int, gamma=5 / 3, M_max=0.1, v0=0.0
) -> np.ndarray:
    """Gresho vortex initial condition on ``[0, 1] x [0, 1]``."""
    x = xy[0]
    y = xy[1]

    xc = x - 0.5
    yc = y - 0.5
    r = np.sqrt(xc**2 + yc**2)

    zone1 = r < 0.2
    zone2 = np.logical_and(r >= 0.2, r < 0.4)
    zone3 = r >= 0.4

    v_phi = np.zeros(x.shape)
    v_phi[zone1] = 5.0 * r[zone1]
    v_phi[zone2] = 2.0 - 5.0 * r[zone2]

    inv_r = np.divide(1.0, r, out=np.zeros(x.shape), where=r != 0.0)
    vx = -v_phi * yc * inv_r + v0
    vy = v_phi * xc * inv_r

    P = np.empty(x.shape)
    P[zone1] = (25 / 2) * r[zone1] ** 2
    P[zone2] = (
        4 * np.log(5 * r[zone2])
        + 4
        - 20 * r[zone2]
        + (25 / 2) * r[zone2] ** 2
    )
    P[zone3] = 4 * np.log(2) - 2.0
    P = (1 / (gamma * M_max**2)) - 0.5 + P

    if case == 0:
        return np.ones(x.shape)
    elif case == 1:
        return vx
    elif case == 2:
        return vy
    elif case == 4:
        return P
    else:
        return np.zeros(x.shape)
