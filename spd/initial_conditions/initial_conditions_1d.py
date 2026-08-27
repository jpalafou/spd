import numpy as np

def step_function(xy: np.ndarray,case: int, vx=1, P=1):
    x=xy[0]
    if case==0:
        #density
        return np.where(np.fabs(x-0.5)<0.25,2,1)
    elif case==1:
        #vx
        return vx*np.ones(x.shape)
    elif case==4:
        #Pressure
        return P*np.ones(x.shape)
    else:
        return np.zeros(x.shape)
    
def sine_wave(xy: np.ndarray,case: int, A=0.125, vx=1, P=1):
    x=xy[0]
    if case==0:
        #density
        return 1.0+A*(np.sin(2*np.pi*x))
    elif case==1:
        #vx
        return vx*np.ones(x.shape)
    elif case==2:
        #Pressure
        return P*np.ones(x.shape)
    else:
        return np.zeros(x.shape)
    
def sod_shock_tube(x: np.ndarray, case: int) -> np.ndarray:
    x=x[0]
    if case==0:
        #density
        return np.where(x < 0.5, 1., 0.125)
    elif case==2:
        #pressure
        return np.where(x < 0.5, 1., 0.1)
    else:
        #velocities
        return np.zeros(x.shape)

def decaying_isotropic_turbulence(
        x: np.ndarray,
        case: int,
        M: float = 10.0,
        slope: float = -4.0,
        seed: int = None,
        solenoidal: bool = True,
        fine_factor: int = 1,
        seed_fine: int = None,
        xlim: tuple = (0.0, 1.0),
) -> np.ndarray:
    x = x[0]

    if case == 0 or case == 4:
        return np.ones_like(x, dtype=float)
    if case not in (1, 2, 3):
        return np.zeros(x.shape)
    if case != 1:
        return np.zeros_like(x, dtype=float)

    n_elements_total = x.shape[0]
    n_subcells = x.shape[1] if x.ndim > 1 else 1
    tail0 = (0,) * (x.ndim - 1)
    x_line = x[(slice(None),) + tail0]
    active = (x_line >= xlim[0]) & (x_line < xlim[1])
    if not np.any(active):
        active = np.ones(n_elements_total, dtype=bool)
    n_elements = int(np.count_nonzero(active))
    n = n_elements * n_subcells
    offset = int(np.argmax(active)) * n_subcells

    if fine_factor < 1 or n % fine_factor != 0:
        raise ValueError("fine_factor must be a positive divisor of the number of cells.")

    h = (xlim[1] - xlim[0]) / n if n > 1 else 1.0

    def spectral_vx(n, h, this_seed):
        rng = np.random.RandomState(None if this_seed is None else int(this_seed))
        k = np.abs(np.fft.fftfreq(n, d=h))
        env = np.zeros_like(k)
        nonzero = k > 0.0
        if np.any(nonzero):
            env[nonzero] = (k[nonzero] / k[nonzero].min()) ** (slope / 2)
        vx_hat = np.fft.fft(rng.standard_normal(n)) * env
        if solenoidal:
            vx_hat[...] = 0.0
        vx_hat[0] = 0.0
        return vx_hat

    if fine_factor > 1:
        n_coarse = n // fine_factor
        vx_coarse = np.fft.ifft(
            spectral_vx(n_coarse, h * fine_factor, seed)
        ).real
        vx_hat = np.fft.fft(np.kron(vx_coarse, np.ones(fine_factor)))

        seed_fine = seed if seed_fine is None else seed_fine
        k = np.abs(np.fft.fftfreq(n, d=h))
        k_coarse_nyquist = 1 / (2 * h * fine_factor)
        vx_hat += spectral_vx(n, h, seed_fine) * (k > k_coarse_nyquist)
    else:
        vx_hat = spectral_vx(n, h, seed)

    vx = np.fft.ifft(vx_hat).real
    u_rms = float(np.sqrt(np.mean(vx * vx)))
    if u_rms > 0.0:
        vx *= M / u_rms

    element_idx = np.arange(n_elements_total)
    if x.ndim > 1:
        shape = (n_elements_total, n_subcells) + (1,) * (x.ndim - 2)
        idx = (
            element_idx[:, None] * n_subcells
            + np.arange(n_subcells)[None, :]
            - offset
        ) % n
        idx = idx.reshape(shape)
    else:
        idx = (element_idx - offset) % n

    return np.broadcast_to(vx[idx], x.shape)
