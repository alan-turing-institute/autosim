import numpy as np
import torch
from numpy.fft import fft2, ifft2

from autosim.simulations.base import SpatioTemporalSimulator
from autosim.types import NumpyLike, TensorLike

PATTERN_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "gliders": {"F": (0.013, 0.015), "k": (0.053, 0.055)},
    "bubbles": {"F": (0.097, 0.099), "k": (0.056, 0.058)},
    "maze": {"F": (0.028, 0.030), "k": (0.056, 0.058)},
    "worms": {"F": (0.057, 0.059), "k": (0.064, 0.066)},
    "spirals": {"F": (0.017, 0.019), "k": (0.050, 0.052)},
    "spots": {"F": (0.029, 0.031), "k": (0.061, 0.063)},
}
PATTERN_FIXED_PARAMS: dict[str, dict[str, float]] = {
    "gliders": {"F": 0.014, "k": 0.054},
    "bubbles": {"F": 0.098, "k": 0.057},
    "maze": {"F": 0.029, "k": 0.057},
    "worms": {"F": 0.058, "k": 0.065},
    "spirals": {"F": 0.018, "k": 0.051},
    "spots": {"F": 0.030, "k": 0.062},
}

DEFAULT_DIFFUSIONS = (2.0e-5, 1.0e-5)
DEFAULT_GAUSSIAN_SPEC: dict[str, tuple[float, float]] = {
    "count": (10, 100),
    "amplitude": (1.0, 3.0),
    "width": (150.0, 300.0),
}
VALID_INIT_TYPES = {"gaussians", "fourier", "mixed"}


def _validate_gaussian_spec(
    spec: dict[str, tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    validated: dict[str, tuple[float, float]] = {}
    required = ("count", "amplitude", "width")
    for key in required:
        if key not in spec:
            raise ValueError(
                f"Gaussian specification missing '{key}'. Expected keys: {required}."
            )
        lo, hi = spec[key]
        if hi < lo:
            raise ValueError(
                f"Gaussian spec for '{key}' must satisfy min <= max, got {(lo, hi)}."
            )
        validated[key] = (float(lo), float(hi))
    return validated


def _compute_snapshot_count(total_time: float, dt: float, snapshot_dt: float) -> int:
    num_steps = max(1, int(np.ceil(total_time / dt)))
    stride = max(1, int(np.round(snapshot_dt / dt)))
    count = 1 + num_steps // stride
    if num_steps % stride != 0:
        count += 1
    return count


def _normalize_field(field: np.ndarray) -> np.ndarray:
    fmin = float(np.min(field))
    fmax = float(np.max(field))
    if fmax - fmin < 1e-12:
        return np.zeros_like(field)
    return (field - fmin) / (fmax - fmin)


def _random_gaussians_field(
    rng: np.random.Generator,
    X: np.ndarray,
    Y: np.ndarray,
    domain: tuple[float, float, float, float],
    spec: dict[str, tuple[float, float]],
) -> np.ndarray:
    count_low, count_high = spec["count"]
    amp_low, amp_high = spec["amplitude"]
    width_low, width_high = spec["width"]
    m = int(rng.integers(int(count_low), int(count_high) + 1))

    amplitudes = rng.uniform(amp_low, amp_high, size=m)
    widths = rng.uniform(width_low, width_high, size=m)
    cx = rng.uniform(domain[0], domain[1], size=m)
    cy = rng.uniform(domain[2], domain[3], size=m)

    dx = domain[1] - domain[0]
    dy = domain[3] - domain[2]
    shifts_x = np.array([-dx, -dx, -dx, 0.0, 0.0, 0.0, dx, dx, dx])
    shifts_y = np.array([dy, 0.0, -dy, dy, 0.0, -dy, dy, 0.0, -dy])

    amplitudes = np.repeat(amplitudes, 9)
    widths = np.repeat(widths, 9)
    cx = np.repeat(cx, 9) + np.tile(shifts_x, m)
    cy = np.repeat(cy, 9) + np.tile(shifts_y, m)

    field = np.zeros_like(X, dtype=np.float64)
    for a, w, x0, y0 in zip(amplitudes, widths, cx, cy, strict=True):
        field += a * np.exp(-w * ((X - x0) ** 2 + (Y - y0) ** 2))
    return _normalize_field(field)


def _trig_interpolate_to_grid(values: np.ndarray, target_n: int) -> np.ndarray:
    source_n = values.shape[0]
    if source_n == target_n:
        return values.astype(np.float64)

    if source_n > target_n:
        msg = (
            f"Source Fourier grid ({source_n}) cannot exceed target grid ({target_n})."
        )
        raise ValueError(msg)

    source_hat = np.fft.fftshift(fft2(values))
    target_hat = np.zeros((target_n, target_n), dtype=np.complex128)
    start = (target_n - source_n) // 2
    end = start + source_n
    target_hat[start:end, start:end] = source_hat

    upsampled = np.real(ifft2(np.fft.ifftshift(target_hat)))
    scale = (target_n / source_n) ** 2
    return (scale * upsampled).astype(np.float64)


def _fourier_random_field(
    rng: np.random.Generator,
    n: int,
    n_modes: int,
) -> np.ndarray:
    m = max(2, min(int(n_modes), n))
    coarse = rng.random((m, m)) - 0.5
    return _trig_interpolate_to_grid(coarse, n)


def _steady_state(F: float, k: float) -> tuple[float, float, float, float]:
    mean_u = 1.0
    mean_v = 0.0
    du = 1.0
    dv = 1.0
    threshold = (np.sqrt(F) - 2.0 * F) / 2.0
    if k < threshold:
        A = np.sqrt(F) / (F + k)
        rad = np.sqrt(max(A**2 - 4.0, 0.0))
        mean_u = float((A - rad) / (2.0 * A))
        mean_v = float(np.sqrt(F) * (A + rad) / 2.0)
        du = 0.9 * min(mean_u, 1.0 - mean_u)
        dv = 0.9 * min(mean_v, 1.0 - mean_v)
    return mean_u, mean_v, du, dv


def _initialize_fields(
    rng: np.random.Generator,
    n: int,
    X: np.ndarray,
    Y: np.ndarray,
    domain: tuple[float, float, float, float],
    F: float,
    k: float,
    init_type: str,
    gaussian_spec: dict[str, tuple[float, float]],
    n_fourier_modes: int,
) -> tuple[np.ndarray, np.ndarray]:
    choice = init_type
    if init_type == "mixed":
        choice = rng.choice(["gaussians", "fourier"])  # type: ignore[arg-type]

    if choice == "gaussians":
        base_u = _random_gaussians_field(rng, X, Y, domain, gaussian_spec)
        base_v = _random_gaussians_field(rng, X, Y, domain, gaussian_spec)
        u0 = 1.0 - base_u
        v0 = base_v
    elif choice == "fourier":
        mean_u, mean_v, du, dv = _steady_state(F, k)
        perturb_u = _fourier_random_field(rng, n, n_fourier_modes)
        perturb_v = _fourier_random_field(rng, n, n_fourier_modes)
        u0 = np.clip(mean_u + 0.5 * du * perturb_u, 0.0, 1.0)
        v0 = np.clip(mean_v + 0.5 * dv * perturb_v, 0.0, 1.0)
    else:
        msg = (
            "Invalid initial condition type. "
            f"Expected one of {sorted(VALID_INIT_TYPES)}, got '{init_type}'."
        )
        raise ValueError(msg)

    return u0.astype(np.float64), v0.astype(np.float64)


def _etdrk4_coefficients(
    linear_op: np.ndarray,
    dt: float,
    contour_pts: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    E = np.exp(linear_op * dt)
    E2 = np.exp(linear_op * dt / 2.0)
    r = np.exp(1j * np.pi * (np.arange(1, contour_pts + 1) - 0.5) / contour_pts)
    L_flat = linear_op[..., None]
    LR = dt * L_flat + r

    Q = dt * np.mean((np.exp(LR / 2.0) - 1.0) / LR, axis=-1)
    f1 = dt * np.mean(
        (-4.0 - LR + np.exp(LR) * (4.0 - 3.0 * LR + LR**2)) / (LR**3), axis=-1
    )
    f2 = dt * np.mean((2.0 + LR + np.exp(LR) * (-2.0 + LR)) / (LR**3), axis=-1)
    f3 = dt * np.mean(
        (-4.0 - 3.0 * LR - LR**2 + np.exp(LR) * (4.0 - LR)) / (LR**3), axis=-1
    )

    return E, E2, Q, f1, f2, f3


def _nonlinear_terms(
    u: np.ndarray, v: np.ndarray, F: float, k: float
) -> tuple[np.ndarray, np.ndarray]:
    uv2 = u * (v**2)
    return -uv2 + F * (1.0 - u), uv2 - (F + k) * v


def _build_dealias_mask(n: int) -> np.ndarray:
    freq_idx = np.fft.fftfreq(n) * n
    KX, KY = np.meshgrid(freq_idx, freq_idx, indexing="ij")
    cutoff = n / 3.0
    return ((np.abs(KX) <= cutoff) & (np.abs(KY) <= cutoff)).astype(np.float64)


def _etdrk4_step(
    u_hat: np.ndarray,
    v_hat: np.ndarray,
    u_real: np.ndarray,
    v_real: np.ndarray,
    F: float,
    k: float,
    E_u: np.ndarray,
    E2_u: np.ndarray,
    Q_u: np.ndarray,
    f1_u: np.ndarray,
    f2_u: np.ndarray,
    f3_u: np.ndarray,
    E_v: np.ndarray,
    E2_v: np.ndarray,
    Q_v: np.ndarray,
    f1_v: np.ndarray,
    f2_v: np.ndarray,
    f3_v: np.ndarray,
    dealias_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Nu_real, Nv_real = _nonlinear_terms(u_real, v_real, F, k)
    Nv1_u = fft2(Nu_real)
    Nv1_v = fft2(Nv_real)
    if dealias_mask is not None:
        Nv1_u *= dealias_mask
        Nv1_v *= dealias_mask

    a_hat_u = E2_u * u_hat + Q_u * Nv1_u
    a_hat_v = E2_v * v_hat + Q_v * Nv1_v
    a_u = np.real(ifft2(a_hat_u))
    a_v = np.real(ifft2(a_hat_v))

    Nu2_real, Nv2_real = _nonlinear_terms(a_u, a_v, F, k)
    Nv2_u = fft2(Nu2_real)
    Nv2_v = fft2(Nv2_real)
    if dealias_mask is not None:
        Nv2_u *= dealias_mask
        Nv2_v *= dealias_mask

    b_hat_u = E2_u * u_hat + Q_u * Nv2_u
    b_hat_v = E2_v * v_hat + Q_v * Nv2_v
    b_u = np.real(ifft2(b_hat_u))
    b_v = np.real(ifft2(b_hat_v))

    Nu3_real, Nv3_real = _nonlinear_terms(b_u, b_v, F, k)
    Nv3_u = fft2(Nu3_real)
    Nv3_v = fft2(Nv3_real)
    if dealias_mask is not None:
        Nv3_u *= dealias_mask
        Nv3_v *= dealias_mask

    c_hat_u = E2_u * a_hat_u + Q_u * (2.0 * Nv3_u - Nv1_u)
    c_hat_v = E2_v * a_hat_v + Q_v * (2.0 * Nv3_v - Nv1_v)
    c_u = np.real(ifft2(c_hat_u))
    c_v = np.real(ifft2(c_hat_v))

    Nu4_real, Nv4_real = _nonlinear_terms(c_u, c_v, F, k)
    Nv4_u = fft2(Nu4_real)
    Nv4_v = fft2(Nv4_real)
    if dealias_mask is not None:
        Nv4_u *= dealias_mask
        Nv4_v *= dealias_mask

    u_hat_next = E_u * u_hat + f1_u * Nv1_u + f2_u * (Nv2_u + Nv3_u) + f3_u * Nv4_u
    v_hat_next = E_v * v_hat + f1_v * Nv1_v + f2_v * (Nv2_v + Nv3_v) + f3_v * Nv4_v
    if dealias_mask is not None:
        u_hat_next *= dealias_mask
        v_hat_next *= dealias_mask

    u_real_next = np.real(ifft2(u_hat_next))
    v_real_next = np.real(ifft2(v_hat_next))
    return u_hat_next, v_hat_next, u_real_next, v_real_next


def simulate_spectral_gray_scott(  # noqa: PLR0915
    params: dict[str, float],
    *,
    return_timeseries: bool,
    n: int,
    L: float,
    T: float,
    dt: float,
    snapshot_dt: float,
    initial_condition: str,
    gaussian_spec: dict[str, tuple[float, float]],
    n_fourier_modes: int,
    dealias: bool,
    random_seed: int | None = None,
) -> tuple[NumpyLike, NumpyLike]:
    """Simulate Gray-Scott dynamics using a spectral ETDRK4 discretization.

    The solver uses periodic boundary conditions on a square domain,
    pseudospectral evaluation of nonlinear terms, and optional 2/3-rule dealiasing.
    """
    if dt <= 0:
        msg = "Time step dt must be positive."
        raise ValueError(msg)
    if T <= 0:
        msg = "Total simulation time T must be positive."
        raise ValueError(msg)
    if snapshot_dt <= 0:
        msg = "snapshot_dt must be positive."
        raise ValueError(msg)

    try:
        F = float(params["F"])
        k = float(params["k"])
    except KeyError as exc:
        msg = "Spectral Gray-Scott parameters must include 'F' and 'k'."
        raise KeyError(msg) from exc

    delta_u = float(params.get("delta_u", DEFAULT_DIFFUSIONS[0]))
    delta_v = float(params.get("delta_v", DEFAULT_DIFFUSIONS[1]))

    domain = (-L / 2.0, L / 2.0, -L / 2.0, L / 2.0)
    grid = np.linspace(domain[0], domain[1], n, endpoint=False)
    X, Y = np.meshgrid(grid, grid, indexing="ij")

    rng = np.random.default_rng(random_seed)
    u0, v0 = _initialize_fields(
        rng,
        n,
        X,
        Y,
        domain,
        F,
        k,
        initial_condition,
        gaussian_spec,
        n_fourier_modes,
    )

    u_hat = fft2(u0)
    v_hat = fft2(v0)
    u_real = u0
    v_real = v0

    dx = L / n
    freq = np.fft.fftfreq(n, d=dx)
    kx = 2.0 * np.pi * freq
    KX, KY = np.meshgrid(kx, kx, indexing="ij")
    K2 = KX**2 + KY**2

    linear_u = -delta_u * K2
    linear_v = -delta_v * K2

    E_u, E2_u, Q_u, f1_u, f2_u, f3_u = _etdrk4_coefficients(linear_u, dt)
    E_v, E2_v, Q_v, f1_v, f2_v, f3_v = _etdrk4_coefficients(linear_v, dt)
    dealias_mask = _build_dealias_mask(n) if dealias else None

    num_steps = max(1, int(np.ceil(T / dt)))
    snapshot_stride = max(1, int(np.round(snapshot_dt / dt)))

    snapshots_u: list[np.ndarray] = [u_real.astype(np.float32)]
    snapshots_v: list[np.ndarray] = [v_real.astype(np.float32)]

    for step in range(num_steps):
        u_hat, v_hat, u_real, v_real = _etdrk4_step(
            u_hat,
            v_hat,
            u_real,
            v_real,
            F,
            k,
            E_u,
            E2_u,
            Q_u,
            f1_u,
            f2_u,
            f3_u,
            E_v,
            E2_v,
            Q_v,
            f1_v,
            f2_v,
            f3_v,
            dealias_mask,
        )

        if return_timeseries:
            should_store = ((step + 1) % snapshot_stride == 0) or (
                step == num_steps - 1
            )
            if should_store:
                snapshots_u.append(u_real.astype(np.float32))
                snapshots_v.append(v_real.astype(np.float32))

    if return_timeseries:
        u_output = np.stack(snapshots_u, axis=0)
        v_output = np.stack(snapshots_v, axis=0)
    else:
        u_output = u_real.astype(np.float32)
        v_output = v_real.astype(np.float32)

    return u_output, v_output


class GrayScott(SpatioTemporalSimulator):
    """Spectral Gray-Scott simulator based on danfortunato/spectral-gray-scott."""

    def __init__(  # noqa: PLR0912
        self,
        parameters_range: dict[str, tuple[float, float]] | None = None,
        output_names: list[str] | None = None,
        return_timeseries: bool = False,
        log_level: str = "progress_bar",
        n: int = 128,
        L: float = 2.0,
        T: float = 10000.0,
        dt: float = 1.0,
        snapshot_dt: float = 10.0,
        initial_condition: str = "gaussians",
        initial_gaussian_spec: dict[str, tuple[float, float]] | None = None,
        n_fourier_modes: int = 32,
        dealias: bool = True,
        random_seed: int | None = None,
        pattern: str | None = None,
        fixed_parameters_given_pattern: bool = True,
    ) -> None:
        if parameters_range is not None:
            parameters_range = dict(parameters_range)
        elif pattern is not None:
            if pattern not in PATTERN_RANGES:
                raise ValueError(
                    f"Unknown Gray-Scott pattern '{pattern}'. "
                    f"Available presets: {sorted(PATTERN_RANGES)}."
                )

            if fixed_parameters_given_pattern:
                pattern_spec = PATTERN_FIXED_PARAMS[pattern]
                parameters_range = {
                    "F": (pattern_spec["F"], pattern_spec["F"]),
                    "k": (pattern_spec["k"], pattern_spec["k"]),
                    "delta_u": (DEFAULT_DIFFUSIONS[0], DEFAULT_DIFFUSIONS[0]),
                    "delta_v": (DEFAULT_DIFFUSIONS[1], DEFAULT_DIFFUSIONS[1]),
                }
            else:
                pattern_spec = PATTERN_RANGES[pattern]
                parameters_range = {
                    "F": pattern_spec["F"],
                    "k": pattern_spec["k"],
                    "delta_u": (DEFAULT_DIFFUSIONS[0], DEFAULT_DIFFUSIONS[0]),
                    "delta_v": (DEFAULT_DIFFUSIONS[1], DEFAULT_DIFFUSIONS[1]),
                }
        else:
            parameters_range = {
                "F": (0.014, 0.1),
                "k": (0.051, 0.065),
                "delta_u": (DEFAULT_DIFFUSIONS[0], DEFAULT_DIFFUSIONS[0]),
                "delta_v": (DEFAULT_DIFFUSIONS[1], DEFAULT_DIFFUSIONS[1]),
            }

        if output_names is None:
            output_names = ["u", "v"]

        super().__init__(parameters_range, output_names, log_level)

        if initial_condition not in VALID_INIT_TYPES:
            raise ValueError(
                f"initial_condition must be one of {sorted(VALID_INIT_TYPES)}, "
                f"got '{initial_condition}'."
            )

        if n <= 0:
            msg = "Spatial resolution 'n' must be positive."
            raise ValueError(msg)
        if L <= 0:
            msg = "Domain size L must be positive."
            raise ValueError(msg)
        if T <= 0:
            msg = "Simulation horizon T must be positive."
            raise ValueError(msg)
        if dt <= 0:
            msg = "Time step dt must be positive."
            raise ValueError(msg)
        if snapshot_dt <= 0:
            msg = "snapshot_dt must be positive."
            raise ValueError(msg)

        self.return_timeseries = return_timeseries
        self.n = n
        self.L = L
        self.T = T
        self.dt = dt
        self.snapshot_dt = snapshot_dt
        self.initial_condition = initial_condition
        self.gaussian_spec = (
            _validate_gaussian_spec(initial_gaussian_spec)
            if initial_gaussian_spec is not None
            else dict(DEFAULT_GAUSSIAN_SPEC)
        )
        self.n_fourier_modes = max(1, n_fourier_modes)
        self.dealias = dealias
        self.random_seed = random_seed
        self._rng = (
            np.random.default_rng(random_seed) if random_seed is not None else None
        )

        self.diffusions = (
            parameters_range.get("delta_u", (DEFAULT_DIFFUSIONS[0],))[0],
            parameters_range.get("delta_v", (DEFAULT_DIFFUSIONS[1],))[0],
        )

    def _forward(self, x: TensorLike) -> TensorLike:
        if x.shape[0] != 1:
            msg = "Spectral Gray-Scott simulator expects a single input at a time."
            raise ValueError(msg)

        params = dict(zip(self.param_names, x.cpu().numpy()[0].tolist(), strict=False))
        params.setdefault("delta_u", self.diffusions[0])
        params.setdefault("delta_v", self.diffusions[1])

        seed = None
        if self._rng is not None:
            seed = int(self._rng.integers(0, 2**32 - 1))

        u_sol, v_sol = simulate_spectral_gray_scott(
            params,
            return_timeseries=self.return_timeseries,
            n=self.n,
            L=self.L,
            T=self.T,
            dt=self.dt,
            snapshot_dt=self.snapshot_dt,
            initial_condition=self.initial_condition,
            gaussian_spec=self.gaussian_spec,
            n_fourier_modes=self.n_fourier_modes,
            dealias=self.dealias,
            random_seed=seed,
        )

        concat = np.concatenate([u_sol.ravel(), v_sol.ravel()]).astype(np.float32)
        return torch.from_numpy(concat).reshape(1, -1)

    def forward_samples_spatiotemporal(
        self, n: int, random_seed: int | None = None
    ) -> dict:
        """Run multiple trajectories and return `[batch, time, x, y, channels]` data."""
        if not self.return_timeseries:
            msg = "forward_samples_spatiotemporal requires return_timeseries=True."
            raise RuntimeError(msg)
        x = self.sample_inputs(n, random_seed)
        y, x = self.forward_batch(x)

        timesteps = _compute_snapshot_count(self.T, self.dt, self.snapshot_dt)
        y_reshaped = y.reshape(y.shape[0], 2, timesteps, self.n, self.n).permute(
            0, 2, 3, 4, 1
        )

        return {
            "data": y_reshaped,
            "constant_scalars": x,
            "constant_fields": None,
        }
