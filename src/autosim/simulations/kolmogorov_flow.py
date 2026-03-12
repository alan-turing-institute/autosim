from __future__ import annotations

import numpy as np
import torch
from scipy.integrate import solve_ivp

from autosim.simulations.base import SpatioTemporalSimulator
from autosim.types import TensorLike


class KolmogorovFlow(SpatioTemporalSimulator):
    r"""2D incompressible Kolmogorov flow (periodic) in vorticity form.

    We evolve vorticity \(\omega\) on a periodic domain and recover velocity from the
    streamfunction \(\psi\):

    \[
    \partial_t \omega + \mathbf{u}\cdot\nabla\omega
      = \nu \nabla^2\omega + f_\omega(y) - \alpha \omega,
    \qquad \nabla^2\psi = -\omega,
    \qquad u=\partial_y\psi,\ v=-\partial_x\psi.
    \]

    Forcing corresponds to a body force in the x-momentum equation
    \(f_x = A\sin(k_f y)\), which yields vorticity forcing
    \(f_\omega(y) = -\partial_y f_x = -A k_f \cos(k_f y)\).

    Outputs channels: `[vorticity, u, v, streamfunction]`.
    """

    _ALL_CHANNEL_NAMES = ("vorticity", "u", "v", "streamfunction")

    def __init__(
        self,
        parameters_range: dict[str, tuple[float, float]] | None = None,
        output_names: list[str] | None = None,
        return_timeseries: bool = False,
        log_level: str = "progress_bar",
        n: int = 64,
        L: float = 2.0 * np.pi,
        T: float = 20.0,
        dt: float = 0.1,
        kf: int = 4,
        integrator_kwargs: dict | None = None,
    ) -> None:
        if parameters_range is None:
            parameters_range = {
                # Conservative defaults for long-horizon stability.
                "nu": (0.004, 0.01),
                "forcing": (0.4, 1.0),  # A
                "alpha": (0.04, 0.2),  # linear drag
            }
        if output_names is None:
            output_names = list(self._ALL_CHANNEL_NAMES)
        super().__init__(parameters_range, output_names, log_level)

        self.return_timeseries = return_timeseries
        self.n = int(n)
        self.L = float(L)
        self.T = float(T)
        self.dt = float(dt)
        self.kf = int(kf)
        self.integrator_kwargs = integrator_kwargs or {}

    def _forward(self, x: TensorLike) -> TensorLike:
        if x.shape[0] != 1:
            msg = "Simulator._forward expects a single input (batch size 1)"
            raise ValueError(msg)

        nu = float(x[0, self.get_parameter_idx("nu")].item())
        forcing = float(x[0, self.get_parameter_idx("forcing")].item())
        alpha = float(x[0, self.get_parameter_idx("alpha")].item())

        sol = simulate_kolmogorov_flow(
            nu=nu,
            forcing=forcing,
            alpha=alpha,
            return_timeseries=self.return_timeseries,
            n=self.n,
            L=self.L,
            T=self.T,
            dt=self.dt,
            kf=self.kf,
            integrator_kwargs=self.integrator_kwargs,
        )
        arr = np.asarray(sol, dtype=np.float32)
        expected_per_step = self.n * self.n * 4
        if self.return_timeseries:
            n_time = len(np.arange(0.0, self.T + 1e-12, self.dt))
            expected = n_time * expected_per_step
        else:
            expected = expected_per_step
        if arr.size != expected:
            raise RuntimeError(
                "KolmogorovFlow internal size mismatch: "
                f"return_timeseries={self.return_timeseries} n={self.n} "
                f"T={self.T} dt={self.dt} expected_size={expected} got_size={arr.size} "
                f"sol_shape={getattr(arr, 'shape', None)}"
            )
        return torch.from_numpy(arr.ravel()).reshape(1, -1)

    def forward_samples_spatiotemporal(
        self,
        n: int,
        random_seed: int | None = None,
        ensure_exact_n: bool = False,
    ) -> dict:
        y, x = self._forward_batch_with_optional_retries(
            n=n,
            random_seed=random_seed,
            ensure_exact_n=ensure_exact_n,
        )
        if y.shape[0] == 0:
            raise RuntimeError(
                "All KolmogorovFlow trajectories failed. This usually means the "
                "sampled parameter/time-horizon combination is numerically unstable. "
                "Try increasing dissipation (higher nu/alpha), reducing forcing, "
                "reducing T, or setting ensure_exact_n=True."
            )

        channels = 4
        features_per_step = self.n * self.n * channels
        if self.return_timeseries:
            total_features = y.shape[1]
            if total_features % features_per_step != 0:
                raise RuntimeError(
                    "Returned tensor does not align with n*n*channels; "
                    f"received {total_features} features, expected multiples of "
                    f"{features_per_step}."
                )
            n_time = total_features // features_per_step
            y_reshaped = y.reshape(y.shape[0], n_time, self.n, self.n, channels)
        else:
            if y.shape[1] != features_per_step:
                raise RuntimeError(
                    "Unexpected flattened size for single snapshot; "
                    f"received {y.shape[1]}, expected {features_per_step}."
                )
            y_reshaped = y.reshape(y.shape[0], 1, self.n, self.n, channels)

        return {
            "data": y_reshaped,
            "constant_scalars": x,
            "constant_fields": None,
        }


def _fft_kvec(n: int, L: float) -> np.ndarray:
    return (2.0 * np.pi / L) * np.fft.fftfreq(n, d=L / n)


def _poisson_streamfunction(
    omega_hat: np.ndarray, kx: np.ndarray, ky: np.ndarray
) -> np.ndarray:
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    K2 = KX**2 + KY**2
    psi_hat = np.zeros_like(omega_hat)
    mask = K2 > 0
    psi_hat[mask] = -omega_hat[mask] / K2[mask]
    psi_hat[0, 0] = 0.0
    return psi_hat


def _gradient_periodic(f2d: np.ndarray, dx: float) -> tuple[np.ndarray, np.ndarray]:
    dfdx = (np.roll(f2d, -1, axis=1) - np.roll(f2d, 1, axis=1)) / (2 * dx)
    dfdy = (np.roll(f2d, -1, axis=0) - np.roll(f2d, 1, axis=0)) / (2 * dx)
    return dfdx, dfdy


def _laplacian_periodic(f2d: np.ndarray, dx: float) -> np.ndarray:
    return (np.roll(f2d, -1, axis=0) - 2 * f2d + np.roll(f2d, 1, axis=0)) / dx**2 + (
        np.roll(f2d, -1, axis=1) - 2 * f2d + np.roll(f2d, 1, axis=1)
    ) / dx**2


def simulate_kolmogorov_flow(
    *,
    nu: float,
    forcing: float,
    alpha: float,
    return_timeseries: bool,
    n: int,
    L: float,
    T: float,
    dt: float,
    kf: int,
    integrator_kwargs: dict | None = None,
) -> np.ndarray:
    """Integrate Kolmogorov flow and emit channels `[ω,u,v,ψ]`."""
    t_eval = np.arange(0.0, T + 1e-12, dt)
    n_time = len(t_eval)

    x = np.linspace(0.0, L, n, endpoint=False)
    y = np.linspace(0.0, L, n, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="xy")
    dx = float(x[1] - x[0])

    # Mild random initial vorticity, low-pass filtered.
    rng = np.random.default_rng(0)
    w0 = 0.1 * rng.standard_normal((n, n)).astype(np.float64)
    w0_hat = np.fft.fft2(w0)
    kx = _fft_kvec(n, L)
    ky = _fft_kvec(n, L)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    K2 = KX**2 + KY**2
    k_cut = 0.35 * np.max(np.abs(kx))
    w0_hat[k_cut**2 < K2] = 0.0
    w0 = np.real(np.fft.ifft2(w0_hat))

    # Vorticity forcing (from fx = forcing * sin(kf*y)).
    f_omega = (-forcing * float(kf) * np.cos(float(kf) * Y)).astype(np.float64)

    def rhs(_t: float, w_flat: np.ndarray) -> np.ndarray:
        w = w_flat.reshape(n, n)
        w_hat = np.fft.fft2(w)

        psi_hat = _poisson_streamfunction(w_hat, kx, ky)

        # velocities from streamfunction via periodic finite differences
        psi = np.real(np.fft.ifft2(psi_hat))
        dpsidx, dpsidy = _gradient_periodic(psi, dx)
        u = dpsidy
        v = -dpsidx

        # vorticity transport + diffusion in physical space
        w_x, w_y = _gradient_periodic(w, dx)
        adv = u * w_x + v * w_y
        diff = _laplacian_periodic(w, dx)

        dwdt = -adv + nu * diff + f_omega - alpha * w
        return dwdt.ravel()

    # Fast-by-default for interactive notebook usage, with optional stiff fallback.
    ode_kwargs = {"rtol": 5e-4, "atol": 1e-6, **(integrator_kwargs or {})}
    preferred_method = str(ode_kwargs.pop("method", "RK45"))
    allow_fallback = bool(ode_kwargs.pop("allow_fallback", True))
    methods = [preferred_method]
    if allow_fallback:
        for method in ("BDF", "Radau"):
            if method not in methods:
                methods.append(method)

    sol = None
    last_message = ""
    for method in methods:
        sol = solve_ivp(
            fun=rhs,
            t_span=(0.0, T),
            y0=w0.ravel(),
            t_eval=t_eval,
            method=method,
            **ode_kwargs,
        )
        if sol.success:
            break
        last_message = f"{method}: {sol.message}"

    if sol is None or not sol.success:
        raise RuntimeError(
            "ODE solver failed after trying methods "
            f"{methods}. Last failure: {last_message}"
        )

    def channels_from_omega(w2d: np.ndarray) -> np.ndarray:
        w_hat = np.fft.fft2(w2d)
        psi_hat = _poisson_streamfunction(w_hat, kx, ky)
        psi = np.real(np.fft.ifft2(psi_hat))
        dpsidx, dpsidy = _gradient_periodic(psi, dx)
        u = dpsidy
        v = -dpsidx
        return np.stack([w2d, u, v, psi], axis=-1).astype(np.float32, copy=False)

    if return_timeseries:
        w_ts = sol.y.T.reshape(n_time, n, n).astype(np.float64, copy=False)
        out = np.empty((n_time, n, n, 4), dtype=np.float32)
        for ti, w2d in enumerate(w_ts):
            out[ti] = channels_from_omega(w2d)
        return out

    w_final = sol.y[:, -1].reshape(n, n)
    return channels_from_omega(w_final)
