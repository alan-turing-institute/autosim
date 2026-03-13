import numpy as np
import torch
from numpy.fft import fft2, ifft2

from autosim.simulations.base import SpatioTemporalSimulator
from autosim.types import NumpyLike, TensorLike


def _etdrk4_coefficients(
    linear_op: np.ndarray,
    dt: float,
    contour_pts: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precompute ETDRK4 coefficients for a linear operator (e.g. -d*K2)."""
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


def _rd_nonlinear_terms(
    u: np.ndarray, v: np.ndarray, beta: float
) -> tuple[np.ndarray, np.ndarray]:
    """Reaction-diffusion nonlinear terms in real space (same PDE as RD RHS)."""
    u2v = u * u * v
    uv2 = u * v * v
    N_u = u - u**3 - uv2 + beta * (u2v + v**3)
    N_v = v - u2v - v**3 - beta * (u**3 + uv2)
    return N_u, N_v


def _build_dealias_mask(n: int) -> np.ndarray:
    freq_idx = np.fft.fftfreq(n) * n
    KX, KY = np.meshgrid(freq_idx, freq_idx, indexing="ij")
    cutoff = n / 3.0
    return ((np.abs(KX) <= cutoff) & (np.abs(KY) <= cutoff)).astype(np.float64)


def _etdrk4_step_rd(
    u_hat: np.ndarray,
    v_hat: np.ndarray,
    u_real: np.ndarray,
    v_real: np.ndarray,
    beta: float,
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
    """Single ETDRK4 step for reaction-diffusion in Fourier space."""
    Nu_real, Nv_real = _rd_nonlinear_terms(u_real, v_real, beta)
    Nv1_u = fft2(Nu_real)
    Nv1_v = fft2(Nv_real)
    if dealias_mask is not None:
        Nv1_u *= dealias_mask
        Nv1_v *= dealias_mask

    a_hat_u = E2_u * u_hat + Q_u * Nv1_u
    a_hat_v = E2_v * v_hat + Q_v * Nv1_v
    a_u = np.real(ifft2(a_hat_u))
    a_v = np.real(ifft2(a_hat_v))

    Nu2_real, Nv2_real = _rd_nonlinear_terms(a_u, a_v, beta)
    Nv2_u = fft2(Nu2_real)
    Nv2_v = fft2(Nv2_real)
    if dealias_mask is not None:
        Nv2_u *= dealias_mask
        Nv2_v *= dealias_mask

    b_hat_u = E2_u * u_hat + Q_u * Nv2_u
    b_hat_v = E2_v * v_hat + Q_v * Nv2_v
    b_u = np.real(ifft2(b_hat_u))
    b_v = np.real(ifft2(b_hat_v))

    Nu3_real, Nv3_real = _rd_nonlinear_terms(b_u, b_v, beta)
    Nv3_u = fft2(Nu3_real)
    Nv3_v = fft2(Nv3_real)
    if dealias_mask is not None:
        Nv3_u *= dealias_mask
        Nv3_v *= dealias_mask

    c_hat_u = E2_u * a_hat_u + Q_u * (2.0 * Nv3_u - Nv1_u)
    c_hat_v = E2_v * a_hat_v + Q_v * (2.0 * Nv3_v - Nv1_v)
    c_u = np.real(ifft2(c_hat_u))
    c_v = np.real(ifft2(c_hat_v))

    Nu4_real, Nv4_real = _rd_nonlinear_terms(c_u, c_v, beta)
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


def _compute_snapshot_count(total_time: float, dt: float, snapshot_dt: float) -> int:
    """Match Gray-Scott: initial frame + store when (step+1)%stride==0 or step==last."""
    num_steps = max(1, int(np.ceil(total_time / dt)))
    stride = max(1, int(np.round(snapshot_dt / dt)))
    count = 1 + num_steps // stride
    if num_steps % stride != 0:
        count += 1
    return count


def simulate_reaction_diffusion_spectral(
    x: NumpyLike,
    return_timeseries: bool = False,
    n: int = 32,
    L: int = 20,
    T: float = 10.0,
    dt: float = 0.1,
    snapshot_dt: float | None = None,
    dealias: bool = True,
) -> tuple[NumpyLike, NumpyLike]:
    """
    Simulate reaction-diffusion using fixed-step spectral ETDRK4 (Gray-Scott style).

    Uses only current state plus snapshot lists during time stepping, avoiding
    the large solve_ivp internal storage. snapshot_dt defaults to dt (save every step).
    """
    beta, d = float(x[0]), float(x[1])
    d1 = d2 = d
    if snapshot_dt is None:
        snapshot_dt = dt

    grid = np.linspace(-L / 2.0, L / 2.0, n, endpoint=False)
    X, Y = np.meshgrid(grid, grid, indexing="ij")
    r = np.sqrt(X**2 + Y**2)
    theta = np.angle(X + 1j * Y)
    u0 = np.tanh(r) * np.cos(theta - r)
    v0 = np.tanh(r) * np.sin(theta - r)
    u0 = u0.astype(np.float64)
    v0 = v0.astype(np.float64)

    dx = L / n
    freq = np.fft.fftfreq(n, d=dx)
    kx = 2.0 * np.pi * freq
    KX, KY = np.meshgrid(kx, kx, indexing="ij")
    K2 = KX**2 + KY**2
    linear_u = -d1 * K2
    linear_v = -d2 * K2

    E_u, E2_u, Q_u, f1_u, f2_u, f3_u = _etdrk4_coefficients(linear_u, dt)
    E_v, E2_v, Q_v, f1_v, f2_v, f3_v = _etdrk4_coefficients(linear_v, dt)
    dealias_mask = _build_dealias_mask(n) if dealias else None

    u_hat = fft2(u0)
    v_hat = fft2(v0)
    u_real = u0.copy()
    v_real = v0.copy()

    num_steps = max(1, int(np.ceil(T / dt)))
    snapshot_stride = max(1, int(np.round(snapshot_dt / dt)))

    snapshots_u: list[np.ndarray] = [u_real.astype(np.float32)]
    snapshots_v: list[np.ndarray] = [v_real.astype(np.float32)]

    for step in range(num_steps):
        u_hat, v_hat, u_real, v_real = _etdrk4_step_rd(
            u_hat,
            v_hat,
            u_real,
            v_real,
            beta,
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
        u_out = np.stack(snapshots_u, axis=0)
        v_out = np.stack(snapshots_v, axis=0)
    else:
        u_out = u_real.astype(np.float32)
        v_out = v_real.astype(np.float32)
    return u_out, v_out


class ReactionDiffusion(SpatioTemporalSimulator):
    """Simulate the reaction-diffusion PDE for a given set of parameters.

    Uses spectral ETDRK4 (fixed timestep) by default for lower memory and faster
    runs at large n, similar to the Gray-Scott simulator.
    """

    def __init__(
        self,
        parameters_range: dict[str, tuple[float, float]] | None = None,
        output_names: list[str] | None = None,
        return_timeseries: bool = False,
        log_level: str = "progress_bar",
        n: int = 32,
        L: int = 20,
        T: float = 10.0,
        dt: float = 0.1,
        snapshot_dt: float | None = None,
        dealias: bool = True,
    ):
        """
        Initialize the ReactionDiffusion simulator.

        Parameters
        ----------
        parameters_range: dict[str, tuple[float, float]]
            Dictionary mapping input parameter names to their (min, max) ranges.
        output_names: list[str]
            List of output parameters' names.
        log_level: str
            Logging level for the simulator. Can be one of:
            - "progress_bar": shows a progress bar during batch simulations
            - "debug": shows debug messages
            - "info": shows informational messages
            - "warning": shows warning messages
            - "error": shows error messages
            - "critical": shows critical messages
        return_timeseries: bool
            Whether to return the full timeseries or just the spatial solution at the
            final time step. Defaults to False.
        n: int
            Number of spatial points in each direction.
        L: int
            Domain size in X and Y directions.
        T: float
            Total time to simulate.
        dt: float
            Time step size.
        snapshot_dt: float | None
            Interval between saved frames when return_timeseries is True. Defaults to
            dt (save every step). Use a multiple of dt to reduce output size.
        dealias: bool
            Apply 2/3 dealiasing in spectral space. Default True for stability at large n.
        """
        if parameters_range is None:
            parameters_range = {"beta": (1.0, 2.0), "d": (0.05, 0.3)}
        if output_names is None:
            output_names = ["u", "v"]
        super().__init__(parameters_range, output_names, log_level)
        self.return_timeseries = return_timeseries
        self.n = n
        self.L = L
        self.T = T
        self.dt = dt
        self.snapshot_dt = dt if snapshot_dt is None else snapshot_dt
        self.dealias = dealias

    def _forward(self, x: TensorLike) -> TensorLike:
        assert x.shape[0] == 1, (
            f"Simulator._forward expects a single input, got {x.shape[0]}"
        )
        u_sol, v_sol = simulate_reaction_diffusion_spectral(
            x.cpu().numpy()[0],
            return_timeseries=self.return_timeseries,
            n=self.n,
            L=self.L,
            T=self.T,
            dt=self.dt,
            snapshot_dt=self.snapshot_dt,
            dealias=self.dealias,
        )
        concat_array = np.concatenate([u_sol.ravel(), v_sol.ravel()]).astype(
            np.float32, copy=False
        )
        return torch.from_numpy(concat_array).reshape(1, -1)

    def forward_samples_spatiotemporal(
        self,
        n: int,
        random_seed: int | None = None,
        ensure_exact_n: bool = False,
    ) -> dict:
        """Reshape to spatiotemporal format.

        Parameters
        ----------
        n: int
            Number of samples to generate.
        random_seed: int | None
            Random seed for reproducibility. Defaults to None.

        Returns
        -------
        dict
            A dictionary containing the reshaped spatiotemporal data, constant scalars,
            and constant fields.
        """
        # Run simulation with compact preallocation to avoid list+cat peak-memory blowups
        # for large spatiotemporal outputs (e.g. n=128 with full timeseries).
        y, x = self._forward_batch_compact(
            n=n,
            random_seed=random_seed,
            ensure_exact_n=ensure_exact_n,
        )

        if self.return_timeseries:
            n_time = _compute_snapshot_count(self.T, self.dt, self.snapshot_dt)
            y_reshaped_permuted = y.reshape(
                y.shape[0], 2, n_time, self.n, self.n
            ).permute(0, 2, 3, 4, 1)
        else:
            y_reshaped_permuted = y.reshape(y.shape[0], 2, 1, self.n, self.n).permute(
                0, 2, 3, 4, 1
            )

        return {
            "data": y_reshaped_permuted,
            "constant_scalars": x,
            "constant_fields": None,
        }

    def _forward_batch_compact(
        self,
        n: int,
        random_seed: int | None = None,
        ensure_exact_n: bool = False,
    ) -> tuple[TensorLike, TensorLike]:
        """Run batch simulations while minimizing transient memory allocations."""
        if self.return_timeseries:
            n_time = _compute_snapshot_count(self.T, self.dt, self.snapshot_dt)
        else:
            n_time = 1
        features_per_sample = 2 * n_time * self.n * self.n

        y_buffer = torch.empty((n, features_per_sample), dtype=torch.float32)
        x_buffer = torch.empty((n, self.in_dim), dtype=torch.float32)
        successful = 0

        retry_round = 0
        max_rounds = self._retry_budget(n)

        while True:
            if retry_round == 0:
                sample_count = n
                seed = random_seed
            else:
                if not ensure_exact_n or successful >= n:
                    break
                sample_count = n - successful
                seed = None if random_seed is None else random_seed + retry_round

            sampled_x = self.sample_inputs(sample_count, seed)
            for i in range(sampled_x.shape[0]):
                if ensure_exact_n and successful >= n:
                    break
                xi = sampled_x[i : i + 1]
                result = self.forward(xi, allow_failures=True)
                if result is None:
                    continue
                y_buffer[successful] = result.squeeze(0)
                x_buffer[successful] = xi.squeeze(0)
                successful += 1

            if not ensure_exact_n:
                break
            if successful >= n:
                break

            retry_round += 1
            if retry_round > max_rounds:
                raise RuntimeError(
                    f"Could not collect exactly n={n} successful samples after "
                    f"{max_rounds} retry rounds. Collected {successful}."
                )

        return y_buffer[:successful], x_buffer[:successful]
