from __future__ import annotations

import math

import torch

from autosim.simulations.base import SpatioTemporalSimulator
from autosim.types import TensorLike


def _laplacian(field: torch.Tensor, dx: float) -> torch.Tensor:
    """Laplacian with periodic BCs (torch.roll wrapping)."""
    return (
        (torch.roll(field, -1, dims=0) - 2.0 * field + torch.roll(field, 1, dims=0))
        + (torch.roll(field, -1, dims=1) - 2.0 * field + torch.roll(field, 1, dims=1))
    ) / (dx**2)


def _laplacian_neumann(field: torch.Tensor, dx: float) -> torch.Tensor:
    """Laplacian with Neumann (zero-gradient) BCs via replicate padding."""
    import torch.nn.functional as F  # noqa: PLC0415

    f = F.pad(field.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="replicate")
    f = f.squeeze(0).squeeze(0)
    return (
        f[2:, 1:-1]
        - 2.0 * field
        + f[:-2, 1:-1]
        + f[1:-1, 2:]
        - 2.0 * field
        + f[1:-1, :-2]
    ) / (dx**2)


def _stable_timestep(
    u: torch.Tensor,
    v: torch.Tensor,
    dx: float,
    nu: float,
    smoke_diffusivity: float,
    dt_max: float,
    cfl: float,
    t: float,
    T: float,
) -> float:
    speed = torch.sqrt(u**2 + v**2)
    umax = float(speed.max().item())
    adv_dt = cfl * dx / max(umax, 1e-8)
    diff_coeff = max(nu, smoke_diffusivity, 1e-8)
    diff_dt = 0.25 * dx * dx / diff_coeff
    return min(dt_max, adv_dt, diff_dt, T - t)


def _project_incompressible(
    u: torch.Tensor,
    v: torch.Tensor,
    L: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project velocity onto the divergence-free subspace in Fourier space."""
    n = u.shape[0]
    k = 2.0 * math.pi * torch.fft.fftfreq(n, d=L / n, device=u.device)
    kx, ky = torch.meshgrid(k, k, indexing="ij")

    # Keep correction Hermitian-symmetric for even-sized grids.
    if n % 2 == 0:
        kx = kx.clone()
        ky = ky.clone()
        kx[n // 2, :] = 0.0
        ky[:, n // 2] = 0.0

    k2 = kx**2 + ky**2
    k2_safe = torch.where(k2 > 0, k2, torch.ones_like(k2))

    u_hat = torch.fft.fft2(u)
    v_hat = torch.fft.fft2(v)
    kdotu = kx * u_hat + ky * v_hat

    u_proj = torch.fft.ifft2(u_hat - kx * kdotu / k2_safe).real
    v_proj = torch.fft.ifft2(v_hat - ky * kdotu / k2_safe).real

    p_hat = -1j * kdotu / k2_safe
    p_hat[0, 0] = 0.0
    p = torch.fft.ifft2(p_hat).real

    return u_proj, v_proj, p


def _bilinear_sample_periodic(
    field: torch.Tensor,
    xq: torch.Tensor,
    yq: torch.Tensor,
) -> torch.Tensor:
    n = field.shape[0]
    x0 = torch.floor(xq).long() % n
    y0 = torch.floor(yq).long() % n
    x1 = (x0 + 1) % n
    y1 = (y0 + 1) % n

    wx = xq - torch.floor(xq)
    wy = yq - torch.floor(yq)

    f00 = field[x0, y0]
    f10 = field[x1, y0]
    f01 = field[x0, y1]
    f11 = field[x1, y1]

    return (
        (1.0 - wx) * (1.0 - wy) * f00
        + wx * (1.0 - wy) * f10
        + (1.0 - wx) * wy * f01
        + wx * wy * f11
    )


def _bilinear_sample_bounded(
    field: torch.Tensor,
    xq: torch.Tensor,
    yq: torch.Tensor,
) -> torch.Tensor:
    """Bilinear interpolation with clamped (Neumann/no-slip) boundary sampling."""
    n = field.shape[0]
    xq_c = xq.clamp(0.0, n - 1.0)
    yq_c = yq.clamp(0.0, n - 1.0)
    x0 = torch.floor(xq_c).long().clamp(0, n - 2)
    y0 = torch.floor(yq_c).long().clamp(0, n - 2)
    x1 = (x0 + 1).clamp(0, n - 1)
    y1 = (y0 + 1).clamp(0, n - 1)

    wx = (xq_c - x0.float()).clamp(0.0, 1.0)
    wy = (yq_c - y0.float()).clamp(0.0, 1.0)

    f00 = field[x0, y0]
    f10 = field[x1, y0]
    f01 = field[x0, y1]
    f11 = field[x1, y1]

    return (
        (1.0 - wx) * (1.0 - wy) * f00
        + wx * (1.0 - wy) * f10
        + (1.0 - wx) * wy * f01
        + wx * wy * f11
    )


def _advect_semi_lagrangian(
    field: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    dt: float,
    dx: float,
    grid_x: torch.Tensor,
    grid_y: torch.Tensor,
    *,
    periodic: bool = True,
) -> torch.Tensor:
    n = field.shape[0]
    x_back = grid_x - (dt / dx) * u
    y_back = grid_y - (dt / dx) * v
    if periodic:
        x_back = torch.remainder(x_back, n)
        y_back = torch.remainder(y_back, n)
        return _bilinear_sample_periodic(field, x_back, y_back)
    return _bilinear_sample_bounded(field, x_back, y_back)


def _pdearena_like_smoke_initial_condition(
    n: int,
    L: float,
    rng: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
    smoothness: float = 6.0,
    noise_scale: float = 11.0,
) -> torch.Tensor:
    """Replicates the noise generation from phiflow's Noise class used in PDEArena."""
    # Complex random noise
    rnd_real = torch.randn((n, n), dtype=dtype, generator=rng, device=device)
    rnd_imag = torch.randn((n, n), dtype=dtype, generator=rng, device=device)
    rndj = torch.complex(rnd_real, rnd_imag)

    # Frequency grid
    k_idx = torch.fft.fftfreq(n, d=1.0, device=device) * n
    kx, ky = torch.meshgrid(k_idx, k_idx, indexing="ij")

    # phiflow scales frequencies by (resolution * scale / size)
    # math.fftfreq(resolution, size) is equivalent to torch.fft.fftfreq(n, d=L)
    # k_vec = math.fftfreq(resolution, size) * resolution * scale
    k_vec_x = (kx / L) * noise_scale
    k_vec_y = (ky / L) * noise_scale
    k2 = k_vec_x**2 + k_vec_y**2

    lowest_frequency = 0.1
    weight_mask = (k2 > lowest_frequency).to(dtype)

    inv_k2 = torch.where(k2 > 0, 1.0 / k2, torch.zeros_like(k2))

    fft = rndj * (inv_k2**smoothness) * weight_mask

    array = torch.fft.ifft2(fft).real

    # Normalize
    array = array / array.std()
    array = array - array.mean()

    # PDEArena takes the absolute value of the noise for smoke
    return array.abs()


def simulate_conditioned_navier_stokes_2d(  # noqa: PLR0912, PLR0915
    params: TensorLike,
    *,
    return_timeseries: bool = False,
    n: int = 64,
    L: float = 32.0,
    T: float = 2.0,
    dt: float = 0.01,
    snapshot_dt: float | None = None,
    nu: float = 0.03,
    smoke_diffusivity: float = 5e-4,
    cfl: float = 0.35,
    smoothness: float = 6.0,
    noise_scale: float = 11.0,
    bc_mode: str = "periodic",
    buoyancy_mode: str = "anomaly",
    random_seed: int | None = None,
) -> TensorLike:
    """Simulate a conditioned 2D smoke-flow Navier-Stokes system.

    State channels are `[smoke, u, v]`, where `smoke` is a passive scalar that
    drives buoyancy forcing in the vertical velocity equation.

    Args:
        bc_mode: ``"periodic"`` (default) wraps all fields; ``"neumann"`` uses
            zero-gradient BCs for smoke and no-slip BCs for velocity.
        buoyancy_mode: ``"anomaly"`` (default, Boussinesq) forces with
            ``smoke - mean(smoke)``; ``"raw"`` forces with raw smoke values.
    """
    buoyancy_y = float(params[0].item())

    if n <= 0:
        msg = "n must be positive"
        raise ValueError(msg)
    if L <= 0:
        msg = "L must be positive"
        raise ValueError(msg)
    if T <= 0:
        msg = "T must be positive"
        raise ValueError(msg)
    if dt <= 0:
        msg = "dt must be positive"
        raise ValueError(msg)
    if snapshot_dt is not None and snapshot_dt <= 0:
        msg = "snapshot_dt must be positive when provided"
        raise ValueError(msg)
    if smoothness < 0:
        msg = "smoothness must be non-negative"
        raise ValueError(msg)
    if bc_mode not in ("periodic", "neumann"):
        msg = "bc_mode must be 'periodic' or 'neumann'"
        raise ValueError(msg)
    if buoyancy_mode not in ("anomaly", "raw"):
        msg = "buoyancy_mode must be 'anomaly' or 'raw'"
        raise ValueError(msg)

    if snapshot_dt is None:
        snapshot_dt = dt

    device = params.device if isinstance(params, torch.Tensor) else torch.device("cpu")
    dtype = torch.float32

    dx = L / n

    if random_seed is None:
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
    else:
        seed = int(random_seed)
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    smoke = _pdearena_like_smoke_initial_condition(
        n,
        L,
        rng,
        device,
        dtype,
        smoothness=smoothness,
        noise_scale=noise_scale,
    )
    u = torch.zeros((n, n), device=device, dtype=dtype)
    v = torch.zeros((n, n), device=device, dtype=dtype)

    idx = torch.arange(n, device=device, dtype=dtype)
    grid_x, grid_y = torch.meshgrid(idx, idx, indexing="ij")

    history: list[torch.Tensor] = []

    def _snapshot() -> torch.Tensor:
        return torch.stack([smoke, u, v], dim=-1)

    _periodic = bc_mode == "periodic"
    _lap = _laplacian if _periodic else _laplacian_neumann

    with torch.no_grad():
        next_snapshot_t = float(snapshot_dt)
        last_snapshot_t = 0.0

        if return_timeseries:
            history.append(_snapshot())

        t = 0.0
        while t < T - 1e-12:
            dt_step = _stable_timestep(
                u=u,
                v=v,
                dx=dx,
                nu=nu,
                smoke_diffusivity=smoke_diffusivity,
                dt_max=dt,
                cfl=cfl,
                t=t,
                T=T,
            )
            smoke_adv = _advect_semi_lagrangian(
                smoke, u, v, dt_step, dx, grid_x, grid_y, periodic=_periodic
            )
            u_adv = _advect_semi_lagrangian(
                u, u, v, dt_step, dx, grid_x, grid_y, periodic=_periodic
            )
            v_adv = _advect_semi_lagrangian(
                v, u, v, dt_step, dx, grid_x, grid_y, periodic=_periodic
            )

            if buoyancy_mode == "anomaly":
                buoyancy_term = smoke_adv - smoke_adv.mean()
            else:  # "raw" buoyancy forcing
                buoyancy_term = smoke_adv

            u_tmp = u_adv + dt_step * nu * _lap(u_adv, dx)
            v_tmp = (
                v_adv
                + dt_step * nu * _lap(v_adv, dx)
                + dt_step * buoyancy_y * buoyancy_term
            )
            smoke_tmp = smoke_adv + dt_step * smoke_diffusivity * _lap(smoke_adv, dx)
            smoke_tmp = smoke_tmp.clamp(min=0.0)

            u, v, _ = _project_incompressible(u_tmp, v_tmp, L)

            # Bounded mode: remove mean drift and enforce no-slip walls.
            if not _periodic:
                u = u - u.mean()
                v = v - v.mean()
                u[[0, -1], :] = 0.0
                u[:, [0, -1]] = 0.0
                v[[0, -1], :] = 0.0
                v[:, [0, -1]] = 0.0

            smoke = smoke_tmp
            t += dt_step

            if return_timeseries:
                while next_snapshot_t <= t + 1e-12:
                    history.append(_snapshot())
                    last_snapshot_t = next_snapshot_t
                    next_snapshot_t += snapshot_dt

        if return_timeseries and T - last_snapshot_t > 1e-9:
            history.append(_snapshot())

    if return_timeseries:
        return torch.stack(history, dim=0)

    return _snapshot()


class ConditionedNavierStokes2D(SpatioTemporalSimulator):
    """Conditioned 2D Navier-Stokes smoke simulator inspired by PDEArena."""

    _DEFAULT_SMOOTHNESS = 6.0
    _DEFAULT_NOISE_SCALE = 11.0
    _DEFAULT_SMOKE_DIFFUSIVITY = 5e-4

    _ALLOWED_PARAMETER_NAMES = (
        "buoyancy_y",
        "smoothness",
        "noise_scale",
        "smoke_diffusivity",
    )

    def __init__(
        self,
        parameters_range: dict[str, tuple[float, float]] | None = None,
        output_names: list[str] | None = None,
        return_timeseries: bool = False,
        log_level: str = "progress_bar",
        n: int = 64,
        L: float = 32.0,
        T: float = 2.0,
        dt: float = 0.01,
        snapshot_dt: float | None = None,
        nu: float = 0.03,
        cfl: float = 0.35,
        bc_mode: str = "periodic",
        buoyancy_mode: str = "anomaly",
        skip_nt: int = 0,
        random_seed: int | None = None,
    ) -> None:
        if parameters_range is None:
            parameters_range = {
                "buoyancy_y": (0.2, 0.5),
                "smoothness": (4.0, 8.0),
                "noise_scale": (8.0, 18.0),
                "smoke_diffusivity": (0.0, 1e-3),
            }
        if output_names is None:
            output_names = ["smoke", "u", "v"]

        if "buoyancy_y" not in parameters_range:
            msg = "parameters_range must include 'buoyancy_y'"
            raise ValueError(msg)

        unknown_parameters = set(parameters_range) - set(self._ALLOWED_PARAMETER_NAMES)
        if unknown_parameters:
            unknown = ", ".join(sorted(unknown_parameters))
            msg = (
                "Unsupported parameters in parameters_range: "
                f"{unknown}. Allowed names are: {sorted(self._ALLOWED_PARAMETER_NAMES)}"
            )
            raise ValueError(msg)

        super().__init__(parameters_range, output_names, log_level)

        if n <= 0:
            msg = "n must be positive"
            raise ValueError(msg)
        if L <= 0:
            msg = "L must be positive"
            raise ValueError(msg)
        if T <= 0:
            msg = "T must be positive"
            raise ValueError(msg)
        if dt <= 0:
            msg = "dt must be positive"
            raise ValueError(msg)
        if skip_nt < 0:
            msg = "skip_nt must be non-negative"
            raise ValueError(msg)
        if snapshot_dt is not None and snapshot_dt <= 0:
            msg = "snapshot_dt must be positive when provided"
            raise ValueError(msg)
        if bc_mode not in ("periodic", "neumann"):
            msg = "bc_mode must be 'periodic' or 'neumann'"
            raise ValueError(msg)
        if buoyancy_mode not in ("anomaly", "raw"):
            msg = "buoyancy_mode must be 'anomaly' or 'raw'"
            raise ValueError(msg)

        self.return_timeseries = return_timeseries
        self.n = n
        self.L = L
        self.T = T
        self.dt = dt
        self.snapshot_dt = snapshot_dt
        self.nu = nu
        self.cfl = cfl
        self.bc_mode = bc_mode
        self.buoyancy_mode = buoyancy_mode
        self.skip_nt = skip_nt
        self.random_seed = random_seed
        self._rng = (
            torch.Generator().manual_seed(random_seed)
            if random_seed is not None
            else None
        )

    def _extract_run_parameters(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, float, float, float]:
        param_values = {
            name: float(x[0, idx].item()) for idx, name in enumerate(self.param_names)
        }

        buoyancy_y = param_values["buoyancy_y"]
        smoothness = float(param_values.get("smoothness", self._DEFAULT_SMOOTHNESS))
        smoothness = max(smoothness, 0.0)
        noise_scale = float(param_values.get("noise_scale", self._DEFAULT_NOISE_SCALE))
        smoke_diffusivity = float(
            param_values.get("smoke_diffusivity", self._DEFAULT_SMOKE_DIFFUSIVITY)
        )

        buoyancy_tensor = torch.tensor([buoyancy_y], dtype=x.dtype, device=x.device)
        return buoyancy_tensor, smoothness, noise_scale, smoke_diffusivity

    def _forward(self, x: TensorLike) -> TensorLike:
        if x.shape[0] != 1:
            msg = "ConditionedNavierStokes2D expects a single input sample"
            raise ValueError(msg)

        x = torch.as_tensor(x, dtype=torch.float32)
        buoyancy_param, smoothness, noise_scale, smoke_diffusivity = (
            self._extract_run_parameters(x)
        )

        seed = None
        if self._rng is not None:
            seed = int(torch.randint(0, 2**31 - 1, (1,), generator=self._rng).item())

        sol = simulate_conditioned_navier_stokes_2d(
            params=buoyancy_param,
            return_timeseries=self.return_timeseries,
            n=self.n,
            L=self.L,
            T=self.T,
            dt=self.dt,
            snapshot_dt=self.snapshot_dt,
            nu=self.nu,
            smoke_diffusivity=smoke_diffusivity,
            cfl=self.cfl,
            smoothness=smoothness,
            noise_scale=noise_scale,
            bc_mode=self.bc_mode,
            buoyancy_mode=self.buoyancy_mode,
            random_seed=seed,
        )
        return sol.flatten().unsqueeze(0)

    def forward_samples_spatiotemporal(
        self,
        n: int,
        random_seed: int | None = None,
        ensure_exact_n: bool = False,
    ) -> dict:
        """Run sampled trajectories and return `[batch,time,x,y,channels]` data."""
        y, x = self._forward_batch_with_optional_retries(
            n=n,
            random_seed=random_seed,
            ensure_exact_n=ensure_exact_n,
        )

        channels = 3
        features_per_step = self.n * self.n * channels

        if self.return_timeseries:
            total_features = y.shape[1]
            if total_features % features_per_step != 0:
                raise RuntimeError(
                    "Returned tensor does not align with n*n*channels; "
                    f"received {total_features}, expected multiples of "
                    f"{features_per_step}."
                )
            n_time = total_features // features_per_step
            y_reshaped = y.reshape(y.shape[0], n_time, self.n, self.n, channels)

            start_idx = 1 + self.skip_nt
            if start_idx >= n_time:
                raise ValueError(
                    "skip_nt is too large for the available trajectory length; "
                    f"computed start index {start_idx} for n_time={n_time}."
                )
            y_reshaped = y_reshaped[:, start_idx:, ...]
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
