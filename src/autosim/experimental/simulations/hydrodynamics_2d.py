"""
Simplified 2D hydrodynamics simulator (velocity + pressure only).

This module provides an MHD-style spatiotemporal generator without magnetic fields.
It evolves incompressible 2D flow with a pressure-projection method on a periodic grid.

Returned channels are: [u, v, p].
"""

from __future__ import annotations

import math

import torch

from autosim.simulations.base import SpatioTemporalSimulator
from autosim.types import TensorLike


class Hydrodynamics2D(SpatioTemporalSimulator):
    r"""Simplified 2D hydrodynamics simulator with no magnetic field.

    Parameters
    ----------
    parameters_range: dict[str, tuple[float, float]], optional
        Bounds on sampled parameters:
        - ``nu``: kinematic viscosity
        - ``force``: forcing amplitude
    output_names: list[str], optional
        Names for output channels. Defaults to ``["u", "v", "p"]``.
    return_timeseries: bool, default=False
        If True, returns full trajectory; otherwise final frame only.
    log_level: str, default="progress_bar"
        Logging level passed to base Simulator.
    n: int, default=64
        Grid resolution per axis.
    L: float, default=1.0
        Domain size in each direction.
    T: float, default=1.0
        Total simulation time.
    dt: float, default=0.01
        Fixed integration step.

    Notes
    -----
    Output shape before flattening:
    - timeseries: ``(nt, n, n, 3)``
    - final only: ``(n, n, 3)``
    """

    def __init__(
        self,
        parameters_range: dict[str, tuple[float, float]] | None = None,
        output_names: list[str] | None = None,
        return_timeseries: bool = False,
        log_level: str = "progress_bar",
        n: int = 64,
        L: float = 1.0,
        T: float = 1.0,
        dt: float = 0.01,
        cfl: float = 0.35,
    ) -> None:
        if parameters_range is None:
            parameters_range = {
                "nu": (1e-3, 8e-3),
                "force": (0.0, 0.4),
            }
        if output_names is None:
            output_names = ["u", "v", "p"]

        super().__init__(parameters_range, output_names, log_level)

        self.return_timeseries = return_timeseries
        self.n = n
        self.L = L
        self.T = T
        self.dt = dt
        self.cfl = cfl

    def _forward(self, x: TensorLike) -> TensorLike:
        assert x.shape[0] == 1, (
            f"Simulator._forward expects a single input, got {x.shape[0]}"
        )

        sample = x[0]
        out = simulate_hydrodynamics_2d(
            params=sample,
            return_timeseries=self.return_timeseries,
            n=self.n,
            L=self.L,
            T=self.T,
            dt=self.dt,
            cfl=self.cfl,
        )
        return out.flatten().unsqueeze(0)

    def forward_samples_spatiotemporal(
        self, n: int, random_seed: int | None = None
    ) -> dict:
        """Run sampled trajectories and return spatiotemporal tensors."""
        x = self.sample_inputs(n, random_seed)
        y, x = self.forward_batch(x)

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


def _grad_x(field: torch.Tensor, dx: float) -> torch.Tensor:
    return (torch.roll(field, -1, dims=0) - torch.roll(field, 1, dims=0)) / (2.0 * dx)


def _grad_y(field: torch.Tensor, dx: float) -> torch.Tensor:
    return (torch.roll(field, -1, dims=1) - torch.roll(field, 1, dims=1)) / (2.0 * dx)


def _laplacian(field: torch.Tensor, dx: float) -> torch.Tensor:
    return (
        (torch.roll(field, -1, dims=0) - 2.0 * field + torch.roll(field, 1, dims=0))
        + (torch.roll(field, -1, dims=1) - 2.0 * field + torch.roll(field, 1, dims=1))
    ) / (dx**2)


def _poisson_solve_periodic(rhs: torch.Tensor, L: float) -> torch.Tensor:
    """Solve ∇²_h p = rhs on a periodic square grid using FFT.

    Uses the modified wavenumbers of the central-difference Laplacian so
    that the discrete projection is exactly consistent with the FD
    gradient operators used elsewhere in the solver.
    """
    n = rhs.shape[0]
    dx = L / n

    # Wavenumber indices
    k = 2.0 * math.pi * torch.fft.fftfreq(n, d=dx, device=rhs.device, dtype=rhs.dtype)
    kx, ky = torch.meshgrid(k, k, indexing="ij")

    # Modified wavenumbers for the 5-point Laplacian: (2 - 2*cos(k*dx)) / dx²
    # This is the exact Fourier symbol of the discrete Laplacian used in
    # _laplacian(), ensuring ∇_h · u^{n+1} = 0 to machine precision.
    kx_mod = 2.0 * (1.0 - torch.cos(kx * dx)) / (dx * dx)
    ky_mod = 2.0 * (1.0 - torch.cos(ky * dx)) / (dx * dx)
    k2 = kx_mod + ky_mod

    rhs_hat = torch.fft.fft2(rhs)
    p_hat = -rhs_hat / torch.where(k2 > 0, k2, torch.ones_like(k2))
    p_hat[0, 0] = 0.0

    return torch.fft.ifft2(p_hat).real


def _stable_timestep(
    u: torch.Tensor,
    v: torch.Tensor,
    dx: float,
    nu: float,
    dt_max: float,
    cfl: float,
    t: float,
    T: float,
) -> float:
    speed = torch.sqrt(u**2 + v**2)
    umax = float(speed.max().item())
    adv_dt = cfl * dx / max(umax, 1e-8)
    diff_dt = 0.25 * dx * dx / max(nu, 1e-8)
    return min(dt_max, adv_dt, diff_dt, T - t)


def _advance_one_step(
    u: torch.Tensor,
    v: torch.Tensor,
    fx: torch.Tensor,
    fy: torch.Tensor,
    dx: float,
    dt_step: float,
    nu: float,
    L: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    du_dx = _grad_x(u, dx)
    du_dy = _grad_y(u, dx)
    dv_dx = _grad_x(v, dx)
    dv_dy = _grad_y(v, dx)

    adv_u = u * du_dx + v * du_dy
    adv_v = u * dv_dx + v * dv_dy

    u_star = u + dt_step * (-adv_u + nu * _laplacian(u, dx) + fx)
    v_star = v + dt_step * (-adv_v + nu * _laplacian(v, dx) + fy)

    div_star = _grad_x(u_star, dx) + _grad_y(v_star, dx)
    p_next = _poisson_solve_periodic(div_star / dt_step, L)

    u_next = u_star - dt_step * _grad_x(p_next, dx)
    v_next = v_star - dt_step * _grad_y(p_next, dx)
    return u_next, v_next, p_next


def simulate_hydrodynamics_2d(
    params: TensorLike,
    return_timeseries: bool = False,
    n: int = 64,
    L: float = 1.0,
    T: float = 1.0,
    dt: float = 0.01,
    cfl: float = 0.35,
) -> TensorLike:
    """Simulate a simplified 2D incompressible flow with forcing.

    Parameters
    ----------
    params: TensorLike
        ``[nu, force]`` where
        - ``nu``: kinematic viscosity
        - ``force``: forcing amplitude
    return_timeseries: bool
        Return full trajectory if True, final frame otherwise.

    Returns
    -------
    TensorLike
        Timeseries ``(nt, n, n, 3)`` or final snapshot ``(n, n, 3)`` with
        channels ``[u, v, p]``.
    """
    nu = float(params[0].item())
    force_amp = float(params[1].item())

    device = params.device if isinstance(params, torch.Tensor) else torch.device("cpu")
    dtype = torch.float32

    x = torch.linspace(0.0, L, n + 1, device=device, dtype=dtype)[:-1]
    y = torch.linspace(0.0, L, n + 1, device=device, dtype=dtype)[:-1]
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    dx = L / n

    # Random initial condition (divergence-free via projection)
    # White noise in velocity
    u = torch.rand((n, n), device=device, dtype=dtype) - 0.5
    v = torch.rand((n, n), device=device, dtype=dtype) - 0.5

    # Project to be divergence-free
    # We use the same projection logic as in the stepper, but initializing p=0
    # to begin with implies we just project (u,v).
    # 1. Calculate div
    div = _grad_x(u, dx) + _grad_y(v, dx)
    # 2. Solve Poisson for 'correction potential' phi
    phi = _poisson_solve_periodic(div, L)
    # 3. Correct velocity
    u = u - _grad_x(phi, dx)
    v = v - _grad_y(phi, dx)

    # Scale initial energy
    u = u * 0.5
    v = v * 0.5

    p = torch.zeros_like(u)

    # Kolmogorov Forcing: F = (sin(k*y), 0)
    # This creates shear bands that become unstable and roll up into vortices
    k_force = 4.0 * math.pi
    fx = force_amp * torch.sin(k_force * yy / L)
    fy = torch.zeros_like(xx)

    save_dt = max(1e-6, dt)
    history: list[torch.Tensor] = []

    def _snapshot(
        u_field: torch.Tensor,
        v_field: torch.Tensor,
        p_field: torch.Tensor,
    ) -> torch.Tensor:
        return torch.stack([u_field, v_field, p_field], dim=-1)

    t = 0.0
    next_save_t = 0.0

    with torch.no_grad():
        if return_timeseries:
            history.append(_snapshot(u, v, p))
            next_save_t = save_dt

        while t < T - 1e-12:
            dt_step = _stable_timestep(
                u=u,
                v=v,
                dx=dx,
                nu=nu,
                dt_max=dt,
                cfl=cfl,
                t=t,
                T=T,
            )
            u, v, p = _advance_one_step(
                u=u,
                v=v,
                fx=fx,
                fy=fy,
                dx=dx,
                dt_step=dt_step,
                nu=nu,
                L=L,
            )

            t += dt_step

            if (
                return_timeseries
                and next_save_t <= t + 1e-12
                and next_save_t <= T + 1e-12
            ):
                history.append(_snapshot(u, v, p))
                while next_save_t <= t + 1e-12:
                    next_save_t += save_dt

    final = torch.stack([u, v, p], dim=-1)

    if return_timeseries:
        if history == []:
            history.append(final)
        return torch.stack(history, dim=0)
    return final
