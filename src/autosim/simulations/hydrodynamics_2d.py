"""
Simplified 2D hydrodynamics simulator (velocity + pressure only).

This module provides an MHD-style spatiotemporal generator without magnetic fields.
It evolves incompressible 2D flow with a pressure-projection method on a periodic grid.

Returned channels are: [u, v, p].
"""

from __future__ import annotations

import math

import torch

from autosim.simulations.base import Simulator
from autosim.types import TensorLike


class Hydrodynamics2D(Simulator):
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
    ) -> None:
        if parameters_range is None:
            parameters_range = {
                "nu": (5e-4, 5e-3),
                "force": (0.0, 1.0),
            }
        if output_names is None:
            output_names = ["u", "v", "p"]

        super().__init__(parameters_range, output_names, log_level)

        self.return_timeseries = return_timeseries
        self.n = n
        self.L = L
        self.T = T
        self.dt = dt

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
    """Solve ∇²p = rhs on a periodic square grid using FFT."""
    n = rhs.shape[0]
    dx = L / n

    k = 2.0 * math.pi * torch.fft.fftfreq(n, d=dx, device=rhs.device, dtype=rhs.dtype)
    kx, ky = torch.meshgrid(k, k, indexing="ij")
    k2 = kx**2 + ky**2

    rhs_hat = torch.fft.fft2(rhs)
    p_hat = -rhs_hat / torch.where(k2 > 0, k2, torch.ones_like(k2))
    p_hat[0, 0] = 0.0

    return torch.fft.ifft2(p_hat).real


def simulate_hydrodynamics_2d(
    params: TensorLike,
    return_timeseries: bool = False,
    n: int = 64,
    L: float = 1.0,
    T: float = 1.0,
    dt: float = 0.01,
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

    u = torch.sin(2.0 * math.pi * yy / L)
    v = -torch.sin(2.0 * math.pi * xx / L)
    perturb = (
        0.1 * torch.sin(4.0 * math.pi * xx / L) * torch.sin(4.0 * math.pi * yy / L)
    )
    u = u + 0.25 * force_amp * perturb
    v = v - 0.25 * force_amp * perturb
    p = torch.zeros_like(u)

    fx = force_amp * torch.sin(2.0 * math.pi * yy / L)
    fy = -force_amp * torch.sin(2.0 * math.pi * xx / L)

    n_steps = max(1, int(T / dt))
    history: list[torch.Tensor] = []

    with torch.no_grad():
        for _ in range(n_steps):
            du_dx = _grad_x(u, dx)
            du_dy = _grad_y(u, dx)
            dv_dx = _grad_x(v, dx)
            dv_dy = _grad_y(v, dx)

            adv_u = u * du_dx + v * du_dy
            adv_v = u * dv_dx + v * dv_dy

            u_star = u + dt * (-adv_u + nu * _laplacian(u, dx) + fx)
            v_star = v + dt * (-adv_v + nu * _laplacian(v, dx) + fy)

            div_star = _grad_x(u_star, dx) + _grad_y(v_star, dx)
            p = _poisson_solve_periodic(div_star / dt, L)

            u = u_star - dt * _grad_x(p, dx)
            v = v_star - dt * _grad_y(p, dx)

            if return_timeseries:
                history.append(torch.stack([u, v, p], dim=-1))

    final = torch.stack([u, v, p], dim=-1)

    if return_timeseries:
        return torch.stack(history, dim=0)
    return final
