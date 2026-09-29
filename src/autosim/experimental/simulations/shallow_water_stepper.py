"""Differentiable, fixed-step, forcing-free evolution of physical SWE states."""

from __future__ import annotations

import math

import torch

from autosim.experimental.simulations._shallow_water_dynamics import (
    Fields,
    ShallowWaterDynamics,
    rk4_step,
)
from autosim.experimental.simulations._spectral import (
    spectral_wavenumbers,
    two_thirds_mask,
)
from autosim.experimental.simulations.shallow_water import (
    CORIOLIS_MODES,
    H_MAX_CLIP,
    H_MIN_CLIP,
    MIN_GRID_SIZE,
    N_HYPERVISC,
    UV_ABS_CLIP,
    _coriolis_grid,
)


def _check_fields(h: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> None:
    """Reject states that would need the generator's safety transformations."""
    valid = (
        torch.isfinite(h).all()
        & torch.isfinite(u).all()
        & torch.isfinite(v).all()
        & (h > H_MIN_CLIP).all()
        & (h < H_MAX_CLIP).all()
        & (u.abs() < UV_ABS_CLIP).all()
        & (v.abs() < UV_ABS_CLIP).all()
    )
    if not bool(valid):
        msg = "SWE state is non-finite or outside the valid unclipped physical range"
        raise ValueError(msg)


def advance_swe_2d(  # noqa: PLR0915
    state: torch.Tensor,
    dt: float,
    *,
    n_substeps: int,
    Lx: float,
    Ly: float,
    g: float = 9.81,
    h_mean: float = 1.0,
    nu: float = 5e-4,
    drag: float = 2e-3,
    f0: float | None = None,
    beta: float | None = None,
    coriolis_mode: str = "periodic_beta",
    dealias: bool = True,
    cfl: float = 0.12,
) -> torch.Tensor:
    """Advance a physical state without forcing, retaining its autograd graph.

    Uses the data generator's RHS, RK4, rectangular spectral projection and
    order-eight hyperviscosity. Unlike ``simulate_swe_2d``, this interface uses
    an explicit fixed schedule, preserves precision/device, supports leading
    batch/member axes and never detaches or sanitizes an incoming state.

    Args:
        state: Float32/float64 physical fields shaped ``(..., nx, ny, 3)`` in
            ``[h, u, v]`` order. Gradients are supported with respect to this
            tensor, including through repeated calls. Do not pass normalized
            model channels directly.
        dt: Non-negative forecast interval in simulator time units.
        n_substeps: Positive fixed count of RK4/filter steps over ``dt``.
            This is not adaptive; insufficient resolution raises an error.
        Lx: Positive periodic domain length in x.
        Ly: Positive periodic domain length in y.
        g: Non-negative gravity coefficient.
        h_mean: Positive reference depth, used to resolve the default f0.
        nu: Non-negative Laplacian viscosity.
        drag: Non-negative linear drag.
        f0: Reference Coriolis parameter; default ``sqrt(g*h_mean)/8``.
        beta: Periodic Coriolis variation; default ``0.5*f0/Ly``.
        coriolis_mode: ``"f_plane"`` or ``"periodic_beta"``.
        dealias: Project each RK-stage state to the rectangular two-thirds
            band before evaluating nonlinear products, as in the generator.
        cfl: Positive wave/advection CFL bound, checked at every RK stage.

    Returns:
        Forecast with the same shape, dtype and device as ``state``. Physical
        parameters are fixed Python values, not trainable tensors. No random
        numbers are consumed and the input is not modified.

    Raises:
        TypeError: Physics or time parameters are supplied as tensors rather
            than fixed Python scalars.
        ValueError: Invalid configuration, invalid raw state at any RK stage
            or after filtering, or a timestep violating the CFL/viscosity bound.
            These validation branches are not differentiated. The retained
            state evolution is differentiable within the accepted domain.

    Fixed and adaptive schedules need not be bitwise equal over a full lead;
    choose ``n_substeps`` using a convergence check. Boundary clipping is
    deliberately rejected rather than providing misleading clipped gradients.
    This first interface does not promise ``torch.compile``/``vmap`` support.
    """
    if state.ndim < 3 or state.shape[-1] != 3 or state.numel() == 0:
        msg = "state must have nonempty shape (..., nx, ny, 3)"
        raise ValueError(msg)
    nx, ny = state.shape[-3:-1]
    if min(nx, ny) < MIN_GRID_SIZE:
        msg = f"nx and ny must be at least {MIN_GRID_SIZE}"
        raise ValueError(msg)
    if state.dtype not in (torch.float32, torch.float64):
        msg = "state must use torch.float32 or torch.float64"
        raise ValueError(msg)
    if (
        not isinstance(n_substeps, int)
        or isinstance(n_substeps, bool)
        or n_substeps <= 0
    ):
        msg = "n_substeps must be a positive integer"
        raise ValueError(msg)
    values = (dt, Lx, Ly, g, h_mean, nu, drag, cfl)
    if any(isinstance(value, torch.Tensor) for value in (*values, f0, beta)):
        msg = "physics and time parameters must be Python scalars, not tensors"
        raise TypeError(msg)
    if not all(math.isfinite(value) for value in values):
        msg = "time, geometry and physics parameters must be finite"
        raise ValueError(msg)
    if min(dt, g, nu, drag) < 0 or min(Lx, Ly, h_mean, cfl) <= 0:
        msg = "invalid time, geometry or physics parameter range"
        raise ValueError(msg)
    f0 = math.sqrt(g * h_mean) / 8 if f0 is None else f0
    beta = 0.5 * f0 / Ly if beta is None else beta
    if not math.isfinite(f0) or not math.isfinite(beta):
        msg = "f0 and beta must be finite"
        raise ValueError(msg)
    if coriolis_mode not in CORIOLIS_MODES:
        msg = f"coriolis_mode must be one of {CORIOLIS_MODES}"
        raise ValueError(msg)

    h, u, v = state.unbind(dim=-1)
    _check_fields(h, u, v)
    if dt == 0:
        return state

    dtype, device = state.dtype, state.device
    Kx, Ky, dKx, dKy = spectral_wavenumbers(nx, ny, Lx, Ly, dtype, device=device)
    K2 = Kx**2 + Ky**2
    mask = (
        two_thirds_mask(nx, ny, device=device)
        if dealias
        else torch.ones_like(K2).bool()
    )
    x = torch.linspace(0.0, Lx, nx + 1, dtype=dtype, device=device)[:-1]
    y = torch.linspace(0.0, Ly, ny + 1, dtype=dtype, device=device)[:-1]
    _, Y = torch.meshgrid(x, y, indexing="ij")
    f_grid = _coriolis_grid(Y, f0=f0, beta=beta, Ly=Ly, mode=coriolis_mode)
    k_max = math.pi * max(nx / Lx, ny / Ly)
    nu_h = 1.0 / k_max ** (2 * N_HYPERVISC)
    operators = ShallowWaterDynamics(
        nx=nx,
        ny=ny,
        iKx=1j * dKx,
        iKy=1j * dKy,
        K2=K2,
        dealias_mask=mask,
        hyp_op=-nu_h * K2**N_HYPERVISC,
        f_grid=f_grid,
        g=g,
        nu=nu,
        drag=drag,
        height_floor=H_MIN_CLIP,
    )
    step_dt = dt / n_substeps
    if nu > 0 and step_dt * nu * float(K2[mask].max()) > 2.5:
        msg = "Fixed step exceeds the viscosity bound; increase n_substeps"
        raise ValueError(msg)
    dx = min(Lx / nx, Ly / ny)

    def rhs(h: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> Fields:
        _check_fields(h, u, v)
        if dealias:
            # Learned corrections need not be band-limited. Project operands
            # before multiplying, and reject any resulting nonphysical state.
            h, u, v = (operators.project(field) for field in (h, u, v))
            _check_fields(h, u, v)
        wave_speed = torch.sqrt(g * h)
        max_speed = torch.maximum(
            (u.abs() + wave_speed).max(), (v.abs() + wave_speed).max()
        )
        if bool(step_dt * max_speed > cfl * dx):
            msg = "Fixed step exceeds the CFL bound; increase n_substeps"
            raise ValueError(msg)
        return operators.rhs(h, u, v)

    for _ in range(n_substeps):
        h, u, v = rk4_step(rhs, h, u, v, step_dt)
        _check_fields(h, u, v)
        h, u, v = operators.filter(h, u, v, step_dt)
        _check_fields(h, u, v)
    return torch.stack((h, u, v), dim=-1)
