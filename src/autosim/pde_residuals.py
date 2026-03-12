from __future__ import annotations

import math
from typing import Any

import torch


def _central_diff_x(f: torch.Tensor, dx: float) -> torch.Tensor:
    return (torch.roll(f, shifts=-1, dims=-2) - torch.roll(f, shifts=1, dims=-2)) / (
        2.0 * dx
    )


def _central_diff_y(f: torch.Tensor, dy: float) -> torch.Tensor:
    return (torch.roll(f, shifts=-1, dims=-1) - torch.roll(f, shifts=1, dims=-1)) / (
        2.0 * dy
    )


def _laplacian(f: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    f_xx = (
        torch.roll(f, shifts=-1, dims=-2)
        - 2.0 * f
        + torch.roll(f, shifts=1, dims=-2)
    ) / (dx * dx)
    f_yy = (
        torch.roll(f, shifts=-1, dims=-1)
        - 2.0 * f
        + torch.roll(f, shifts=1, dims=-1)
    ) / (dy * dy)
    return f_xx + f_yy


def shallow_water_diagnostics(
    payload: dict[str, Any],
    *,
    simulator: Any | None = None,
    simulator_kwargs: dict[str, Any] | None = None,  # noqa: ARG001
) -> dict[str, float]:
    """Diagnostics for `autosim.experimental.simulations.ShallowWater2D`.

    Expects payload["data"] shaped [batch,time,nx,ny,channels={h,u,v}].
    """
    data = payload["data"]
    if not isinstance(data, torch.Tensor) or data.ndim != 5 or data.shape[-1] < 3:
        raise ValueError("Expected payload['data'] as [b,t,nx,ny,3+] torch.Tensor")

    h = data[..., 0]
    u = data[..., 1]
    v = data[..., 2]

    eps = 1e-12
    frac_h_nonpos = float((h <= 0.0).float().mean().item())
    frac_h_small = float((h <= 1e-4).float().mean().item())
    max_uv = float(torch.maximum(u.abs(), v.abs()).max().item())
    mean_h = float(h.mean().item())
    min_h = float(h.min().item())

    out = {
        "min_h": min_h,
        "mean_h": mean_h,
        "frac_h_nonpos": frac_h_nonpos,
        "frac_h_le_1e-4": frac_h_small,
        "max_abs_uv": max_uv,
    }

    if simulator is not None and hasattr(simulator, "Lx") and hasattr(simulator, "Ly"):
        nx = int(data.shape[2])
        ny = int(data.shape[3])
        dx = float(simulator.Lx) / float(nx)
        dy = float(simulator.Ly) / float(ny)
        div = _central_diff_x(u, dx) + _central_diff_y(v, dy)
        out["div_uv_l2"] = float(torch.linalg.vector_norm(div).item())
        out["div_uv_rel_l2"] = float(
            torch.linalg.vector_norm(div).item()
            / (torch.linalg.vector_norm(u).item() + torch.linalg.vector_norm(v).item() + eps)
        )

    return out


def shallow_water_residual(
    payload: dict[str, Any],
    *,
    simulator: Any | None = None,
    simulator_kwargs: dict[str, Any] | None = None,  # noqa: ARG001
) -> dict[str, float]:
    """Discrete PDE residual for the full 2D shallow-water equations.

    Uses periodic central differences and a simple forward-difference time derivative:
      (q_{t+1}-q_t)/dt.

    This is an *approximate* residual intended for regression/sanity. It will not
    match the simulator’s internal pseudo-spectral operator exactly (and that’s OK).
    """
    data = payload["data"]
    if not isinstance(data, torch.Tensor) or data.ndim != 5 or data.shape[-1] < 3:
        raise ValueError("Expected payload['data'] as [b,t,nx,ny,3+] torch.Tensor")
    if data.shape[1] < 2:
        raise ValueError("Need at least 2 time steps to compute a time residual")
    if simulator is None:
        raise ValueError("shallow_water_residual requires simulator=ShallowWater2D")

    nx = int(data.shape[2])
    ny = int(data.shape[3])
    dx = float(simulator.Lx) / float(nx)
    dy = float(simulator.Ly) / float(ny)
    dt = float(getattr(simulator, "dt_save", 1.0))

    g = float(getattr(simulator, "g", 9.81))
    h_mean = float(getattr(simulator, "h_mean", 1.0))
    nu = float(getattr(simulator, "nu", 0.0))
    drag = float(getattr(simulator, "drag", 0.0))

    # Reconstruct beta-plane Coriolis f(y) consistent with simulator code.
    c = math.sqrt(g * h_mean)
    f0 = c / 8.0
    beta = 0.5 * f0 / float(simulator.Ly)
    y = torch.linspace(0.0, float(simulator.Ly), ny + 1, device=data.device, dtype=data.dtype)[
        :-1
    ]
    f_grid = (f0 + beta * (y - 0.5 * float(simulator.Ly))).reshape(1, 1, 1, ny)

    h = data[..., 0]
    u = data[..., 1]
    v = data[..., 2]

    # Time derivatives on [b,t-1,nx,ny]
    dh_dt = (h[:, 1:] - h[:, :-1]) / dt
    du_dt = (u[:, 1:] - u[:, :-1]) / dt
    dv_dt = (v[:, 1:] - v[:, :-1]) / dt

    # Evaluate RHS terms at t (use q_t = q[:,:-1])
    h_t = h[:, :-1]
    u_t = u[:, :-1]
    v_t = v[:, :-1]

    # Continuity: d_t h + div(h u) = 0
    hu = h_t * u_t
    hv = h_t * v_t
    div_hu = _central_diff_x(hu, dx) + _central_diff_y(hv, dy)
    r_h = dh_dt + div_hu

    # Momentum:
    # d_t u + u u_x + v u_y = f v - g h_x + nu lap(u) - drag u
    # d_t v + u v_x + v v_y = -f u - g h_y + nu lap(v) - drag v
    u_x = _central_diff_x(u_t, dx)
    u_y = _central_diff_y(u_t, dy)
    v_x = _central_diff_x(v_t, dx)
    v_y = _central_diff_y(v_t, dy)
    h_x = _central_diff_x(h_t, dx)
    h_y = _central_diff_y(h_t, dy)

    adv_u = u_t * u_x + v_t * u_y
    adv_v = u_t * v_x + v_t * v_y
    lap_u = _laplacian(u_t, dx, dy)
    lap_v = _laplacian(v_t, dx, dy)

    r_u = du_dt + adv_u - f_grid * v_t + g * h_x - nu * lap_u + drag * u_t
    r_v = dv_dt + adv_v + f_grid * u_t + g * h_y - nu * lap_v + drag * v_t

    def l2(x: torch.Tensor) -> float:
        return float(torch.linalg.vector_norm(x.float()).item())

    eps = 1e-12
    state_scale = l2(h_t) + l2(u_t) + l2(v_t) + eps
    res_scale = l2(r_h) + l2(r_u) + l2(r_v)

    return {
        "residual_h_l2": l2(r_h),
        "residual_u_l2": l2(r_u),
        "residual_v_l2": l2(r_v),
        "residual_total_l2_sum": float(res_scale),
        "residual_total_rel_l2_sum": float(res_scale / state_scale),
    }

