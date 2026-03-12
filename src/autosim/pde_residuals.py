from __future__ import annotations

import math
from typing import Any

import torch


def _get_param(
    payload: dict[str, Any], simulator: Any, name: str, *, default: float | None = None
) -> float:
    const = payload.get("constant_scalars")
    if isinstance(const, torch.Tensor) and hasattr(simulator, "get_parameter_idx"):
        if hasattr(simulator, "param_names") and name in simulator.param_names:
            idx = int(simulator.get_parameter_idx(name))
            return float(const[0, idx].item())
    if default is None:
        raise ValueError(f"Could not infer parameter {name!r} from payload/simulator")
    return float(default)


def _channel_index_by_name(simulator: Any, name: str) -> int | None:
    names = getattr(simulator, "output_names", None)
    if isinstance(names, list) and name in names:
        return int(names.index(name))
    return None


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
        torch.roll(f, shifts=-1, dims=-2) - 2.0 * f + torch.roll(f, shifts=1, dims=-2)
    ) / (dx * dx)
    f_yy = (
        torch.roll(f, shifts=-1, dims=-1) - 2.0 * f + torch.roll(f, shifts=1, dims=-1)
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
            / (
                torch.linalg.vector_norm(u).item()
                + torch.linalg.vector_norm(v).item()
                + eps
            )
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
    y = torch.linspace(
        0.0, float(simulator.Ly), ny + 1, device=data.device, dtype=data.dtype
    )[:-1]
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


def reaction_diffusion_residual(
    payload: dict[str, Any],
    *,
    simulator: Any | None = None,
    simulator_kwargs: dict[str, Any] | None = None,  # noqa: ARG001
) -> dict[str, float]:
    """Approximate residual for the repo's two-species reaction-diffusion system.

    Uses periodic central differences for the Laplacian and forward time differences.
    PDE form (in physical space) follows the implementation in `simulations/reaction_diffusion.py`.
    """
    data = payload["data"]
    if not isinstance(data, torch.Tensor) or data.ndim != 5 or data.shape[-1] < 2:
        raise ValueError("Expected payload['data'] as [b,t,n,n,2+] torch.Tensor")
    if data.shape[1] < 2:
        raise ValueError("Need at least 2 time steps to compute a time residual")
    if simulator is None:
        raise ValueError(
            "reaction_diffusion_residual requires simulator=ReactionDiffusion"
        )

    n = int(data.shape[2])
    L = float(getattr(simulator, "L", 1.0))
    dx = L / float(n)
    dy = dx
    dt = float(getattr(simulator, "dt", 1.0))

    beta = _get_param(payload, simulator, "beta")
    d = _get_param(payload, simulator, "d")
    d1 = float(d)
    d2 = float(d)

    u = data[..., 0]
    v = data[..., 1]

    du_dt = (u[:, 1:] - u[:, :-1]) / dt
    dv_dt = (v[:, 1:] - v[:, :-1]) / dt
    u_t = u[:, :-1]
    v_t = v[:, :-1]

    lap_u = _laplacian(u_t, dx, dy)
    lap_v = _laplacian(v_t, dx, dy)

    u3 = u_t**3
    v3 = v_t**3
    u2v = (u_t**2) * v_t
    uv2 = u_t * (v_t**2)

    ru = u_t - u3 - uv2 + beta * u2v + beta * v3
    rv = v_t - u2v - v3 - beta * u3 - beta * uv2

    r_u = du_dt - (d1 * lap_u + ru)
    r_v = dv_dt - (d2 * lap_v + rv)

    def l2(x: torch.Tensor) -> float:
        return float(torch.linalg.vector_norm(x.float()).item())

    eps = 1e-12
    state_scale = l2(u_t) + l2(v_t) + eps
    res_scale = l2(r_u) + l2(r_v)
    return {
        "residual_u_l2": l2(r_u),
        "residual_v_l2": l2(r_v),
        "residual_total_l2_sum": float(res_scale),
        "residual_total_rel_l2_sum": float(res_scale / state_scale),
    }


def advection_diffusion_multichannel_diagnostics(
    payload: dict[str, Any],
    *,
    simulator: Any | None = None,
    simulator_kwargs: dict[str, Any] | None = None,  # noqa: ARG001
) -> dict[str, float]:
    """Diagnostics for `AdvectionDiffusionMultichannel`.

    Uses whatever channels are present in `simulator.output_names`.
    """
    data = payload["data"]
    if not isinstance(data, torch.Tensor) or data.ndim != 5:
        raise ValueError("Expected payload['data'] as [b,t,n,n,c] torch.Tensor")

    idx_w = _channel_index_by_name(simulator, "vorticity") if simulator else None
    idx_u = _channel_index_by_name(simulator, "u") if simulator else None
    idx_v = _channel_index_by_name(simulator, "v") if simulator else None
    idx_psi = _channel_index_by_name(simulator, "streamfunction") if simulator else None

    out: dict[str, float] = {}
    if idx_w is not None:
        w = data[..., idx_w]
        out["omega_l2"] = float(torch.linalg.vector_norm(w.float()).item())
        out["omega_abs_max"] = float(w.abs().max().item())
    if idx_u is not None and idx_v is not None and simulator is not None:
        u = data[..., idx_u]
        v = data[..., idx_v]
        n = int(data.shape[2])
        L = float(getattr(simulator, "L", 1.0))
        dx = L / float(n)
        div = _central_diff_x(u, dx) + _central_diff_y(v, dx)
        out["div_uv_l2"] = float(torch.linalg.vector_norm(div.float()).item())
    if idx_w is not None and idx_psi is not None and simulator is not None:
        w = data[..., idx_w]
        psi = data[..., idx_psi]
        n = int(data.shape[2])
        L = float(getattr(simulator, "L", 1.0))
        dx = L / float(n)
        lap_psi = _laplacian(psi, dx, dx)
        poisson_res = lap_psi + w
        out["poisson_l2"] = float(torch.linalg.vector_norm(poisson_res.float()).item())
    return out


def advection_diffusion_multichannel_residual(
    payload: dict[str, Any],
    *,
    simulator: Any | None = None,
    simulator_kwargs: dict[str, Any] | None = None,  # noqa: ARG001
) -> dict[str, float]:
    """Residual for the vorticity PDE used by `AdvectionDiffusionMultichannel`.

    PDE (as implemented):  ω_t = ν ∇²ω - μ (u ω_x + v ω_y)
    with u = ψ_y, v = -ψ_x, periodic domain.
    """
    data = payload["data"]
    if not isinstance(data, torch.Tensor) or data.ndim != 5:
        raise ValueError("Expected payload['data'] as [b,t,n,n,c] torch.Tensor")
    if data.shape[1] < 2:
        raise ValueError("Need at least 2 time steps to compute a time residual")
    if simulator is None:
        raise ValueError(
            "advection_diffusion_multichannel_residual requires simulator=AdvectionDiffusionMultichannel"
        )

    idx_w = _channel_index_by_name(simulator, "vorticity")
    if idx_w is None:
        raise ValueError("Simulator outputs do not include 'vorticity'")
    idx_u = _channel_index_by_name(simulator, "u")
    idx_v = _channel_index_by_name(simulator, "v")
    idx_psi = _channel_index_by_name(simulator, "streamfunction")

    w = data[..., idx_w]
    if idx_u is not None and idx_v is not None:
        u = data[..., idx_u]
        v = data[..., idx_v]
    elif idx_psi is not None:
        psi = data[..., idx_psi]
        n = int(data.shape[2])
        L = float(getattr(simulator, "L", 1.0))
        dx = L / float(n)
        u = _central_diff_y(psi, dx)
        v = -_central_diff_x(psi, dx)
    else:
        raise ValueError(
            "Need either (u,v) or streamfunction channel to compute advection"
        )

    n = int(data.shape[2])
    L = float(getattr(simulator, "L", 1.0))
    dx = L / float(n)
    dt = float(getattr(simulator, "dt", 1.0))

    nu = _get_param(payload, simulator, "nu")
    mu = _get_param(payload, simulator, "mu")

    dw_dt = (w[:, 1:] - w[:, :-1]) / dt
    w_t = w[:, :-1]
    u_t = u[:, :-1]
    v_t = v[:, :-1]

    dw_dx = _central_diff_x(w_t, dx)
    dw_dy = _central_diff_y(w_t, dx)
    lap_w = _laplacian(w_t, dx, dx)
    rhs = nu * lap_w - mu * (u_t * dw_dx + v_t * dw_dy)
    r = dw_dt - rhs

    def l2(x: torch.Tensor) -> float:
        return float(torch.linalg.vector_norm(x.float()).item())

    eps = 1e-12
    state_scale = l2(w_t) + eps
    res = l2(r)
    return {
        "residual_omega_l2": res,
        "residual_omega_rel_l2": float(res / state_scale),
    }
