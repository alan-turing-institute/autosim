"""
Simple 2D compressible-fluid simulator (PDEBench-style fields).

State channels are [rho, u, v, p], evolved with a compact finite-volume
local Lax-Friedrichs (Rusanov) scheme for the 2D Euler equations on a periodic grid.
"""

from __future__ import annotations

import math

import torch

from autosim.simulations.base import SpatioTemporalSimulator
from autosim.types import TensorLike


class CompressibleFluid2D(SpatioTemporalSimulator):
    """Minimal 2D compressible Euler simulator.

    Parameters
    ----------
    parameters_range:
        ``gamma`` (adiabatic index), ``amp`` (initial perturbation amplitude).
    return_timeseries:
        If True, return full trajectory, otherwise final snapshot.
    n:
        Grid size (n x n).
    L:
        Domain length per axis.
    T:
        Final time.
    dt_save:
        Save interval when `return_timeseries=True`.
    cfl:
        CFL number for adaptive stepping.
    scenario:
        Initial-condition family. One of:
        - ``"shear_layers"`` (default): dual shear layers with multimode perturbations
        - ``"vortex_sheet"``: single shear sheet with sinusoidal displacement
        - ``"blast_wave"``: smooth radial over-pressure/density pulse
    flux_scheme:
        Numerical interface flux. One of:
        - ``"llf"``: local Lax-Friedrichs (more diffusive, robust)
        - ``"hll"``: HLL flux (less diffusive, sharper fronts)
    """

    def __init__(
        self,
        parameters_range: dict[str, tuple[float, float]] | None = None,
        output_names: list[str] | None = None,
        return_timeseries: bool = False,
        log_level: str = "progress_bar",
        n: int = 64,
        L: float = 1.0,
        T: float = 0.8,
        dt_save: float = 0.01,
        cfl: float = 0.35,
        scenario: str = "shear_layers",
        flux_scheme: str = "llf",
    ) -> None:
        if parameters_range is None:
            parameters_range = {
                "gamma": (1.35, 1.67),
                "amp": (0.02, 0.15),
            }
        if output_names is None:
            output_names = ["rho", "u", "v", "p"]

        super().__init__(parameters_range, output_names, log_level)
        self.return_timeseries = return_timeseries
        self.n = n
        self.L = L
        self.T = T
        self.dt_save = dt_save
        self.cfl = cfl
        self.scenario = scenario
        self.flux_scheme = flux_scheme

    def _forward(self, x: TensorLike) -> TensorLike:
        assert x.shape[0] == 1, "Simulator._forward expects a single input"
        gamma = float(x[0, 0].item())
        amp = float(x[0, 1].item())

        y = simulate_compressible_fluid_2d(
            gamma=gamma,
            amp=amp,
            return_timeseries=self.return_timeseries,
            n=self.n,
            L=self.L,
            T=self.T,
            dt_save=self.dt_save,
            cfl=self.cfl,
            scenario=self.scenario,
            flux_scheme=self.flux_scheme,
        )
        return y.flatten().unsqueeze(0)

    def forward_samples_spatiotemporal(  # noqa: D102
        self, n: int, random_seed: int | None = None
    ) -> dict:
        x = self.sample_inputs(n, random_seed)
        y, x = self.forward_batch(x)

        channels = 4
        features_per_step = self.n * self.n * channels

        if self.return_timeseries:
            total = y.shape[1]
            if total % features_per_step != 0:
                raise RuntimeError(
                    f"Unexpected flattened size {total}; "
                    f"expected multiple of {features_per_step}."
                )
            n_time = total // features_per_step
            y = y.reshape(n, n_time, self.n, self.n, channels)
        else:
            y = y.reshape(n, 1, self.n, self.n, channels)

        return {
            "data": y,
            "constant_scalars": x,
            "constant_fields": None,
        }


def _prim_to_cons(
    rho: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    mx = rho * u
    my = rho * v
    E = p / (gamma - 1.0) + 0.5 * rho * (u * u + v * v)
    return torch.stack([rho, mx, my, E], dim=0)


def _cons_to_prim(
    U: torch.Tensor,
    gamma: float,
    rho_floor: float = 1e-6,
    p_floor: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rho = U[0].clamp(min=rho_floor)
    u = U[1] / rho
    v = U[2] / rho
    kinetic = 0.5 * rho * (u * u + v * v)
    p = ((gamma - 1.0) * (U[3] - kinetic)).clamp(min=p_floor)
    return rho, u, v, p


def _flux_x(U: torch.Tensor, gamma: float) -> torch.Tensor:
    rho, u, v, p = _cons_to_prim(U, gamma)
    E = U[3]
    return torch.stack(
        [
            rho * u,
            rho * u * u + p,
            rho * u * v,
            (E + p) * u,
        ],
        dim=0,
    )


def _flux_y(U: torch.Tensor, gamma: float) -> torch.Tensor:
    rho, u, v, p = _cons_to_prim(U, gamma)
    E = U[3]
    return torch.stack(
        [
            rho * v,
            rho * u * v,
            rho * v * v + p,
            (E + p) * v,
        ],
        dim=0,
    )


def _signal_speed(U: torch.Tensor, gamma: float) -> tuple[torch.Tensor, torch.Tensor]:
    rho, u, v, p = _cons_to_prim(U, gamma)
    c = (gamma * p / rho).sqrt()
    return u.abs() + c, v.abs() + c


def _hll_flux_x(UL: torch.Tensor, UR: torch.Tensor, gamma: float) -> torch.Tensor:
    """HLL numerical flux in x-direction at interfaces between UL and UR."""
    FL = _flux_x(UL, gamma)
    FR = _flux_x(UR, gamma)

    rhoL, uL, _vL, pL = _cons_to_prim(UL, gamma)
    rhoR, uR, _vR, pR = _cons_to_prim(UR, gamma)
    cL = (gamma * pL / rhoL).sqrt()
    cR = (gamma * pR / rhoR).sqrt()

    sL = torch.minimum(uL - cL, uR - cR)
    sR = torch.maximum(uL + cL, uR + cR)

    sLz = sL.unsqueeze(0)
    sRz = sR.unsqueeze(0)
    denom = (sR - sL).unsqueeze(0).clamp(min=1e-8)
    FH = (sRz * FL - sLz * FR + sLz * sRz * (UR - UL)) / denom

    return torch.where(sLz >= 0.0, FL, torch.where(sRz <= 0.0, FR, FH))


def _hll_flux_y(UL: torch.Tensor, UR: torch.Tensor, gamma: float) -> torch.Tensor:
    """HLL numerical flux in y-direction at interfaces between UL and UR."""
    GL = _flux_y(UL, gamma)
    GR = _flux_y(UR, gamma)

    rhoL, _uL, vL, pL = _cons_to_prim(UL, gamma)
    rhoR, _uR, vR, pR = _cons_to_prim(UR, gamma)
    cL = (gamma * pL / rhoL).sqrt()
    cR = (gamma * pR / rhoR).sqrt()

    sL = torch.minimum(vL - cL, vR - cR)
    sR = torch.maximum(vL + cL, vR + cR)

    sLz = sL.unsqueeze(0)
    sRz = sR.unsqueeze(0)
    denom = (sR - sL).unsqueeze(0).clamp(min=1e-8)
    GH = (sRz * GL - sLz * GR + sLz * sRz * (UR - UL)) / denom

    return torch.where(sLz >= 0.0, GL, torch.where(sRz <= 0.0, GR, GH))


def _step(
    U: torch.Tensor,
    dx: float,
    dy: float,
    gamma: float,
    cfl: float,
    flux_scheme: str,
    dt_max: float = float("inf"),
) -> tuple[torch.Tensor, float]:
    sx, sy = _signal_speed(U, gamma)
    max_sx = float(sx.max().item())
    max_sy = float(sy.max().item())
    dt = cfl * min(dx / max(max_sx, 1e-8), dy / max(max_sy, 1e-8))
    dt = min(dt, dt_max)

    UxR = torch.roll(U, shifts=-1, dims=1)
    if flux_scheme == "llf":
        FxL = _flux_x(U, gamma)
        FxR = _flux_x(UxR, gamma)
        sxL, _ = _signal_speed(U, gamma)
        sxR, _ = _signal_speed(UxR, gamma)
        ax = torch.maximum(sxL, sxR).unsqueeze(0)
        F_half_x = 0.5 * (FxL + FxR) - 0.5 * ax * (UxR - U)
    elif flux_scheme == "hll":
        F_half_x = _hll_flux_x(U, UxR, gamma)
    else:
        msg = "flux_scheme must be one of ['llf', 'hll']"
        raise ValueError(msg)
    div_x = (F_half_x - torch.roll(F_half_x, shifts=1, dims=1)) / dx

    UyR = torch.roll(U, shifts=-1, dims=2)
    if flux_scheme == "llf":
        GyL = _flux_y(U, gamma)
        GyR = _flux_y(UyR, gamma)
        _, syL = _signal_speed(U, gamma)
        _, syR = _signal_speed(UyR, gamma)
        ay = torch.maximum(syL, syR).unsqueeze(0)
        G_half_y = 0.5 * (GyL + GyR) - 0.5 * ay * (UyR - U)
    else:
        G_half_y = _hll_flux_y(U, UyR, gamma)
    div_y = (G_half_y - torch.roll(G_half_y, shifts=1, dims=2)) / dy

    U_next = U - dt * (div_x + div_y)

    rho, u, v, p = _cons_to_prim(U_next, gamma)
    U_next = _prim_to_cons(rho, u, v, p, gamma)
    return U_next, dt


def simulate_compressible_fluid_2d(  # noqa: PLR0915
    gamma: float,
    amp: float,
    return_timeseries: bool,
    n: int,
    L: float,
    T: float,
    dt_save: float,
    cfl: float,
    scenario: str = "shear_layers",
    flux_scheme: str = "llf",
) -> torch.Tensor:
    """Run a simple 2D compressible Euler simulation with periodic boundaries."""
    dtype = torch.float32
    device = torch.device("cpu")

    x = torch.linspace(0.0, L, n + 1, dtype=dtype, device=device)[:-1]
    y = torch.linspace(0.0, L, n + 1, dtype=dtype, device=device)[:-1]
    xx, yy = torch.meshgrid(x, y, indexing="ij")

    if scenario == "shear_layers":
        y1 = 0.25 * L
        y2 = 0.75 * L
        shear_w = 0.035 * L
        base_u = 0.55

        u0 = base_u * (
            torch.tanh((yy - y1) / shear_w) - torch.tanh((yy - y2) / shear_w) - 1.0
        )
        v0 = (
            0.18
            * amp
            * torch.sin(4.0 * math.pi * xx / L)
            * torch.sin(2.0 * math.pi * yy / L)
        )

        mode1 = torch.sin(2.0 * math.pi * xx / L) * torch.cos(2.0 * math.pi * yy / L)
        mode2 = torch.cos(5.0 * math.pi * xx / L) * torch.sin(3.0 * math.pi * yy / L)
        mode3 = torch.sin(6.0 * math.pi * (xx + yy) / L)

        rho0 = 1.0 + amp * (0.70 * mode1 + 0.20 * mode2 + 0.10 * mode3)
        p0 = 1.0 + 0.60 * amp * mode1 - 0.30 * amp * mode2

    elif scenario == "vortex_sheet":
        y_mid = 0.5 * L
        shear_w = 0.02 * L
        sheet_disp = 0.06 * L * torch.sin(2.0 * math.pi * xx / L)

        u0 = torch.tanh((yy - y_mid - sheet_disp) / shear_w)
        u0 = 0.7 * u0
        v0 = 0.12 * amp * torch.sin(2.0 * math.pi * xx / L)

        rho0 = 1.0 + 0.35 * amp * torch.cos(2.0 * math.pi * yy / L)
        p0 = 1.0 + 0.25 * amp * torch.sin(4.0 * math.pi * xx / L)

    elif scenario == "blast_wave":
        x0 = 0.5 * L
        y0 = 0.5 * L
        r2 = (xx - x0) ** 2 + (yy - y0) ** 2
        sigma2 = (0.08 * L) ** 2
        pulse = torch.exp(-r2 / sigma2)

        u0 = 0.04 * amp * torch.sin(2.0 * math.pi * yy / L)
        v0 = -0.04 * amp * torch.sin(2.0 * math.pi * xx / L)
        rho0 = 1.0 + 0.50 * amp * pulse
        p0 = 1.0 + 1.20 * amp * pulse

    else:
        raise ValueError(
            f"Unknown scenario '{scenario}'. "
            "Expected one of ['shear_layers', 'vortex_sheet', 'blast_wave']."
        )

    rho0 = rho0.clamp(min=0.2)
    p0 = p0.clamp(min=0.2)

    U = _prim_to_cons(rho0, u0, v0, p0, gamma)

    dx = L / n
    dy = L / n

    history: list[torch.Tensor] = []

    def _snapshot(U_state: torch.Tensor) -> torch.Tensor:
        rho, u, v, p = _cons_to_prim(U_state, gamma)
        return torch.stack([rho, u, v, p], dim=-1)

    t = 0.0
    next_save = 0.0

    with torch.no_grad():
        if return_timeseries:
            history.append(_snapshot(U))
            next_save = dt_save

        while t < T - 1e-12:
            U, dt = _step(
                U,
                dx=dx,
                dy=dy,
                gamma=gamma,
                cfl=cfl,
                flux_scheme=flux_scheme,
                dt_max=T - t,
            )
            t += dt

            if return_timeseries and t + 1e-12 >= next_save:
                history.append(_snapshot(U))
                while next_save <= t + 1e-12:
                    next_save += dt_save

    if return_timeseries:
        return torch.stack(history, dim=0)
    return _snapshot(U)
