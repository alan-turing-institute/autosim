"""Shared deterministic SWE numerics for generation and differentiable stepping."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

Fields = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class ShallowWaterDynamics:
    """Fixed spectral operators; fields may have arbitrary leading dimensions."""

    nx: int
    ny: int
    iKx: torch.Tensor
    iKy: torch.Tensor
    K2: torch.Tensor
    dealias_mask: torch.Tensor
    hyp_op: torch.Tensor
    f_grid: torch.Tensor
    g: float
    nu: float
    drag: float
    height_floor: float

    def linear_timestep_limit(self) -> float:
        """Bound explicit RK4 steps for viscosity, drag and rotation together.

        The sum bounds the combined linear rate, including variable Coriolis
        frequency. Separate limits can admit unstable mixed damping/rotation.
        The factor 2.5 retains the existing viscous stability margin; this is
        a safeguard alongside the wave/advection CFL check, not an accuracy
        guarantee or a general nonlinear stability theorem.
        """
        viscous_rate = self.nu * float(self.K2[self.dealias_mask].max())
        rate = viscous_rate + self.drag + float(self.f_grid.abs().max())
        return 2.5 / rate if rate > 0 else math.inf

    def to_phys(self, field_hat: torch.Tensor) -> torch.Tensor:
        """Invert the last two Fourier axes using the original grid shape."""
        return torch.fft.irfft2(field_hat, s=(self.nx, self.ny))

    def project(self, field: torch.Tensor) -> torch.Tensor:
        """Apply the same resolved-mode projection as the data generator."""
        return self.to_phys(torch.fft.rfft2(field) * self.dealias_mask)

    def rhs(self, h: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> Fields:
        """Evaluate the existing conservative-depth and advective-velocity RHS."""
        h_safe = h.clamp(min=self.height_floor)
        u_h = torch.fft.rfft2(u)
        v_h = torch.fft.rfft2(v)
        h_h = torch.fft.rfft2(h)

        du_dx = self.to_phys(self.iKx * u_h)
        du_dy = self.to_phys(self.iKy * u_h)
        dv_dx = self.to_phys(self.iKx * v_h)
        dv_dy = self.to_phys(self.iKy * v_h)
        dh_dx = self.to_phys(self.iKx * h_h)
        dh_dy = self.to_phys(self.iKy * h_h)
        lap_u = self.to_phys(-self.K2 * u_h)
        lap_v = self.to_phys(-self.K2 * v_h)

        hu_h = torch.fft.rfft2(h_safe * u)
        hv_h = torch.fft.rfft2(h_safe * v)
        div_hu = self.to_phys(self.iKx * hu_h) + self.to_phys(self.iKy * hv_h)
        dudt = (
            -(u * du_dx + v * du_dy)
            + self.f_grid * v
            - self.g * dh_dx
            + self.nu * lap_u
            - self.drag * u
        )
        dvdt = (
            -(u * dv_dx + v * dv_dy)
            - self.f_grid * u
            - self.g * dh_dy
            + self.nu * lap_v
            - self.drag * v
        )
        return self.project(-div_hu), self.project(dudt), self.project(dvdt)

    def filter(
        self, h: torch.Tensor, u: torch.Tensor, v: torch.Tensor, dt: float
    ) -> Fields:
        """Apply existing hyperviscosity/dealiasing, before any depth floor."""
        factor = torch.exp(self.hyp_op * dt) * self.dealias_mask
        u = self.to_phys(torch.fft.rfft2(u) * factor)
        v = self.to_phys(torch.fft.rfft2(v) * factor)
        # Preserve the exact unbatched reduction used by the generator. Leading
        # dimensions in training must never share a depth mean with each other.
        mean = h.mean() if h.ndim == 2 else h.mean(dim=(-2, -1), keepdim=True)
        anomaly = self.to_phys(torch.fft.rfft2(h - mean) * factor)
        return mean + anomaly, u, v


def rk4_step(
    rhs: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], Fields],
    h: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    dt: float,
) -> Fields:
    """Advance the same four RK stages without mutation or state detachment."""
    k1_h, k1_u, k1_v = rhs(h, u, v)
    k2_h, k2_u, k2_v = rhs(
        h + 0.5 * dt * k1_h, u + 0.5 * dt * k1_u, v + 0.5 * dt * k1_v
    )
    k3_h, k3_u, k3_v = rhs(
        h + 0.5 * dt * k2_h, u + 0.5 * dt * k2_u, v + 0.5 * dt * k2_v
    )
    k4_h, k4_u, k4_v = rhs(h + dt * k3_h, u + dt * k3_u, v + dt * k3_v)
    return (
        h + (dt / 6.0) * (k1_h + 2.0 * k2_h + 2.0 * k3_h + k4_h),
        u + (dt / 6.0) * (k1_u + 2.0 * k2_u + 2.0 * k3_u + k4_u),
        v + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v),
    )
