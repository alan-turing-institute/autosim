"""Initial conditions of the shallow water equations on the sphere."""

from __future__ import annotations

import math

import torch

from autosim.types import TensorLike

from .dynamics import SphereShallowWaterDynamics

GALEWSKY_RANGES: dict[str, tuple[float, float]] = {
    "jet_speed": (60.0, 100.0),  # u_max [m/s], paper: 80
    "height_perturbation_m": (60.0, 180.0),  # bump amplitude [m], paper: 120
    "perturbation_lon": (0.0, 2 * math.pi),  # bump longitude [rad], paper: 0
    "perturbation_lat": (math.pi / 5, math.pi / 3),  # bump latitude [rad], paper: pi/4
}


def galewsky_initial_condition(
    dynamics: SphereShallowWaterDynamics,
    jet_speed: TensorLike,
    height_perturbation: TensorLike,
    perturbation_lon: TensorLike,
    perturbation_lat: TensorLike,
    noise_amplitude: float,
    noise_lmax: int,
    device: str = "cuda",
) -> TensorLike:
    """Galewsky et al. (2004) jet, bump and height noise for B simulations.

    Jet (their eq. 2), balanced height with a global mean depth ``mean_depth``
    (eq. 3, computed spectrally from the steady divergence equation), bump
    h' = h_hat cos(phi) exp(-(dlon/alpha)^2) exp(-((phi_2 - phi)/beta)^2) (eq. 4) with
    alpha = 1/3, beta = 1/15, and a random height noise below degree ``noise_lmax``
    drawn from the global torch generator.

    Args:
        dynamics: Fixed operators.
        jet_speed: Maximum jet speed u_max [m/s], shape (B,).
        height_perturbation: Bump amplitude h_hat [m], shape (B,).
        perturbation_lon: Bump centre longitude [rad], shape (B,).
        perturbation_lat: Bump centre latitude phi_2 [rad], shape (B,).
        noise_amplitude: Root-mean-square of the height noise [m].
        noise_lmax: The noise has degrees and orders below this value.
        device: Torch device of the state.

    Returns:
        Spectral state (B, 3, L, L).
    """
    batch = jet_speed.shape[0]
    lmax = dynamics.laplacian.shape[0]
    gravity = dynamics.gravity

    # Latitude and longitude of every grid point, shape (1, *grid.shape)
    grid_shape = dynamics.grid.shape
    coords = dynamics.grid.coords.to(device=device, dtype=torch.float64)
    latitude = (torch.pi / 2 - coords[:, 0]).reshape(1, *grid_shape)
    longitude = coords[:, 1].reshape(1, *grid_shape)

    # Parameters as shape (B, 1, ...) so they broadcast over the grid
    parameter_shape = (batch,) + (1,) * len(grid_shape)
    jet_speed = jet_speed.to(device, torch.float64).reshape(parameter_shape)
    height_perturbation = height_perturbation.to(device, torch.float64).reshape(
        parameter_shape
    )
    perturbation_lon = perturbation_lon.to(device, torch.float64).reshape(
        parameter_shape
    )
    perturbation_lat = perturbation_lat.to(device, torch.float64).reshape(
        parameter_shape
    )

    # 1. Zonal jet, eq. (2)
    phi0 = torch.pi / 7
    phi1 = torch.pi / 2 - phi0
    en = math.exp(-4 / (phi1 - phi0) ** 2)
    inside = (latitude > phi0) & (latitude < phi1)
    product = torch.where(inside, (latitude - phi0) * (latitude - phi1), -1.0)
    u = torch.where(inside, jet_speed / en * torch.exp(1.0 / product), 0.0)
    wind = torch.stack([u, torch.zeros_like(u)], dim=1)
    vorticity, divergence = dynamics.vorticity_divergence(wind)
    # The jet as represented with degrees l < L: balancing this wind (not the analytic
    # one) makes the represented jet exactly steady at any resolution.
    wind = dynamics.wind(vorticity, divergence)

    # 2. Balanced geopotential, eq. (3): lap^-1 curl((zeta + f) u) - |u|^2 / 2 + const
    absolute_vorticity = dynamics.to_grid(vorticity) + dynamics.coriolis
    flux_curl, _ = dynamics.vorticity_divergence(wind * absolute_vorticity.unsqueeze(1))
    kinetic_energy = 0.5 * (wind[:, 0] ** 2 + wind[:, 1] ** 2)
    geopotential = dynamics.inverse_laplacian * flux_curl - dynamics.to_coefficients(
        kinetic_energy
    )
    geopotential[:, 0, 0] = math.sqrt(4 * math.pi) * gravity * dynamics.mean_depth

    # 3. Bump, eq. (4)
    alpha = 1 / 3
    beta = 1 / 15
    delta_lon = (
        torch.remainder(longitude - perturbation_lon + torch.pi, 2 * torch.pi)
        - torch.pi
    )
    bump = (
        height_perturbation
        * torch.cos(latitude)
        * torch.exp(-((delta_lon / alpha) ** 2))
        * torch.exp(-(((perturbation_lat - latitude) / beta) ** 2))
    )
    geopotential = geopotential + gravity * dynamics.to_coefficients(bump)

    # 4. Height noise below degree noise_lmax (global RNG, on CPU)
    k = min(noise_lmax, lmax)
    noise = torch.randn(batch, k, k, dtype=torch.complex128)
    scale = gravity * noise_amplitude * math.sqrt(4 * math.pi / (k * (k + 1)))
    geopotential[:, :k, :k] += scale * noise.to(device)

    state = torch.stack([geopotential, vorticity, divergence], dim=1)
    return torch.tril(state)  # keep m <= l
