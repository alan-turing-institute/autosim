r"""Initial conditions of the shallow water equations on the sphere.

One function per family. Each takes the sampled parameters of B simulations as
tensors of shape (B,) and returns the spectral state (B, 3, L, L). The default
sampling ranges of a family are in its ``*_RANGES`` dictionary; equal bounds at the
published values give the published case.

In the formulas of Williamson et al. (1992, W92) :math:`\theta` is the latitude and
:math:`\lambda` the longitude, as in the paper.
"""

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

# W92 case 5, zonal flow over an isolated mountain, with the Gaussian mountain of
# DCMIP-2008 (section 1.5) in place of the cone of W92. The flow speed and the depth
# h_0 (5960 m, the simulator's mean_depth) are the values of the journal version as
# quoted in the literature; the ORNL report prints 5400 m and no speed.
MOUNTAIN_RANGES: dict[str, tuple[float, float]] = {
    "flow_speed": (10.0, 30.0),  # u_0 [m/s], W92: 20
    "mountain_height": (1000.0, 3000.0),  # h_0 [m], W92 and DCMIP-2008: 2000
    # half-width d [rad], 1000 to 2000 km on the Earth; DCMIP-2008: 1500 km
    "mountain_width": (1.0e6 / 6.37122e6, 2.0e6 / 6.37122e6),
    "mountain_lon": (0.0, 2 * math.pi),  # lambda_c [rad], W92: -pi/2, i.e. 3 pi/2
    "mountain_lat": (-math.pi / 3, math.pi / 3),  # theta_c [rad], W92: pi/6
}


def balanced_geopotential(
    dynamics: SphereShallowWaterDynamics,
    vorticity: TensorLike,
    divergence: TensorLike,
    coriolis: TensorLike | None = None,
) -> TensorLike:
    r"""Geopotential in nonlinear balance with a wind.

    The divergence equation with a steady divergence,
    :math:`0 = \hat{\mathbf{k}}\cdot\nabla\times((\zeta+f)\,\mathbf{u})
    - \nabla^2(\Phi + \tfrac12|\mathbf{u}|^2)`, gives

    .. math::

        \Phi = \nabla^{-2}\,\hat{\mathbf{k}}\cdot\nabla\times
            \big((\zeta+f)\,\mathbf{u}\big) - \tfrac12|\mathbf{u}|^2
            + \text{constant},

    with the constant chosen so that the global mean depth is ``mean_depth``. For a
    zonal jet this is the integral of Galewsky et al. (2004), eq. (3).

    Args:
        dynamics: Fixed operators.
        vorticity: Vorticity coefficients of the wind, (B, L, L).
        divergence: Divergence coefficients of the wind, (B, L, L).
        coriolis: Coriolis parameter (B, *grid.shape); None is the polar one.

    Returns:
        Geopotential coefficients (B, L, L).
    """
    if coriolis is None:
        coriolis = dynamics.coriolis
    wind = dynamics.wind(vorticity, divergence)
    absolute_vorticity = dynamics.to_grid(vorticity) + coriolis
    flux_curl, _ = dynamics.vorticity_divergence(wind * absolute_vorticity.unsqueeze(1))
    kinetic_energy = 0.5 * (wind[:, 0] ** 2 + wind[:, 1] ** 2)
    geopotential = dynamics.inverse_laplacian * flux_curl - dynamics.to_coefficients(
        kinetic_energy
    )
    geopotential[:, 0, 0] = (
        math.sqrt(4 * math.pi) * dynamics.gravity * dynamics.mean_depth
    )
    return geopotential


def balance_state(
    dynamics: SphereShallowWaterDynamics,
    state: TensorLike,
    coriolis: TensorLike | None = None,
    surface_geopotential: TensorLike | None = None,
) -> TensorLike:
    """Balance step of analysed fields: no divergence, height from the vorticity.

    The divergence is removed and the free surface is replaced by the one in
    nonlinear balance with the remaining wind (``balanced_geopotential``), with the
    global mean of the supplied free surface. W92 (p. 19) suggests a discrete balance
    of this kind for initial states; for analysed fields (case 7) it recommends a
    normal-mode initialisation instead, which this step replaces.

    Args:
        dynamics: Fixed operators.
        state: Spectral state (B, 3, L, L), the geopotential of the fluid depth.
        coriolis: Coriolis parameter (B, *grid.shape); None is the polar one.
        surface_geopotential: Coefficients of g h_s, (B, L, L); None is a flat bottom.

    Returns:
        The balanced spectral state (B, 3, L, L).
    """
    vorticity = state[:, 1]
    divergence = torch.zeros_like(vorticity)
    free_surface = balanced_geopotential(dynamics, vorticity, divergence, coriolis)
    supplied_free_surface = state[:, 0]
    if surface_geopotential is not None:
        supplied_free_surface = supplied_free_surface + surface_geopotential
    free_surface[:, 0, 0] = supplied_free_surface[:, 0, 0]
    geopotential = free_surface
    if surface_geopotential is not None:
        geopotential = free_surface - surface_geopotential
    return torch.tril(torch.stack([geopotential, vorticity, divergence], dim=1))


def galewsky_initial_condition(
    dynamics: SphereShallowWaterDynamics,
    jet_speed: TensorLike,
    height_perturbation: TensorLike,
    perturbation_lon: TensorLike,
    perturbation_lat: TensorLike,
    noise_amplitude: float,
    noise_lmax: int,
    device: str = "cuda",
    generators: list[torch.Generator] | None = None,
) -> TensorLike:
    """Galewsky et al. (2004) jet, bump and height noise for B simulations.

    Jet (their eq. 2), balanced height with a global mean depth ``mean_depth``
    (eq. 3, computed spectrally from the steady divergence equation), bump
    h' = h_hat cos(phi) exp(-(dlon/alpha)^2) exp(-((phi_2 - phi)/beta)^2) (eq. 4) with
    alpha = 1/3, beta = 1/15, and a random height noise below degree ``noise_lmax``.
    A zero bump and a zero noise give the unperturbed jet of their section 3.

    Args:
        dynamics: Fixed operators.
        jet_speed: Maximum jet speed u_max [m/s], shape (B,).
        height_perturbation: Bump amplitude h_hat [m], shape (B,).
        perturbation_lon: Bump centre longitude [rad], shape (B,).
        perturbation_lat: Bump centre latitude phi_2 [rad], shape (B,).
        noise_amplitude: Root-mean-square of the height noise [m].
        noise_lmax: The noise has degrees and orders below this value.
        device: Torch device of the state.
        generators: One random generator per simulation, for the noise. None draws
            from the global torch generator.

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

    # 2. Balanced geopotential, eq. (3): lap^-1 curl((zeta + f) u) - |u|^2 / 2 + const.
    # The balance uses the jet as represented with degrees l < L (not the analytic
    # one), which makes the represented jet exactly steady at any resolution.
    geopotential = balanced_geopotential(dynamics, vorticity, divergence)

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

    # 4. Height noise below degree noise_lmax, one generator per simulation
    k = min(noise_lmax, lmax)
    if generators is None:
        noise = torch.randn(batch, k, k, dtype=torch.complex128)  # global generator
    else:
        noise = torch.zeros(batch, k, k, dtype=torch.complex128, device=device)
        for index in range(batch):
            generator = generators[index]
            noise[index] = torch.randn(
                k,
                k,
                dtype=torch.complex128,
                generator=generator,
                device=generator.device,
            )
    # A real field has real m = 0 coefficients. The real part is copied first: torch
    # rejects a write from a view of the same memory (seen with one simulation and
    # noise_lmax = 1).
    noise[:, :, 0] = noise[:, :, 0].real.clone()
    scale = gravity * noise_amplitude * math.sqrt(4 * math.pi / (k * (k + 1)))
    geopotential[:, :k, :k] += scale * noise.to(device)

    state = torch.stack([geopotential, vorticity, divergence], dim=1)
    return torch.tril(state)  # keep m <= l


def solid_body_zonal_flow(
    dynamics: SphereShallowWaterDynamics, flow_speed: TensorLike
) -> TensorLike:
    r"""Solid-body zonal flow about the geographic pole, W92 case 2 without tilt.

    :math:`u = u_0\cos\theta`, :math:`v = 0` and
    :math:`g h = g h_0 - (a\Omega u_0 + \tfrac12 u_0^2)\sin^2\theta` (W92 eqs. 90, 91,
    95 with :math:`\alpha = 0`), with :math:`h_0` = ``mean_depth``: the background
    flow of W92 case 5.

    Args:
        dynamics: Fixed operators.
        flow_speed: Speed u_0 at the equator [m/s], shape (B,).

    Returns:
        Spectral state (B, 3, L, L).
    """
    latitude = dynamics.latitude.unsqueeze(0)
    parameter_shape = (flow_speed.shape[0],) + (1,) * len(dynamics.grid.shape)
    u0 = flow_speed.to(latitude.device, torch.float64).reshape(parameter_shape)
    eastward = u0 * torch.cos(latitude)
    geopotential_grid = (
        dynamics.gravity * dynamics.mean_depth
        - (dynamics.radius * dynamics.omega * u0 + 0.5 * u0**2)
        * torch.sin(latitude) ** 2
    )
    wind = torch.stack([eastward, torch.zeros_like(eastward)], dim=1)
    vorticity, divergence = dynamics.vorticity_divergence(wind)
    geopotential = dynamics.to_coefficients(geopotential_grid)
    state = torch.stack([geopotential, vorticity, divergence], dim=1)
    return torch.tril(state)  # keep m <= l


def mountain_orography(
    dynamics: SphereShallowWaterDynamics,
    mountain_lon: TensorLike,
    mountain_lat: TensorLike,
    mountain_height: TensorLike,
    mountain_width: TensorLike,
) -> TensorLike:
    r"""Isolated Gaussian mountain of DCMIP-2008 (section 1.5, eqs. 81 and 82).

    :math:`h_s = h_0 \exp[-(r/d)^2]` with :math:`r` the great-circle angle to the
    summit, :math:`\cos r = \sin\theta_c\sin\theta + \cos\theta_c\cos\theta
    \cos(\lambda - \lambda_c)`, and :math:`d` the half-width. DCMIP-2008 uses
    :math:`h_0 = 2000` m and :math:`d` = 1500 km. It replaces the cone of W92 case 5
    (eq. 134), which is not differentiable at its summit and foot, so that its
    truncated series rings (J95; Galewsky et al. 2004); the Gaussian is smooth. The
    solver uses the truncated series returned here.

    Args:
        dynamics: Fixed operators.
        mountain_lon: Longitude of the summit lambda_c [rad], shape (B,).
        mountain_lat: Latitude of the summit theta_c [rad], shape (B,).
        mountain_height: Height of the summit h_0 [m], shape (B,).
        mountain_width: Half-width d as an angle on the sphere [rad], shape (B,).

    Returns:
        Coefficients of the surface geopotential g h_s, shape (B, L, L).
    """
    latitude = dynamics.latitude.unsqueeze(0)
    longitude = dynamics.longitude.unsqueeze(0)
    parameter_shape = (mountain_lon.shape[0],) + (1,) * len(dynamics.grid.shape)
    lon_c = mountain_lon.to(latitude.device, torch.float64).reshape(parameter_shape)
    lat_c = mountain_lat.to(latitude.device, torch.float64).reshape(parameter_shape)
    height = mountain_height.to(latitude.device, torch.float64).reshape(parameter_shape)
    width = mountain_width.to(latitude.device, torch.float64).reshape(parameter_shape)

    # Great-circle angle to the summit, DCMIP-2008 eq. (82) divided by a
    cosine = torch.sin(lat_c) * torch.sin(latitude) + torch.cos(lat_c) * torch.cos(
        latitude
    ) * torch.cos(longitude - lon_c)
    distance = torch.acos(cosine.clamp(-1.0, 1.0))
    surface_height = height * torch.exp(-((distance / width) ** 2))

    surface_geopotential = dynamics.gravity * dynamics.to_coefficients(surface_height)
    return torch.tril(surface_geopotential)  # keep m <= l


def mountain_initial_condition(
    dynamics: SphereShallowWaterDynamics,
    flow_speed: TensorLike,
    mountain_lon: TensorLike,
    mountain_lat: TensorLike,
    mountain_height: TensorLike,
    mountain_width: TensorLike,
) -> tuple[TensorLike, TensorLike]:
    """Zonal flow over an isolated mountain, W92 case 5 with a Gaussian mountain.

    The wind and the free surface are those of case 2 without tilt
    (``solid_body_zonal_flow``), with h_0 = ``mean_depth``. The mountain
    (``mountain_orography``) is placed under this balanced flow, so the fluid depth of
    the state is the free surface minus the mountain, and the flow adjusts by
    radiating gravity waves.

    Args:
        dynamics: Fixed operators.
        flow_speed: Speed u_0 of the zonal flow at the equator [m/s], shape (B,).
        mountain_lon: Longitude of the summit [rad], shape (B,).
        mountain_lat: Latitude of the summit [rad], shape (B,).
        mountain_height: Height of the summit [m], shape (B,).
        mountain_width: Half-width of the mountain [rad], shape (B,).

    Returns:
        The spectral state (B, 3, L, L) and the coefficients of the surface
        geopotential g h_s, shape (B, L, L).
    """
    state = solid_body_zonal_flow(dynamics, flow_speed)
    surface_geopotential = mountain_orography(
        dynamics, mountain_lon, mountain_lat, mountain_height, mountain_width
    )
    state[:, 0] = state[:, 0] - surface_geopotential  # Phi = g * fluid depth
    return state, surface_geopotential


def zonal_jet(
    dynamics: SphereShallowWaterDynamics,
    jet_speed: TensorLike,
    jet_latitude: TensorLike,
) -> tuple[TensorLike, TensorLike]:
    r"""Zonal jet with the profile of Galewsky et al. (2004), at any latitude.

    Their eq. (2), :math:`u = (u_\text{max}/e_n)\exp[1/((\varphi-\varphi_0)
    (\varphi-\varphi_1))]` between :math:`\varphi_0` and :math:`\varphi_1` and zero
    elsewhere, with :math:`e_n = \exp[-4/(\varphi_1-\varphi_0)^2]`. The jet is
    centred at ``jet_latitude`` and keeps their half width
    :math:`(\varphi_1-\varphi_0)/2 = 3\pi/28`; their jet is at :math:`\pi/4`.

    Args:
        dynamics: Fixed operators.
        jet_speed: Maximum speed u_max [m/s], shape (B,).
        jet_latitude: Latitude of the maximum [rad], shape (B,).

    Returns:
        Vorticity and divergence coefficients of the jet, each (B, L, L).
    """
    latitude = dynamics.latitude.unsqueeze(0)
    parameter_shape = (jet_speed.shape[0],) + (1,) * len(dynamics.grid.shape)
    u_max = jet_speed.to(latitude.device, torch.float64).reshape(parameter_shape)
    centre = jet_latitude.to(latitude.device, torch.float64).reshape(parameter_shape)
    half_width = 3 * torch.pi / 28
    phi0 = centre - half_width
    phi1 = centre + half_width
    en = math.exp(-4 / (2 * half_width) ** 2)

    inside = (latitude > phi0) & (latitude < phi1)
    product = torch.where(inside, (latitude - phi0) * (latitude - phi1), -1.0)
    eastward = torch.where(inside, u_max / en * torch.exp(1.0 / product), 0.0)
    wind = torch.stack([eastward, torch.zeros_like(eastward)], dim=1)
    return dynamics.vorticity_divergence(wind)
