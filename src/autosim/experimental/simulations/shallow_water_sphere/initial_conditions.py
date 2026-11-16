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

from .diagnostics import degree_power
from .dynamics import SphereShallowWaterDynamics
from .forcing import complex_noise

GALEWSKY_RANGES: dict[str, tuple[float, float]] = {
    "jet_speed": (60.0, 100.0),  # u_max [m/s], paper: 80
    "height_perturbation_m": (60.0, 180.0),  # bump amplitude [m], paper: 120
    "perturbation_lon": (0.0, 2 * math.pi),  # bump longitude [rad], paper: 0
    "perturbation_lat": (math.pi / 5, math.pi / 3),  # bump latitude [rad], paper: pi/4
}

# W92 case 2, steady zonal flow about an axis tilted by alpha.
WILLIAMSON2_RANGES: dict[str, tuple[float, float]] = {
    "flow_speed": (20.0, 40.0),  # u_0 [m/s], W92: 2 pi a / 12 days = 38.61
    "tilt": (0.0, math.pi / 2),  # alpha [rad], W92: 0, 0.05, pi/2 - 0.05, pi/2
}

# W92 case 6, Rossby-Haurwitz wave.
ROSSBY_HAURWITZ_RANGES: dict[str, tuple[float, float]] = {
    "wavenumber": (3.0, 6.0),  # R, rounded to the nearest integer, W92: 4
    "wave_amplitude": (5.0e-6, 1.0e-5),  # K = omega [1/s], W92: 7.848e-6
    "mean_depth": (7000.0, 9000.0),  # h_0 [m], W92: 8000
}

# W92 case 5, zonal flow over an isolated mountain. The flow speed and the depth h_0
# (5960 m, the simulator's mean_depth) are the values of the journal version as quoted
# in the literature; the ORNL report prints 5400 m and no speed.
MOUNTAIN_RANGES: dict[str, tuple[float, float]] = {
    "flow_speed": (10.0, 30.0),  # u_0 [m/s], W92: 20
    "mountain_height": (1000.0, 3000.0),  # h_s0 [m], W92: 2000
    "mountain_radius": (math.pi / 12, math.pi / 6),  # R [rad], W92: pi/9
    "mountain_lon": (0.0, 2 * math.pi),  # lambda_c [rad], W92: -pi/2, i.e. 3 pi/2
    "mountain_lat": (-math.pi / 3, math.pi / 3),  # theta_c [rad], W92: pi/6
}


# Random balanced flow, the names of the planar "balanced_random_pv" family.
RANDOM_BALANCED_RANGES: dict[str, tuple[float, float]] = {
    "amp": (10.0, 30.0),  # root-mean-square wind speed [m/s]
    "initial_wavenumber": (4.0, 10.0),  # central degree of the streamfunction
    "initial_bandwidth": (1.0, 3.0),  # width of its spectrum, in degrees
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
    noise[:, :, 0] = noise[:, :, 0].real  # a real field has real m = 0 coefficients
    scale = gravity * noise_amplitude * math.sqrt(4 * math.pi / (k * (k + 1)))
    geopotential[:, :k, :k] += scale * noise.to(device)

    state = torch.stack([geopotential, vorticity, divergence], dim=1)
    return torch.tril(state)  # keep m <= l


def williamson2_initial_condition(
    dynamics: SphereShallowWaterDynamics, flow_speed: TensorLike, tilt: TensorLike
) -> TensorLike:
    r"""Steady zonal flow about a tilted axis, W92 case 2.

    The flow is a solid-body rotation about an axis tilted by :math:`\alpha` from the
    geographic pole, and the Coriolis parameter uses the same axis (W92 eq. 96,
    ``coriolis_parameter`` with the pole at longitude :math:`\pi` and latitude
    :math:`\pi/2 - \alpha`), so the state is steady for any tilt. With
    :math:`s = -\cos\lambda\cos\theta\sin\alpha + \sin\theta\cos\alpha`, the sine of
    the latitude measured from the tilted pole (W92 eqs. 90, 91, 95):

    .. math::

        u = u_0(\cos\theta\cos\alpha + \cos\lambda\sin\theta\sin\alpha), \qquad
        v = -u_0\sin\lambda\sin\alpha, \qquad
        g h = g h_0 - \big(a\Omega u_0 + \tfrac12 u_0^2\big)\,s^2.

    :math:`h_0` is ``mean_depth``. W92 uses :math:`u_0 = 2\pi a / 12` days,
    :math:`g h_0 = 2.94\times10^4` m^2/s^2 and
    :math:`\alpha = 0,\ 0.05,\ \pi/2 - 0.05,\ \pi/2`.

    Args:
        dynamics: Fixed operators.
        flow_speed: Speed u_0 at the equator of the flow [m/s], shape (B,).
        tilt: Angle alpha between the flow axis and the geographic pole [rad],
            shape (B,).

    Returns:
        Spectral state (B, 3, L, L).
    """
    latitude = dynamics.latitude.unsqueeze(0)
    longitude = dynamics.longitude.unsqueeze(0)
    parameter_shape = (flow_speed.shape[0],) + (1,) * len(dynamics.grid.shape)
    u0 = flow_speed.to(latitude.device, torch.float64).reshape(parameter_shape)
    alpha = tilt.to(latitude.device, torch.float64).reshape(parameter_shape)

    tilted_sine = -torch.cos(longitude) * torch.cos(latitude) * torch.sin(
        alpha
    ) + torch.sin(latitude) * torch.cos(alpha)
    eastward = u0 * (
        torch.cos(latitude) * torch.cos(alpha)
        + torch.cos(longitude) * torch.sin(latitude) * torch.sin(alpha)
    )
    northward = -u0 * torch.sin(longitude) * torch.sin(alpha)
    northward = northward.expand_as(eastward)
    geopotential_grid = (
        dynamics.gravity * dynamics.mean_depth
        - (dynamics.radius * dynamics.omega * u0 + 0.5 * u0**2) * tilted_sine**2
    )

    wind = torch.stack([eastward, northward], dim=1)
    vorticity, divergence = dynamics.vorticity_divergence(wind)
    geopotential = dynamics.to_coefficients(geopotential_grid)
    state = torch.stack([geopotential, vorticity, divergence], dim=1)
    return torch.tril(state)  # keep m <= l


def rossby_haurwitz_initial_condition(
    dynamics: SphereShallowWaterDynamics,
    wavenumber: TensorLike,
    wave_amplitude: TensorLike,
    mean_depth: TensorLike,
) -> TensorLike:
    r"""Rossby-Haurwitz wave, W92 case 6.

    Wind of the stream function
    :math:`\psi = -a^2\omega\sin\theta + a^2 K\cos^R\theta\sin\theta\cos R\lambda`
    (W92 eqs. 137, 138) and the height that makes the initial divergence tendency
    zero (eqs. 140-143):

    .. math::

        u &= a\omega\cos\theta
            + aK\cos^{R-1}\theta\,(R\sin^2\theta - \cos^2\theta)\cos R\lambda, \\
        v &= -aKR\cos^{R-1}\theta\,\sin\theta\,\sin R\lambda, \\
        g h &= g h_0 + a^2 A(\theta) + a^2 B(\theta)\cos R\lambda
            + a^2 C(\theta)\cos 2R\lambda,

    .. math::

        A &= \tfrac{\omega}{2}(2\Omega+\omega)\cos^2\theta + \tfrac14 K^2\cos^{2R}\theta
            \big[(R+1)\cos^2\theta + (2R^2-R-2) - 2R^2\cos^{-2}\theta\big], \\
        B &= \frac{2(\Omega+\omega)K}{(R+1)(R+2)}\cos^R\theta
            \big[(R^2+2R+2) - (R+1)^2\cos^2\theta\big], \\
        C &= \tfrac14 K^2\cos^{2R}\theta\big[(R+1)\cos^2\theta - (R+2)\big].

    W92 uses :math:`\omega = K = 7.848\times10^{-6}` 1/s, :math:`h_0 = 8000` m and
    :math:`R = 4`. Here :math:`\omega = K` is one parameter.

    Args:
        dynamics: Fixed operators.
        wavenumber: Zonal wavenumber R, integer values, shape (B,).
        wave_amplitude: omega = K [1/s], shape (B,).
        mean_depth: Depth h_0 of eq. (140) [m], shape (B,).

    Returns:
        Spectral state (B, 3, L, L).
    """
    latitude = dynamics.latitude.unsqueeze(0)
    longitude = dynamics.longitude.unsqueeze(0)
    parameter_shape = (wavenumber.shape[0],) + (1,) * len(dynamics.grid.shape)
    R = wavenumber.to(latitude.device, torch.float64).reshape(parameter_shape)
    K = wave_amplitude.to(latitude.device, torch.float64).reshape(parameter_shape)
    h0 = mean_depth.to(latitude.device, torch.float64).reshape(parameter_shape)
    omega_wave = K  # W92 takes omega = K
    a = dynamics.radius
    Omega = dynamics.omega
    cos_lat = torch.cos(latitude)
    sin_lat = torch.sin(latitude)

    # Wind, eqs. (137) and (138)
    eastward = a * omega_wave * cos_lat + a * K * cos_lat ** (R - 1) * (
        R * sin_lat**2 - cos_lat**2
    ) * torch.cos(R * longitude)
    northward = -a * K * R * cos_lat ** (R - 1) * sin_lat * torch.sin(R * longitude)

    # Height, eqs. (140) to (143). cos^(2R) * cos^(-2) is written cos^(2R - 2).
    A = 0.5 * omega_wave * (2 * Omega + omega_wave) * cos_lat**2 + 0.25 * K**2 * (
        (R + 1) * cos_lat ** (2 * R + 2)
        + (2 * R**2 - R - 2) * cos_lat ** (2 * R)
        - 2 * R**2 * cos_lat ** (2 * R - 2)
    )
    B = (
        2
        * (Omega + omega_wave)
        * K
        / ((R + 1) * (R + 2))
        * cos_lat**R
        * ((R**2 + 2 * R + 2) - (R + 1) ** 2 * cos_lat**2)
    )
    C = 0.25 * K**2 * cos_lat ** (2 * R) * ((R + 1) * cos_lat**2 - (R + 2))
    geopotential_grid = (
        dynamics.gravity * h0
        + a**2 * A
        + a**2 * B * torch.cos(R * longitude)
        + a**2 * C * torch.cos(2 * R * longitude)
    )

    wind = torch.stack([eastward, northward], dim=1)
    vorticity, divergence = dynamics.vorticity_divergence(wind)
    geopotential = dynamics.to_coefficients(geopotential_grid)
    state = torch.stack([geopotential, vorticity, divergence], dim=1)
    return torch.tril(state)  # keep m <= l


def mountain_orography(
    dynamics: SphereShallowWaterDynamics,
    mountain_lon: TensorLike,
    mountain_lat: TensorLike,
    mountain_height: TensorLike,
    mountain_radius: TensorLike,
) -> TensorLike:
    r"""Isolated conical mountain of W92 case 5 (eq. 134).

    :math:`h_s = h_{s0}\,(1 - r/R)` with
    :math:`r^2 = \min[R^2, (\lambda-\lambda_c)^2 + (\theta-\theta_c)^2]`: a cone in
    longitude and latitude. W92 uses :math:`h_{s0} = 2000` m, :math:`R = \pi/9`,
    :math:`\lambda_c = -\pi/2` and :math:`\theta_c = \pi/6`. The cone has a corner at
    its summit and at its foot, so its truncated series rings (Galewsky et al. 2004).
    The solver uses the truncated series returned here.

    Args:
        dynamics: Fixed operators.
        mountain_lon: Longitude of the summit lambda_c [rad], shape (B,).
        mountain_lat: Latitude of the summit theta_c [rad], shape (B,).
        mountain_height: Height of the summit h_s0 [m], shape (B,).
        mountain_radius: Radius R of the base [rad], shape (B,).

    Returns:
        Coefficients of the surface geopotential g h_s, shape (B, L, L).
    """
    latitude = dynamics.latitude.unsqueeze(0)
    longitude = dynamics.longitude.unsqueeze(0)
    parameter_shape = (mountain_lon.shape[0],) + (1,) * len(dynamics.grid.shape)
    lon_c = mountain_lon.to(latitude.device, torch.float64).reshape(parameter_shape)
    lat_c = mountain_lat.to(latitude.device, torch.float64).reshape(parameter_shape)
    height = mountain_height.to(latitude.device, torch.float64).reshape(parameter_shape)
    radius = mountain_radius.to(latitude.device, torch.float64).reshape(parameter_shape)

    # Longitude difference in [-pi, pi)
    delta_lon = torch.remainder(longitude - lon_c + torch.pi, 2 * torch.pi) - torch.pi
    distance = torch.sqrt(delta_lon**2 + (latitude - lat_c) ** 2)
    distance = torch.minimum(distance, radius)
    surface_height = height * (1 - distance / radius)

    surface_geopotential = dynamics.gravity * dynamics.to_coefficients(surface_height)
    return torch.tril(surface_geopotential)  # keep m <= l


def mountain_initial_condition(
    dynamics: SphereShallowWaterDynamics,
    flow_speed: TensorLike,
    mountain_lon: TensorLike,
    mountain_lat: TensorLike,
    mountain_height: TensorLike,
    mountain_radius: TensorLike,
) -> tuple[TensorLike, TensorLike]:
    """Zonal flow over an isolated mountain, W92 case 5.

    The wind and the free surface are those of case 2 without tilt, with h_0 =
    ``mean_depth``. The mountain is placed under this balanced flow, so the fluid depth
    of the state is the free surface minus the mountain, and the flow adjusts by
    radiating gravity waves.

    Args:
        dynamics: Fixed operators.
        flow_speed: Speed u_0 of the zonal flow at the equator [m/s], shape (B,).
        mountain_lon: Longitude of the summit [rad], shape (B,).
        mountain_lat: Latitude of the summit [rad], shape (B,).
        mountain_height: Height of the summit [m], shape (B,).
        mountain_radius: Radius of the base [rad], shape (B,).

    Returns:
        The spectral state (B, 3, L, L) and the coefficients of the surface
        geopotential g h_s, shape (B, L, L).
    """
    no_tilt = torch.zeros_like(flow_speed)
    state = williamson2_initial_condition(dynamics, flow_speed, no_tilt)
    surface_geopotential = mountain_orography(
        dynamics, mountain_lon, mountain_lat, mountain_height, mountain_radius
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


def random_balanced_initial_condition(
    dynamics: SphereShallowWaterDynamics,
    amp: TensorLike,
    initial_wavenumber: TensorLike,
    initial_bandwidth: TensorLike,
    generators: list[torch.Generator],
    coriolis: TensorLike | None = None,
) -> TensorLike:
    r"""Random flow without divergence, with the height in nonlinear balance.

    The streamfunction has random coefficients with a Gaussian spectrum,
    :math:`\hat\psi_l^m = w_l\,\epsilon_l^m` with
    :math:`w_l = \exp[-(l - l_0)^2 / (2\sigma^2)]`, scaled so that the root-mean-square
    wind speed is ``amp``. The height follows from ``balanced_geopotential``, so the
    flow starts without gravity waves. This is the spherical counterpart of the
    planar ``balanced_random_pv`` family.

    Args:
        dynamics: Fixed operators.
        amp: Root-mean-square wind speed [m/s], shape (B,).
        initial_wavenumber: Central degree l_0 of the streamfunction, shape (B,).
        initial_bandwidth: Width sigma of its spectrum, in degrees, shape (B,).
        generators: One random generator per simulation.
        coriolis: Coriolis parameter (B, *grid.shape); None is the polar one.

    Returns:
        Spectral state (B, 3, L, L).
    """
    device = dynamics.laplacian.device
    lmax = dynamics.lmax
    batch = amp.shape[0]
    speed = amp.to(device, torch.float64).reshape(batch, 1, 1)
    centre = initial_wavenumber.to(device, torch.float64).reshape(batch, 1, 1)
    width = initial_bandwidth.to(device, torch.float64).reshape(batch, 1, 1)

    degree = torch.arange(lmax, dtype=torch.float64, device=device).reshape(1, lmax, 1)
    weight = torch.exp(-0.5 * ((degree - centre) / width) ** 2)
    weight = weight * (degree >= 1)  # no global mean
    noise = complex_noise(generators, 1, lmax)[:, 0]
    vorticity = dynamics.laplacian * (weight * noise)

    # Global mean of |u|^2: sum over degrees of a^2 / (4 pi l (l + 1)) * power
    factor = torch.zeros(lmax, dtype=torch.float64, device=device)
    factor[1:] = -dynamics.inverse_laplacian[1:, 0] / (4 * math.pi)
    mean_square_speed = (factor * degree_power(vorticity)).sum(dim=-1)
    vorticity = vorticity * speed / torch.sqrt(mean_square_speed).reshape(batch, 1, 1)

    divergence = torch.zeros_like(vorticity)
    geopotential = balanced_geopotential(dynamics, vorticity, divergence, coriolis)
    state = torch.stack([geopotential, vorticity, divergence], dim=1)
    return torch.tril(state)  # keep m <= l
