r"""Forcing of the shallow water equations on the sphere.

Three forcings, which can be combined:

- kinetic-energy forcing and backscatter (``forcing_type``), below;
- stochastically perturbed physics tendencies (SPPT, ``SpptSettings``): the increment
  of the drag and the relaxation multiplied by a random pattern;
- The Well's daily and yearly height forcing (``PeriodicForcingSettings``).

Kinetic-energy backscatter forcing follows Berner, Shutts, Leutbecher and Palmer
(2009, J. Atmos. Sci. 66, 603-626, "B09" below). A streamfunction pattern
:math:`\psi'` (and, for a divergent part, a velocity-potential pattern
:math:`\chi'`) is expanded in spherical harmonics, and each coefficient follows the
first-order autoregressive process of B09 eq. (2),

.. math::

    \psi'^m_l(t + \Delta t) = (1 - \alpha)\,\psi'^m_l(t)
        + g_l \sqrt{\alpha}\,\epsilon^m_l(t), \qquad g_l = b\,l^p,

with :math:`\alpha = \Delta t / \tau` for a decorrelation time :math:`\tau`
(:math:`\alpha = 1` is white noise) and :math:`\epsilon` a complex Gaussian noise of
unit variance. The pattern is an increment per step: the vorticity gains
:math:`\nabla^2\psi'` and the divergence :math:`\nabla^2\chi'` after each solver step.

**Energy norm.** The amplitude :math:`b` is set from the rate at which a free random
walk of the increments gains kinetic energy per unit mass, the global mean of
:math:`\tfrac12|\mathbf{u}|^2`. With the orthonormal harmonics of torch-harmonics
(see ``diagnostics``), a pattern has the kinetic energy
:math:`\sum_l l(l+1)(2l+1)\,\langle|\psi'_l|^2\rangle / (8\pi a^2)`. Its stationary
variance is :math:`g_l^2 / (2 - \alpha)` (B09 eq. 3), and the memory of the process
multiplies the energy gained per step by :math:`2/\alpha - 1` (B09 eqs. 7 and A11),
so the energy gained per step is

.. math::

    \varepsilon\,\Delta t = \frac{b^2\,\Gamma}{8\pi a^2 \alpha}, \qquad
    \Gamma = \sum_{l} l(l+1)(2l+1)\,l^{2p},

summed over the forced degrees. This is B09 eq. (8) in the present convention: their
:math:`4\pi` and :math:`\sigma_z` become :math:`8\pi` and 1 because the energy carries
its factor one half and the noise has unit variance. The kinetic-energy spectrum of
the forcing follows :math:`l^{2p+3}` (B09 eq. 10). ``forcing_energy_rate`` is
:math:`\varepsilon` [m^2/s^3].

The patterns are stored for a unit energy rate. The rate that is applied multiplies
them by its square root, either one value per simulation or, for the local
backscatter of B09 eq. (13), a field on the grid.

The functions take one random generator per simulation, so that the draws of a
simulation do not depend on the other simulations of its batch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from autosim.types import TensorLike

from .diagnostics import degree_power
from .dynamics import SphereShallowWaterDynamics

# Values of forcing_type: no forcing, a streamfunction pattern, or a streamfunction
# and a velocity-potential pattern
FORCING_TYPES = ("none", "vortical", "momentum")


@dataclass
class ForcingSettings:
    """Settings of the stochastic forcing of one batch of B simulations.

    Attributes:
        forcing_type: "none", "vortical" or "momentum".
        energy_rate: Base energy rate of the kinetic-energy forcing [m^2/s^3], (B,).
        correlation_time: Decorrelation time of the pattern [s], (B,); 0 is white.
        spectral_slope: Exponent p of the noise amplitude g_l = b l^p, (B,).
        degree_min: Lowest forced degree.
        degree_max: Highest forced degree.
        divergent_fraction: Fraction of the energy rate given to the divergent
            pattern, (B,); 0 for "vortical".
        backscatter_fraction: Fraction of the diagnosed dissipation added to the
            energy rate, (B,).
        backscatter_mode: "global" or "local".
        backscatter_include_drag: Whether the drag loss counts as dissipation.
    """

    forcing_type: str
    energy_rate: TensorLike
    correlation_time: TensorLike
    spectral_slope: TensorLike
    degree_min: int
    degree_max: int
    divergent_fraction: TensorLike
    backscatter_fraction: TensorLike
    backscatter_mode: str
    backscatter_include_drag: bool


def no_forcing(batch: int) -> ForcingSettings:
    """Settings of B simulations without stochastic forcing."""
    zeros = torch.zeros(batch, dtype=torch.float64)
    return ForcingSettings(
        forcing_type="none",
        energy_rate=zeros,
        correlation_time=zeros,
        spectral_slope=zeros,
        degree_min=1,
        degree_max=1,
        divergent_fraction=zeros,
        backscatter_fraction=zeros,
        backscatter_mode="global",
        backscatter_include_drag=False,
    )


def autoregressive_parameter(correlation_time: TensorLike, dt: float) -> TensorLike:
    r"""Parameter :math:`\alpha = \Delta t / \tau` of the AR(1) process.

    A zero decorrelation time gives :math:`\alpha = 1`, noise without memory.

    Args:
        correlation_time: Decorrelation time tau [s], shape (B,).
        dt: Solver time step [s].

    Returns:
        alpha in (0, 1], shape (B,).
    """
    correlation_time = correlation_time.to(torch.float64)
    if ((correlation_time > 0) & (correlation_time < dt)).any():
        msg = "forcing_correlation_time must be 0 or at least one solver step"
        raise ValueError(msg)
    alpha = torch.ones_like(correlation_time)
    has_memory = correlation_time > 0
    alpha[has_memory] = dt / correlation_time[has_memory]
    return alpha


def unit_rate_amplitudes(
    dynamics: SphereShallowWaterDynamics,
    alpha: TensorLike,
    spectral_slope: TensorLike,
    degree_min: int,
    degree_max: int,
) -> TensorLike:
    r"""Noise amplitudes :math:`g_l = b\,l^p` for an energy rate of 1 m^2/s^3.

    :math:`b^2 = 8\pi a^2 \alpha\,\Delta t / \Gamma` with
    :math:`\Gamma = \sum_l l(l+1)(2l+1)\,l^{2p}` over the forced degrees.

    Args:
        dynamics: Fixed operators and time step.
        alpha: Parameter of the AR(1) process, shape (B,).
        spectral_slope: Exponent p, shape (B,).
        degree_min: Lowest forced degree (at least 1).
        degree_max: Highest forced degree (at most L - 1).

    Returns:
        Amplitudes (B, L), zero outside the forced degrees.
    """
    device = dynamics.laplacian.device
    lmax = dynamics.lmax
    alpha = alpha.to(device, torch.float64).reshape(-1, 1)
    slope = spectral_slope.to(device, torch.float64).reshape(-1, 1)
    degree = torch.arange(lmax, dtype=torch.float64, device=device).reshape(1, lmax)
    forced = (degree >= degree_min) & (degree <= degree_max) & (degree >= 1)

    power_law = torch.zeros(slope.shape[0], lmax, dtype=torch.float64, device=device)
    forced_rows = forced.expand_as(power_law)
    degree_rows = degree.expand_as(power_law)
    slope_rows = slope.expand_as(power_law)
    power_law[forced_rows] = degree_rows[forced_rows] ** slope_rows[forced_rows]

    gamma = (degree * (degree + 1) * (2 * degree + 1) * power_law**2).sum(
        dim=1, keepdim=True
    )
    b = torch.sqrt(8 * math.pi * dynamics.radius**2 * alpha * dynamics.dt / gamma)
    return b * power_law


def complex_noise(
    generators: list[torch.Generator], n_patterns: int, lmax: int
) -> TensorLike:
    r"""Unit-variance complex Gaussian coefficients of real fields.

    A coefficient of order :math:`m > 0` is :math:`(x + i y)/\sqrt2` with x and y
    standard normal. A coefficient of order 0 is real, x alone. The sum of
    :math:`\langle|\epsilon|^2\rangle` over the orders :math:`-l..l` is then
    :math:`2l + 1`.

    Args:
        generators: One random generator per simulation.
        n_patterns: Number of independent fields per simulation.
        lmax: Band limit L.

    Returns:
        Coefficients (B, n_patterns, L, L), zero for m > l.
    """
    batch = len(generators)
    device = generators[0].device
    noise = torch.zeros(
        batch, n_patterns, lmax, lmax, dtype=torch.complex128, device=device
    )
    for index in range(batch):
        normal = torch.randn(
            n_patterns,
            lmax,
            lmax,
            2,
            generator=generators[index],
            device=device,
            dtype=torch.float64,
        )
        noise[index] = torch.complex(normal[..., 0], normal[..., 1]) / math.sqrt(2)
        noise[index, :, :, 0] = normal[:, :, 0, 0]  # m = 0: real, unit variance
    return torch.tril(noise)


def stationary_pattern(
    noise: TensorLike, amplitudes: TensorLike, alpha: TensorLike
) -> TensorLike:
    r"""Draw of the pattern from its stationary distribution.

    The stationary variance of the AR(1) process is :math:`g_l^2 / (2 - \alpha)`.

    Args:
        noise: Unit-variance coefficients (B, P, L, L).
        amplitudes: Noise amplitudes g_l, shape (B, P, L).
        alpha: Parameter of the AR(1) process, shape (B,) or, one per pattern, (B, P).

    Returns:
        Pattern coefficients (B, P, L, L).
    """
    alpha = alpha.to(noise.device, torch.float64)
    alpha = alpha.reshape(alpha.shape[0], -1, 1, 1)
    return amplitudes.unsqueeze(-1) * noise / torch.sqrt(2 - alpha)


def advance_pattern(
    pattern: TensorLike, noise: TensorLike, amplitudes: TensorLike, alpha: TensorLike
) -> TensorLike:
    r"""One step of the AR(1) process, B09 eq. (2).

    :math:`\psi'(t + \Delta t) = (1 - \alpha)\,\psi'(t)
    + g_l\sqrt{\alpha}\,\epsilon(t)`.

    Args:
        pattern: Pattern coefficients (B, P, L, L).
        noise: Unit-variance coefficients (B, P, L, L).
        amplitudes: Noise amplitudes g_l, shape (B, P, L).
        alpha: Parameter of the AR(1) process, shape (B,) or, one per pattern, (B, P).

    Returns:
        The pattern one step later, (B, P, L, L).
    """
    alpha = alpha.to(noise.device, torch.float64)
    alpha = alpha.reshape(alpha.shape[0], -1, 1, 1)
    return (1 - alpha) * pattern + amplitudes.unsqueeze(-1) * torch.sqrt(alpha) * noise


def dissipation_taper(
    lmax: int,
    device: torch.device | str,
    keep_degree: int = 10,
    zero_degree: int = 30,
) -> TensorLike:
    """Spectral filter of the dissipation rate, B09 section 2b.

    Degrees up to ``keep_degree`` are kept, degrees from ``zero_degree`` are removed.
    B09 does not give the shape of the transition; a half cosine is used.

    Args:
        lmax: Band limit L.
        device: Torch device.
        keep_degree: Highest degree kept unchanged.
        zero_degree: Lowest degree removed.

    Returns:
        Weights (L, 1), one per degree.
    """
    degree = torch.arange(lmax, dtype=torch.float64, device=device)
    position = (degree - keep_degree) / (zero_degree - keep_degree)
    position = position.clamp(min=0.0, max=1.0)
    taper = 0.5 * (1 + torch.cos(math.pi * position))
    return taper.reshape(lmax, 1)


def global_dissipation_rate(
    dynamics: SphereShallowWaterDynamics,
    undamped_state: TensorLike,
    damping: TensorLike,
) -> TensorLike:
    r"""Kinetic energy per unit mass lost to a damping in one step, per unit time.

    The damping multiplies the vorticity and divergence of degree l by
    :math:`E_l`, so the global mean of :math:`\tfrac12|\mathbf{u}|^2` decreases by

    .. math::

        \sum_l \frac{a^2}{8\pi\,l(l+1)} (1 - E_l^2)
            \sum_{m=-l}^{l} \big(|\hat\zeta_l^m|^2 + |\hat\delta_l^m|^2\big).

    Args:
        dynamics: Fixed operators and time step.
        undamped_state: Spectral state before the damping, (B, 3, L, L).
        damping: Factor E_l of each degree, shape (L,) or (B, L).

    Returns:
        Dissipation rate [m^2/s^3], shape (B,).
    """
    degree = torch.arange(
        dynamics.lmax, dtype=torch.float64, device=undamped_state.device
    )
    factor = torch.zeros_like(degree)
    factor[1:] = dynamics.radius**2 / (8 * math.pi * degree[1:] * (degree[1:] + 1))
    power = degree_power(undamped_state[:, 1]) + degree_power(undamped_state[:, 2])
    loss = (factor * (1 - damping**2) * power).sum(dim=-1)
    return loss / dynamics.dt


def local_dissipation_rate(
    dynamics: SphereShallowWaterDynamics,
    undamped_state: TensorLike,
    damping: TensorLike,
    taper: TensorLike,
) -> TensorLike:
    r"""Smoothed field of the kinetic energy lost to a damping, per unit time.

    :math:`D = (|\mathbf{u}_\text{before}|^2 - |\mathbf{u}_\text{after}|^2) /
    (2\,\Delta t)` across the damping. Its global mean is ``global_dissipation_rate``,
    and to first order in the damping it is the :math:`-\mathbf{u}\cdot\mathbf{u}'` of
    B09 eq. (11). The field is smoothed with ``dissipation_taper`` and its negative
    values are set to zero.

    Args:
        dynamics: Fixed operators and time step.
        undamped_state: Spectral state before the damping, (B, 3, L, L).
        damping: Factor E_l applied to vorticity and divergence, shape (L, 1) or
            (B, L, 1).
        taper: Weights of ``dissipation_taper``, shape (L, 1).

    Returns:
        Dissipation rate on the grid [m^2/s^3], shape (B, *grid.shape).
    """
    vorticity = undamped_state[:, 1]
    divergence = undamped_state[:, 2]
    wind_before = dynamics.wind(vorticity, divergence)
    wind_after = dynamics.wind(damping * vorticity, damping * divergence)
    speed_squared_before = wind_before[:, 0] ** 2 + wind_before[:, 1] ** 2
    speed_squared_after = wind_after[:, 0] ** 2 + wind_after[:, 1] ** 2
    dissipation = (speed_squared_before - speed_squared_after) / (2 * dynamics.dt)
    smoothed = dynamics.to_grid(taper * dynamics.to_coefficients(dissipation))
    return smoothed.clamp(min=0.0)


def kinetic_forcing_increment(
    dynamics: SphereShallowWaterDynamics,
    pattern: TensorLike,
    energy_rate: TensorLike,
) -> TensorLike:
    r"""Increment of the state from the patterns, for one step.

    The patterns hold :math:`\psi'` (index 0) and, when present, :math:`\chi'`
    (index 1) for a unit energy rate. They are multiplied by the square root of the
    energy rate, then the vorticity gains :math:`\nabla^2\psi'` and the divergence
    :math:`\nabla^2\chi'`. A rate of shape (B,) multiplies the coefficients. A rate
    on the grid, (B, \*grid.shape), multiplies the patterns in grid space, as in B09
    eq. (13).

    Args:
        dynamics: Fixed operators.
        pattern: Pattern coefficients (B, P, L, L), P = 1 or 2.
        energy_rate: Energy rate [m^2/s^3], shape (B,) or (B, *grid.shape).

    Returns:
        Increment of the spectral state, (B, 3, L, L); the geopotential part is zero.
    """
    batch = pattern.shape[0]
    energy_rate = energy_rate.to(pattern.device, torch.float64)
    if energy_rate.ndim == 1:
        scaled = torch.sqrt(energy_rate).reshape(batch, 1, 1, 1) * pattern
    else:
        pattern_grid = dynamics.to_grid(pattern)  # (B, P, *grid.shape)
        scaled_grid = torch.sqrt(energy_rate).unsqueeze(1) * pattern_grid
        scaled = dynamics.to_coefficients(scaled_grid)

    increment = torch.zeros(
        batch,
        3,
        dynamics.lmax,
        dynamics.lmax,
        dtype=pattern.dtype,
        device=pattern.device,
    )
    increment[:, 1] = dynamics.laplacian * scaled[:, 0]
    if pattern.shape[1] == 2:
        increment[:, 2] = dynamics.laplacian * scaled[:, 1]
    return increment


# ---------------------------------------------------------------------------
# Stochastically perturbed physics tendencies (SPPT)
# ---------------------------------------------------------------------------


@dataclass
class SpptSettings:
    """Settings of the stochastically perturbed physics tendencies.

    The pattern r is a sum of independent AR(1) Gaussian fields, one per scale k, and
    is clipped to [-clip, clip] on the grid (Leutbecher et al. 2017, section 2.1).

    Attributes:
        amplitudes: Grid-point standard deviation sigma_k of each field, (K,).
        length_scales: Correlation length L_k of each field [m], (K,).
        correlation_times: Decorrelation time tau_k of each field [s], (K,); 0 is
            white noise.
        clip: Bound of the summed pattern.
    """

    amplitudes: tuple[float, ...]
    length_scales: tuple[float, ...]
    correlation_times: tuple[float, ...]
    clip: float = 1.0


def sppt_noise_amplitudes(
    dynamics: SphereShallowWaterDynamics, sppt: SpptSettings, alpha: TensorLike
) -> TensorLike:
    r"""Noise amplitudes :math:`g_l` of the SPPT fields.

    ECMWF Technical Memorandum 598 (appendix 8.1, eqs. 17 and 18) gives each
    coefficient of degree :math:`l \ge 1` the variance spectrum
    :math:`\exp(-\kappa T\,l(l+1))`, with :math:`\kappa T = (L/a)^2/2` for the
    correlation length :math:`L`, and normalises it so that the grid-point variance is
    :math:`\sigma^2`. With the orthonormal harmonics of torch-harmonics, a field whose
    coefficients have the stationary variance :math:`g_l^2/(2-\alpha)` of the AR(1)
    process of this module has the grid-point variance
    :math:`\sum_l (2l+1)\,g_l^2 / (4\pi(2-\alpha))`, hence

    .. math::

        g_l = b\,e^{-\kappa T\,l(l+1)/2}, \qquad
        b^2 = \frac{4\pi\sigma^2(2-\alpha)}{\sum_{l\ge1}(2l+1)\,e^{-\kappa T\,l(l+1)}}.

    The AR(1) step of this module, :math:`1-\alpha` with :math:`\alpha = \Delta t/\tau`
    (B09 eq. 2), replaces the :math:`\exp(-\Delta t/\tau)` of TM 598; the two agree
    for :math:`\Delta t \ll \tau`.

    Args:
        dynamics: Fixed operators.
        sppt: Settings.
        alpha: Parameter of the AR(1) process of each field, shape (K,).

    Returns:
        Amplitudes (K, L).
    """
    device = dynamics.laplacian.device
    lmax = dynamics.lmax
    sigma = torch.tensor(sppt.amplitudes, dtype=torch.float64, device=device)
    length = torch.tensor(sppt.length_scales, dtype=torch.float64, device=device)
    alpha = alpha.to(device, torch.float64)
    degree = torch.arange(lmax, dtype=torch.float64, device=device).reshape(1, lmax)

    kappa = 0.5 * (length.reshape(-1, 1) / dynamics.radius) ** 2  # kappa T, (K, 1)
    spectrum = torch.exp(-kappa * degree * (degree + 1) / 2)
    spectrum[:, 0] = 0.0  # no global mean
    total = ((2 * degree + 1) * spectrum**2).sum(dim=1)
    b = torch.sqrt(4 * math.pi * sigma**2 * (2 - alpha) / total)
    return b.reshape(-1, 1) * spectrum


def sppt_pattern_on_grid(
    dynamics: SphereShallowWaterDynamics, pattern: TensorLike, clip: float
) -> TensorLike:
    """The SPPT pattern r on the grid: the sum of the fields, clipped.

    Args:
        dynamics: Fixed operators.
        pattern: Coefficients of the K fields, (B, K, L, L).
        clip: Bound of the pattern.

    Returns:
        r on the grid, (B, *grid.shape).
    """
    return dynamics.to_grid(pattern.sum(dim=1)).clamp(-clip, clip)


def physics_increment(
    undamped_state: TensorLike, damped_state: TensorLike, hyperdiffusion: TensorLike
) -> TensorLike:
    """Change of the state by the drag and the relaxation in one step.

    The damping of a step applies the hyperdiffusion, the drag and the relaxation
    together. The physics are the drag and the relaxation: their change is the damped
    state minus the state damped by the hyperdiffusion alone, which is numerical.

    Args:
        undamped_state: State before the damping, (B, 3, L, L).
        damped_state: State after the damping, (B, 3, L, L).
        hyperdiffusion: Factor of the hyperdiffusion for one step, (L, L).

    Returns:
        Increment of the state, (B, 3, L, L).
    """
    after_hyperdiffusion = undamped_state.clone()
    after_hyperdiffusion[:, 1:] = hyperdiffusion * undamped_state[:, 1:]
    return damped_state - after_hyperdiffusion


def sppt_increment(
    dynamics: SphereShallowWaterDynamics, pattern_grid: TensorLike, physics: TensorLike
) -> TensorLike:
    r"""Perturbation of the physics increment by the pattern, SPPT.

    The tendencies of the height and of the wind are multiplied by :math:`1 + r`
    (Buizza et al. 1999; Palmer et al. 2009), so the perturbation is :math:`r` times
    the increment of the drag and the relaxation, on the grid. The wind is perturbed
    as a vector, as ECMWF perturbs the tendencies of u and v, and its vorticity and
    divergence follow. The global mean of the height perturbation is removed, so that
    the mass is conserved (the "global fix" of Leutbecher et al. 2017).

    Args:
        dynamics: Fixed operators.
        pattern_grid: r on the grid, (B, *grid.shape).
        physics: Increment of the drag and the relaxation, (B, 3, L, L).

    Returns:
        Increment of the spectral state, (B, 3, L, L).
    """
    height = dynamics.to_grid(physics[:, 0])
    wind = dynamics.wind(physics[:, 1], physics[:, 2])
    geopotential = dynamics.to_coefficients(pattern_grid * height)
    geopotential[:, 0, 0] = 0.0  # global fix: no change of the mass
    vorticity, divergence = dynamics.vorticity_divergence(
        pattern_grid.unsqueeze(1) * wind
    )
    return torch.tril(torch.stack([geopotential, vorticity, divergence], dim=1))


# ---------------------------------------------------------------------------
# Periodic forcing of The Well (planetswe)
# ---------------------------------------------------------------------------


@dataclass
class PeriodicForcingSettings:
    """Settings of the daily and yearly height forcing of The Well's planetswe.

    The defaults of the simulator are those of ``gen_SWE_from_ic_file.py`` in
    PolymathicAI/the_well at commit aac88c0: 2 m per hour, width 2 pi / 3, maximum
    declination 0.4 rad, a day of 24 h and a year of 1008 h.

    Attributes:
        amplitude: Amplitude h_f0 of the height tendency [m/s].
        width: Width sigma of the latitude profile [rad].
        declination: Maximum latitude of the centre [rad].
        day: Period of the longitude of the centre [s].
        year: Period of the latitude of the centre [s].
    """

    amplitude: float
    width: float
    declination: float
    day: float
    year: float


def periodic_height_forcing(
    dynamics: SphereShallowWaterDynamics,
    periodic: PeriodicForcingSettings,
    time: float,
) -> TensorLike:
    r"""Height tendency of The Well's forcing at one time.

    .. math::

        F = h_{f0}\,\cos(\lambda - \lambda_c)\,
            \exp\big[-(\varphi - \varphi_c)^2/\sigma^2\big], \qquad
        \lambda_c = 2\pi t / \text{day}, \quad
        \varphi_c = \delta\,\sin(2\pi t / \text{year}),

    with :math:`\varphi` the latitude: a source on the half of the sphere centred on
    :math:`\lambda_c` and a sink on the other half, with no net mass. As in The Well,
    the centre moves eastward, and the profile does not vanish at the poles.

    Args:
        dynamics: Fixed operators.
        periodic: Settings.
        time: Time t since the start of the run [s].

    Returns:
        F on the grid [m/s], shape (*grid.shape).
    """
    centre_lon = 2 * math.pi * time / periodic.day
    centre_lat = periodic.declination * math.sin(2 * math.pi * time / periodic.year)
    return (
        periodic.amplitude
        * torch.cos(dynamics.longitude - centre_lon)
        * torch.exp(-(((dynamics.latitude - centre_lat) / periodic.width) ** 2))
    )


def periodic_forcing_increment(
    dynamics: SphereShallowWaterDynamics,
    periodic: PeriodicForcingSettings,
    time: float,
    batch: int,
) -> TensorLike:
    """Increment of the state by The Well's forcing over the step from ``time``.

    The height gains F dt, with F at the middle of the step, as the geopotential of
    the fluid depth: g F dt. The forcing is the same for every simulation.

    Args:
        dynamics: Fixed operators and time step.
        periodic: Settings.
        time: Time at the start of the step [s].
        batch: Number of simulations B.

    Returns:
        Increment of the spectral state, (B, 3, L, L); only the geopotential changes.
    """
    forcing = periodic_height_forcing(dynamics, periodic, time + dynamics.dt / 2)
    coefficients = dynamics.to_coefficients(forcing.unsqueeze(0))
    increment = torch.zeros(
        batch,
        3,
        dynamics.lmax,
        dynamics.lmax,
        dtype=coefficients.dtype,
        device=coefficients.device,
    )
    increment[:, 0] = dynamics.gravity * dynamics.dt * torch.tril(coefficients[0])
    return increment
