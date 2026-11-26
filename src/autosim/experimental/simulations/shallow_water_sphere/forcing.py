r"""Forcing of the shallow water equations on the sphere.

Three forcings, which can be combined, each a class that owns its random state and
gives the increment of one solver step:

- ``SkebsForcing``: kinetic-energy backscatter (SKEBS), below, with the global and the
  local backscatter of the dissipation;
- ``SpptForcing``: stochastically perturbed physics tendencies (SPPT), the increment of
  the drag and the relaxation multiplied by a random pattern;
- ``PeriodicForcing``: The Well's daily and yearly height forcing.

Their settings, in user units, are the dataclasses of module ``settings``.

Kinetic-energy backscatter forcing follows Berner, Shutts, Leutbecher and Palmer
(2009, J. Atmos. Sci. 66, 603-626, "B09" below). A streamfunction pattern
:math:`\psi'` is expanded in spherical harmonics, and each coefficient follows the
first-order autoregressive process of B09 eq. (2),

.. math::

    \psi'^m_l(t + \Delta t) = (1 - \alpha)\,\psi'^m_l(t)
        + g_l \sqrt{\alpha}\,\epsilon^m_l(t), \qquad g_l = b\,l^p,

with :math:`\alpha = \Delta t / \tau` for a decorrelation time :math:`\tau`
(:math:`\alpha = 1` is white noise) and :math:`\epsilon` a complex Gaussian noise of
unit variance. The pattern is an increment per step: the vorticity gains
:math:`\nabla^2\psi'` after each solver step.

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
the forcing follows :math:`l^{2p+3}` (B09 eq. 10). ``SkebsSettings.energy_rate`` is
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

from .diagnostics import degree_power, global_integral
from .dynamics import SphereShallowWaterDynamics
from .settings import PeriodicForcingSettings, SpptSettings


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
        msg = "a correlation time must be 0 or at least one solver step"
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
    smoothed = dynamics.isht(taper * dynamics.sht(dissipation))
    return smoothed.clamp(min=0.0)


def kinetic_forcing_increment(
    dynamics: SphereShallowWaterDynamics,
    pattern: TensorLike,
    energy_rate: TensorLike,
) -> TensorLike:
    r"""Increment of the state from the patterns, for one step.

    The pattern holds :math:`\psi'` for a unit energy rate. It is multiplied by the
    square root of the energy rate, then the vorticity gains :math:`\nabla^2\psi'`.
    A rate of shape (B,) multiplies the coefficients. A rate
    on the grid, (B, \*grid.shape), multiplies the patterns in grid space, as in B09
    eq. (13).

    Args:
        dynamics: Fixed operators.
        pattern: Pattern coefficients (B, 1, L, L).
        energy_rate: Energy rate [m^2/s^3], shape (B,) or (B, *grid.shape).

    Returns:
        Increment of the spectral state, (B, 3, L, L); the geopotential part is zero.
    """
    batch = pattern.shape[0]
    energy_rate = energy_rate.to(pattern.device, torch.float64)
    if energy_rate.ndim == 1:
        scaled = torch.sqrt(energy_rate).reshape(batch, 1, 1, 1) * pattern
    else:
        pattern_grid = dynamics.isht(pattern)  # (B, P, *grid.shape)
        scaled_grid = torch.sqrt(energy_rate).unsqueeze(1) * pattern_grid
        scaled = dynamics.sht(scaled_grid)

    increment = torch.zeros(
        batch,
        3,
        dynamics.lmax,
        dynamics.lmax,
        dtype=pattern.dtype,
        device=pattern.device,
    )
    increment[:, 1] = dynamics.laplacian * scaled[:, 0]
    return increment


# ---------------------------------------------------------------------------
# Stochastically perturbed physics tendencies (SPPT)
# ---------------------------------------------------------------------------


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
    return dynamics.isht(pattern.sum(dim=1)).clamp(-clip, clip)


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
    height = dynamics.isht(physics[:, 0])
    wind = dynamics.wind(physics[:, 1], physics[:, 2])
    geopotential = dynamics.sht(pattern_grid * height)
    geopotential[:, 0, 0] = 0.0  # global fix: no change of the mass
    vorticity, divergence = dynamics.vorticity_divergence(
        pattern_grid.unsqueeze(1) * wind
    )
    return torch.tril(torch.stack([geopotential, vorticity, divergence], dim=1))


# ---------------------------------------------------------------------------
# Periodic forcing of The Well (planetswe)
# ---------------------------------------------------------------------------


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
        periodic: Settings, amplitude in m/hour and periods in hours.
        time: Time t since the start of the run [s].

    Returns:
        F on the grid [m/s], shape (*grid.shape).
    """
    day, year = 3600 * periodic.day, 3600 * periodic.year
    centre_lon = 2 * math.pi * time / day
    centre_lat = periodic.declination * math.sin(2 * math.pi * time / year)
    return (
        periodic.amplitude
        / 3600
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
    coefficients = dynamics.sht(forcing.unsqueeze(0))
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


# ---------------------------------------------------------------------------
# The forcings of one batch, each with its own random state
# ---------------------------------------------------------------------------


@dataclass
class StepState:
    """The solver step that a forcing increment follows.

    Attributes:
        time: Time at the start of the step [s].
        undamped_state: State after the conservative step, before the damping,
            (B, 3, L, L).
        damped_state: State after the damping, (B, 3, L, L).
        hyperdiffusion: Factor of the hyperdiffusion for one step, (L, L).
        drag: Factor of the drag for one step, (B,), or None without drag.
    """

    time: float
    undamped_state: TensorLike
    damped_state: TensorLike
    hyperdiffusion: TensorLike
    drag: TensorLike | None


class SkebsForcing:
    """SKEBS pattern of B simulations, with its global and local backscatter.

    The forcing holds the values of its batch, in SI units, with one value per
    simulation for the fields sampled in ``parameters_range``: the simulator builds it
    from ``SkebsSettings`` and the backscatter settings. The pattern starts from its
    stationary distribution. Each ``increment`` uses the current pattern, then
    advances it by one step of the AR(1) process, with one random generator per
    simulation.

    Args:
        dynamics: Fixed operators and time step.
        generators: One random generator per simulation.
        energy_rate: Base energy rate [m^2/s^3], (B,).
        correlation_time: Decorrelation time of the pattern [s], (B,); 0 is white.
        spectral_slope: Exponent p of the noise amplitude g_l = b l^p, (B,).
        degree_min: Lowest forced degree.
        degree_max: Highest forced degree.
        global_fraction: Share of the global mean dissipation added to the rate, (B,),
            or None without global backscatter.
        global_include_drag: Whether the drag loss counts in the global dissipation.
        local_fraction: Share of the local dissipation added to the rate, (B,), or None
            without local backscatter.
        local_include_drag: Whether the drag loss counts in the local dissipation.
    """

    def __init__(
        self,
        dynamics: SphereShallowWaterDynamics,
        generators: list[torch.Generator],
        energy_rate: TensorLike,
        correlation_time: TensorLike,
        spectral_slope: TensorLike,
        degree_min: int,
        degree_max: int,
        global_fraction: TensorLike | None = None,
        global_include_drag: bool = False,
        local_fraction: TensorLike | None = None,
        local_include_drag: bool = False,
    ) -> None:
        """Store the values and draw the initial pattern from its stationary law."""
        device = dynamics.laplacian.device
        self.dynamics = dynamics
        self.generators = generators
        self.base_rate = energy_rate.to(device, torch.float64)
        self.correlation_time = correlation_time
        self.spectral_slope = spectral_slope
        self.degree_min = degree_min
        self.degree_max = degree_max
        self.global_fraction = global_fraction
        self.global_include_drag = global_include_drag
        self.local_fraction = local_fraction
        self.local_include_drag = local_include_drag

        self.alpha = autoregressive_parameter(correlation_time, dynamics.dt).to(device)
        self.amplitudes = unit_rate_amplitudes(
            dynamics, self.alpha, spectral_slope, degree_min, degree_max
        ).unsqueeze(1)  # (B, 1, L): one streamfunction pattern
        noise = complex_noise(generators, 1, dynamics.lmax)
        self.pattern = stationary_pattern(noise, self.amplitudes, self.alpha)
        self.global_on = global_fraction is not None and bool(
            (global_fraction > 0).any()
        )
        self.local_on = local_fraction is not None and bool((local_fraction > 0).any())
        self.needs_undamped = self.global_on or self.local_on
        self.taper = dissipation_taper(dynamics.lmax, device)
        self.mean_rate = self.base_rate  # global mean rate of the last increment

    def energy_rate(self, step: StepState) -> TensorLike:
        """Energy rate of this step: base rate plus the backscatter of the dissipation.

        The global backscatter adds its fraction of the global mean dissipation, one
        value per simulation. The local backscatter adds its fraction of the smoothed
        dissipation field (B09 eq. 13), and the rate becomes a field on the grid.

        Args:
            step: The solver step.

        Returns:
            Energy rate [m^2/s^3], shape (B,) or (B, *grid.shape).
        """
        device = self.base_rate.device
        batch = self.base_rate.shape[0]
        rate = self.base_rate
        lost = step.hyperdiffusion[:, :1].unsqueeze(0)  # (1, L, 1): damping per degree
        with_drag = lost if step.drag is None else lost * step.drag.reshape(batch, 1, 1)
        if self.global_on:
            assert self.global_fraction is not None  # global_on implies it
            damping = with_drag if self.global_include_drag else lost
            fraction = self.global_fraction.to(device, torch.float64)
            dissipation = global_dissipation_rate(
                self.dynamics, step.undamped_state, damping[..., 0]
            )
            rate = rate + fraction * dissipation
        if self.local_on:
            assert self.local_fraction is not None  # local_on implies it
            damping = with_drag if self.local_include_drag else lost
            fraction = self.local_fraction.to(device, torch.float64)
            dissipation = local_dissipation_rate(
                self.dynamics, step.undamped_state, damping, self.taper
            )
            shape = (batch,) + (1,) * (dissipation.ndim - 1)
            rate = rate.reshape(shape) + fraction.reshape(shape) * dissipation
        return rate

    def increment(self, step: StepState) -> TensorLike:
        """Increment of this step, then one AR(1) step of the pattern.

        Args:
            step: The solver step.

        Returns:
            Increment of the spectral state, (B, 3, L, L); only the vorticity changes.
        """
        rate = self.energy_rate(step)
        increment = kinetic_forcing_increment(self.dynamics, self.pattern, rate)
        noise = complex_noise(self.generators, 1, self.dynamics.lmax)
        self.pattern = advance_pattern(self.pattern, noise, self.amplitudes, self.alpha)
        if rate.ndim == 1:
            self.mean_rate = rate
        else:
            self.mean_rate = global_integral(rate, self.dynamics.quad_weights) / (
                4 * torch.pi
            )
        return increment

    def unit_tendency(self) -> TensorLike:
        """Tendency of the pattern for a unit energy rate, (B, 3, L, L), per second."""
        unit = torch.ones_like(self.base_rate)
        return kinetic_forcing_increment(self.dynamics, self.pattern, unit) / (
            self.dynamics.dt
        )


class SpptForcing:
    """SPPT pattern of B simulations: K AR(1) fields, one per scale.

    The fields start from their stationary distribution. Each ``increment`` multiplies
    the physics increment of the step by the current clipped pattern, then advances the
    fields by one AR(1) step, with one random generator per simulation.

    Args:
        dynamics: Fixed operators and time step.
        settings: Settings, correlation times in hours.
        generators: One random generator per simulation.
    """

    needs_undamped = True

    def __init__(
        self,
        dynamics: SphereShallowWaterDynamics,
        settings: SpptSettings,
        generators: list[torch.Generator],
    ) -> None:
        """Draw the initial fields from their stationary distribution."""
        device = dynamics.laplacian.device
        batch, lmax = len(generators), dynamics.lmax
        self.dynamics = dynamics
        self.settings = settings
        self.generators = generators
        self.n_fields = len(settings.amplitudes)
        times = torch.tensor(
            [3600 * value for value in settings.correlation_times], dtype=torch.float64
        )
        field_alpha = autoregressive_parameter(times, dynamics.dt).to(device)  # (K,)
        amplitudes = sppt_noise_amplitudes(dynamics, settings, field_alpha)
        self.amplitudes = amplitudes.expand(batch, self.n_fields, lmax)
        self.alpha = field_alpha.expand(batch, self.n_fields)
        noise = complex_noise(generators, self.n_fields, lmax)
        self.pattern = stationary_pattern(noise, self.amplitudes, self.alpha)

    def increment(self, step: StepState) -> TensorLike:
        """Perturbation of the physics increment of this step, then one AR(1) step.

        Args:
            step: The solver step.

        Returns:
            Increment of the spectral state, (B, 3, L, L).
        """
        physics = physics_increment(
            step.undamped_state, step.damped_state, step.hyperdiffusion
        )
        pattern_grid = sppt_pattern_on_grid(
            self.dynamics, self.pattern, self.settings.clip
        )
        increment = sppt_increment(self.dynamics, pattern_grid, physics)
        noise = complex_noise(self.generators, self.n_fields, self.dynamics.lmax)
        self.pattern = advance_pattern(self.pattern, noise, self.amplitudes, self.alpha)
        return increment


class PeriodicForcing:
    """The Well's height forcing, the same for the B simulations of a batch.

    Args:
        dynamics: Fixed operators and time step.
        settings: Settings, amplitude in m/hour and periods in hours.
        batch: Number of simulations B.
    """

    needs_undamped = False

    def __init__(
        self,
        dynamics: SphereShallowWaterDynamics,
        settings: PeriodicForcingSettings,
        batch: int,
    ) -> None:
        """Store the settings."""
        self.dynamics = dynamics
        self.settings = settings
        self.batch = batch

    def increment(self, step: StepState) -> TensorLike:
        """Height increment of the step that starts at ``step.time``, (B, 3, L, L)."""
        return periodic_forcing_increment(
            self.dynamics, self.settings, step.time, self.batch
        )
