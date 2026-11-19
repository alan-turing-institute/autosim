r"""Rossby and inertia-gravity waves of Shamir and Paldor (2016).

Shamir and Paldor (2016, Q. J. R. Meteorol. Soc. 142, 2705-2714, "SP16") give closed
forms of the wave solutions of the shallow water equations linearised about a fluid at
rest of depth :math:`H` (Paldor, De-Leon and Shamir 2013). A wave is

.. math::

    (u', v', h') = \big(\tilde u(\varphi), \tilde v(\varphi), \tilde h(\varphi)\big)
        \, e^{ik(\lambda - Ct)},

with an integer zonal wavenumber :math:`k`, a mode number :math:`n \ge 1` (the number
of zero crossings between the poles) and an angular phase speed :math:`C` [rad/s].

**Phase speeds** (SP16 eqs. 7-11). The three roots of a cubic,

.. math::

    C_j = -\frac{\Delta_j}{3k^2}\Big(1 + \frac{\Delta_0}{\Delta_j^2}\Big), \quad
    \Delta_0 = 3k^2E_n, \quad
    \Delta_j = \Big[\frac{\Delta_4 + \sqrt{\Delta_4^2 - 4\Delta_0^3}}{2}\Big]^{1/3}
        e^{2\pi i j/3}, \quad
    \Delta_4 = -\frac{54\,gH\Omega k^4}{a^2},

with :math:`E_n = gH(n+\sigma)^2/a^2` and :math:`\sigma = \tfrac12 + \sqrt{\tfrac14 +
k^2}`, give the Rossby wave :math:`C_R = -\min_j|C_j|`, the westward inertia-gravity
wave :math:`C_{WIG} = \min_j C_j` and the eastward one :math:`C_{EIG} = \max_j C_j`.

**Structure** (SP16 eqs. 12-18). With :math:`\psi_n = \cos^\sigma\varphi\,
C_n^{(\sigma)}(\sin\varphi)` (eq. 14, :math:`C_n^{(\sigma)}` the Gegenbauer
polynomial), :math:`\omega = kC` and :math:`\tilde V = i\tilde v/k`:

- Rossby (eqs. 12, 13): :math:`\tilde V = \sqrt{2\Omega|K_-|}\,\psi_n/\cos\varphi`,
  :math:`\tilde h = \frac{aH\sqrt{2\Omega|K_-|}}{K_-}\{\psi_n' + \tan\varphi\,
  (\tfrac12 K_+/K_- - 2\Omega/C)\,\psi_n\}`, with :math:`K_\pm = (gH \pm C^2a^2
  \cos^2\varphi)/(C\cos\varphi)`. This prefactor follows from rows 1 and 3 of SP16
  eq. 6 and gives a height; the printed eq. 12b reads as
  :math:`\sqrt{2\Omega a^2 H|K_-|}/K_-`, which does not;
- inertia-gravity (eqs. 15-17): :math:`\tilde h = \sqrt{2\Omega a^2 H|W_1|/g}\,
  \psi_n`, :math:`\tilde V = \frac{\sqrt{2\Omega gH|W_1|}}{W_1\cos\varphi}\{\psi_n'
  + \tan\varphi\,(\tfrac12 - 2\Omega/W_2 + 2\Omega/C)\,\psi_n\}`, with :math:`W_1 =
  (\omega^2 - 4\Omega^2\sin^2\varphi)/(C\cos\varphi)` and :math:`W_2 = (\omega^2 -
  4\Omega^2\sin^2\varphi)/(2\Omega\cos^2\varphi)`;
- both (eq. 18): :math:`\tilde u = \frac{2\Omega\sin\varphi}{C}\tilde V +
  \frac{g}{aC\cos\varphi}\tilde h`.

:math:`W_1` vanishes where :math:`|\omega| = 2\Omega|\sin\varphi|`, so an
inertia-gravity wave needs :math:`|\omega| > 2\Omega` (a period below 12 h).

At :math:`t = 0` the fields are :math:`u = \tilde u\cos k\lambda`,
:math:`v = k\tilde V\sin k\lambda` and :math:`h = \tilde h\cos k\lambda`. SP16 use the
amplitude :math:`A = 10^{-8}` of :math:`\psi_n`; here the fields are scaled so that
the largest wind speed equals the sampled amplitude, which may reach the nonlinear
regime. The formulas approximate the exact linear waves for
:math:`gH \gg (2\Omega a)^2/n^2`. For the published wave, :math:`(n, k) = (5, 10)` at
:math:`gH = 5\times10^4` m^2/s^2, SP16 find frequencies within 1.8 % (Rossby) and
0.3 % (eastward) of the exact ones.
"""

from __future__ import annotations

import cmath
import math

import torch

from autosim.types import TensorLike

from .dynamics import SphereShallowWaterDynamics

# Integer codes of the wave types
WAVE_TYPES = ("rossby", "eastward_gravity", "westward_gravity")

# Default sampling ranges. The integer parameters are rounded to the nearest integer;
# their bounds are half-integers so that every integer is drawn equally often. With
# k >= 3 every inertia-gravity wave of n <= 5 is faster than 2 Omega at the depth of
# SP16 (gH = 5e4 m^2/s^2), and with k <= 8 the period stays above 3.7 h. The published
# wave is (n, k) = (5, 10).
SHAMIR_PALDOR_RANGES: dict[str, tuple[float, float]] = {
    "wave_type": (-0.5, 2.5),  # 0 Rossby, 1 eastward, 2 westward inertia-gravity
    "zonal_wavenumber": (2.5, 8.5),  # k = 3 to 8, SP16: 10
    "mode_number": (0.5, 5.5),  # n = 1 to 5, SP16: 5
    "wave_amplitude": (5.0, 40.0),  # largest wind speed [m/s]
}


def phase_speeds(
    mode_number: int,
    zonal_wavenumber: int,
    gravity_depth: float,
    omega: float,
    radius: float,
) -> tuple[float, float, float]:
    """Angular phase speeds of the Rossby, westward and eastward waves, SP16 eqs. 7-11.

    Args:
        mode_number: n >= 1.
        zonal_wavenumber: k >= 1.
        gravity_depth: gH [m^2/s^2].
        omega: Rotation rate [1/s].
        radius: Planet radius [m].

    Returns:
        (C_R, C_WIG, C_EIG) [rad/s].
    """
    n, k = mode_number, zonal_wavenumber
    if n < 1 or k < 1:
        msg = "the waves of SP16 need a mode number n >= 1 and a wavenumber k >= 1"
        raise ValueError(msg)
    sigma = 0.5 + math.sqrt(0.25 + k * k)
    energy = gravity_depth / radius**2 * (n + sigma) ** 2  # E_n, eq. 9
    delta_0 = 3 * k * k * energy  # eq. 8a
    delta_4 = -54 * gravity_depth * omega * k**4 / radius**2  # eq. 8c
    base = ((delta_4 + cmath.sqrt(delta_4 * delta_4 - 4 * delta_0**3)) / 2) ** (1 / 3)
    roots = []
    for j in (1, 2, 3):
        delta_j = base * cmath.exp(2j * math.pi * j / 3)  # eq. 8b
        roots.append((-(1 / (3 * k * k)) * delta_j * (1 + delta_0 / delta_j**2)).real)
    rossby = -min(abs(root) for root in roots)
    return rossby, min(roots), max(roots)


def wave_phase_speed(
    wave_type: int,
    mode_number: int,
    zonal_wavenumber: int,
    gravity_depth: float,
    omega: float,
    radius: float,
) -> float:
    """Angular phase speed C [rad/s] of one wave type (0 Rossby, 1 EIG, 2 WIG)."""
    rossby, westward, eastward = phase_speeds(
        mode_number, zonal_wavenumber, gravity_depth, omega, radius
    )
    return {0: rossby, 1: eastward, 2: westward}[wave_type]


def gegenbauer(degree: int, order: float, x: TensorLike) -> TensorLike:
    r"""Gegenbauer polynomial :math:`C_n^{(\alpha)}(x)` by its three-term recurrence.

    :math:`C_0 = 1`, :math:`C_1 = 2\alpha x`, and
    :math:`m\,C_m = 2x(m + \alpha - 1)\,C_{m-1} - (m + 2\alpha - 2)\,C_{m-2}`.

    Args:
        degree: n >= 0; a negative degree gives zero.
        order: alpha > 0.
        x: Points, any shape.

    Returns:
        The polynomial at x.
    """
    if degree < 0:
        return torch.zeros_like(x)
    previous = torch.ones_like(x)
    if degree == 0:
        return previous
    current = 2 * order * x
    for m in range(2, degree + 1):
        previous, current = (
            current,
            (2 * x * (m + order - 1) * current - (m + 2 * order - 2) * previous) / m,
        )
    return current


def meridional_structure(
    wave_type: int,
    mode_number: int,
    zonal_wavenumber: int,
    latitude: TensorLike,
    gravity: float,
    depth: float,
    omega: float,
    radius: float,
) -> tuple[TensorLike, TensorLike, TensorLike]:
    r"""Latitude profiles :math:`\tilde u`, :math:`\tilde V` and :math:`\tilde h`, SP16.

    For :math:`A = 1` in eq. 14. The meridional wind is :math:`\tilde v = -ik\tilde V`.
    The profiles vanish at the poles, where they are set to zero.

    Args:
        wave_type: 0 Rossby, 1 eastward, 2 westward inertia-gravity.
        mode_number: n >= 1.
        zonal_wavenumber: k >= 1.
        latitude: Latitudes [rad], any shape, float64.
        gravity: g [m/s^2].
        depth: H [m].
        omega: Rotation rate [1/s].
        radius: Planet radius [m].

    Returns:
        u~ [m/s], V~ [m/s] and h~ [m] at the latitudes, for the amplitude A = 1.

    Raises:
        ValueError: For an inertia-gravity wave with |omega| <= 2 Omega, whose
            structure is singular where W_1 vanishes.
    """
    n, k = mode_number, zonal_wavenumber
    gravity_depth = gravity * depth
    speed = wave_phase_speed(wave_type, n, k, gravity_depth, omega, radius)
    frequency = k * speed
    if wave_type != 0 and abs(frequency) <= 2 * omega:
        msg = (
            f"the inertia-gravity wave (n, k) = ({n}, {k}) has |omega| <= 2 Omega: its "
            "structure (SP16 eq. 15) is singular where W_1 vanishes"
        )
        raise ValueError(msg)

    sigma = 0.5 + math.sqrt(0.25 + k * k)
    cosine = torch.cos(latitude)
    inside = cosine > 1e-12  # the poles, where every profile vanishes
    cosine = torch.where(inside, cosine, 1.0)
    sine = torch.sin(latitude)
    tangent = sine / cosine

    # psi_n (eq. 14) and its latitude derivative: dC_n/dx = 2 sigma C_{n-1}^(sigma + 1)
    polynomial = gegenbauer(n, sigma, sine)
    slope = 2 * sigma * gegenbauer(n - 1, sigma + 1, sine)
    psi = cosine**sigma * polynomial
    dpsi = (
        -sigma * cosine ** (sigma - 1) * sine * polynomial
        + cosine ** (sigma + 1) * slope
    )

    two_omega = 2 * omega
    if wave_type == 0:  # Rossby wave, eqs. 12 and 13
        k_plus = (gravity_depth + speed**2 * radius**2 * cosine**2) / (speed * cosine)
        k_minus = (gravity_depth - speed**2 * radius**2 * cosine**2) / (speed * cosine)
        v_tilde = torch.sqrt(two_omega * k_minus.abs()) / cosine * psi
        # Prefactor a H sqrt(2 Omega |K_-|) / K_-, which follows from rows 1 and 3 of
        # SP16 eq. 6 and is a height (the printed eq. 12b reads as sqrt(2 Omega a^2 H
        # |K_-|) / K_-, which is not)
        h_tilde = (
            radius
            * depth
            * torch.sqrt(two_omega * k_minus.abs())
            / k_minus
            * (dpsi + tangent * (0.5 * k_plus / k_minus - two_omega / speed) * psi)
        )
    else:  # inertia-gravity waves, eqs. 15 to 17
        gap = frequency**2 - two_omega**2 * sine**2
        w_1 = gap / (speed * cosine)
        w_2 = gap / (two_omega * cosine**2)
        h_tilde = torch.sqrt(two_omega * radius**2 * depth * w_1.abs() / gravity) * psi
        v_tilde = (
            torch.sqrt(two_omega * gravity_depth * w_1.abs())
            / (w_1 * cosine)
            * (dpsi + tangent * (0.5 - two_omega / w_2 + two_omega / speed) * psi)
        )
    # zonal wind, eq. 18
    u_tilde = (
        two_omega * sine / speed * v_tilde
        + gravity / (radius * speed * cosine) * h_tilde
    )

    zero = torch.zeros_like(latitude)
    return (
        torch.where(inside, u_tilde, zero),
        torch.where(inside, v_tilde, zero),
        torch.where(inside, h_tilde, zero),
    )


def shamir_paldor_initial_condition(
    dynamics: SphereShallowWaterDynamics,
    wave_type: TensorLike,
    zonal_wavenumber: TensorLike,
    mode_number: TensorLike,
    wave_amplitude: TensorLike,
) -> TensorLike:
    """One wave of SP16 per simulation, at rest depth H = ``mean_depth``.

    The fields at t = 0 are scaled so that the largest wind speed on the grid equals
    ``wave_amplitude``. The geopotential is g (H + h).

    Args:
        dynamics: Fixed operators; ``mean_depth`` is the depth H of the waves.
        wave_type: 0 Rossby, 1 eastward, 2 westward inertia-gravity, integer values,
            shape (B,).
        zonal_wavenumber: k, integer values, shape (B,).
        mode_number: n, integer values, shape (B,).
        wave_amplitude: Largest wind speed [m/s], shape (B,).

    Returns:
        Spectral state (B, 3, L, L).
    """
    latitude = dynamics.latitude
    longitude = dynamics.longitude
    winds = []
    heights = []
    for index in range(wave_type.shape[0]):
        k = round(float(zonal_wavenumber[index]))
        u_tilde, v_tilde, h_tilde = meridional_structure(
            round(float(wave_type[index])),
            round(float(mode_number[index])),
            k,
            latitude,
            dynamics.gravity,
            dynamics.mean_depth,
            dynamics.omega,
            dynamics.radius,
        )
        eastward = u_tilde * torch.cos(k * longitude)
        northward = k * v_tilde * torch.sin(k * longitude)
        height = h_tilde * torch.cos(k * longitude)
        scale = (
            float(wave_amplitude[index]) / torch.sqrt(eastward**2 + northward**2).max()
        )
        winds.append(scale * torch.stack([eastward, northward]))
        heights.append(scale * height)

    wind = torch.stack(winds)  # (B, 2, *grid.shape)
    height = torch.stack(heights)
    vorticity, divergence = dynamics.vorticity_divergence(wind)
    geopotential = dynamics.to_coefficients(
        dynamics.gravity * (dynamics.mean_depth + height)
    )
    state = torch.stack([geopotential, vorticity, divergence], dim=1)
    return torch.tril(state)  # keep m <= l
