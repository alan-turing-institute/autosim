r"""Rotation of fields on the sphere by an element of SO(3).

A rotation :math:`R` maps a field :math:`f` to :math:`g(\mathbf{x}) =
f(R^{-1}\mathbf{x})`: the pattern of :math:`f` is carried by :math:`R`. Rotations do
not mix the degrees of a spherical-harmonic expansion, so the rotation of a
band-limited field is exact: the coefficients of each degree :math:`l` are multiplied
by the Wigner D-matrix of that degree.

**Angles.** :math:`R = R_z(\alpha)\,R_y(\beta)\,R_z(\gamma)`: a rotation by
:math:`\gamma` about the z axis (the geographic axis), then by :math:`\beta` about the
y axis, then by :math:`\alpha` about the z axis, all about fixed axes. The inverse
rotation has the angles :math:`(-\gamma, -\beta, -\alpha)`.

**Coefficients.** With the usual harmonics :math:`Y_l^m` (Condon-Shortley phase,
orders :math:`-l..l`) and coefficients :math:`a_l^m`,

.. math::

    a'^m_l = \sum_{m'} D^l_{m m'}\, a^{m'}_l, \qquad
    D^l_{m m'} = e^{-i m \alpha}\, d^l_{m m'}(\beta)\, e^{-i m' \gamma}, \qquad
    d^l(\beta) = \exp(-i \beta J_y).

:math:`-i J_y` is a real antisymmetric matrix with entries
:math:`\mp\tfrac12\sqrt{(l - m)(l + m + 1)}` next to the diagonal, so
:math:`d^l(\beta)` is the exponential of a real matrix. torch-harmonics stores the
orders :math:`m \ge 0` without the Condon-Shortley phase, :math:`c_l^m`; the two sets
are related by :math:`a_l^m = (-1)^m c_l^m` and :math:`a_l^{-m} = \overline{c_l^m}`.

**Winds.** The geopotential, the vorticity, the divergence, the Coriolis parameter
and the orography are scalar fields, and this module rotates their coefficients. The
wind is a spin-1 field: its two components are measured in the local east and north
directions, which a rotation changes from point to point, so rotating the maps of u
and v as two scalars is wrong. The wind rebuilt from the rotated vorticity and
divergence is the correctly rotated wind. ``rotate_fields`` does this for saved
frames.
"""

from __future__ import annotations

import math

import torch

from autosim.types import TensorLike

from .dynamics import (
    SphereShallowWaterDynamics,
    fields_to_spectral,
    spectral_to_fields,
)


def wigner_d_matrix(degree: int, beta: float) -> TensorLike:
    r"""Wigner small-d matrix :math:`d^l_{m m'}(\beta)` of one degree.

    Args:
        degree: Degree l.
        beta: Rotation angle about the y axis [rad].

    Returns:
        Real matrix (2l + 1, 2l + 1), float64, indexed [m + l, m' + l].
    """
    order = torch.arange(-degree, degree, dtype=torch.float64)  # m = -l .. l - 1
    ladder = torch.sqrt((degree - order) * (degree + order + 1))
    generator = torch.zeros(2 * degree + 1, 2 * degree + 1, dtype=torch.float64)
    index = torch.arange(2 * degree)
    generator[index + 1, index] = -0.5 * ladder  # -i J_y, element (m + 1, m)
    generator[index, index + 1] = 0.5 * ladder  # -i J_y, element (m, m + 1)
    return torch.linalg.matrix_exp(beta * generator)


def rotate_coefficients(
    coefficients: TensorLike, alpha: float, beta: float, gamma: float
) -> TensorLike:
    r"""Rotate scalar fields given by their spherical-harmonic coefficients.

    Args:
        coefficients: Coefficients (..., L, L) of real scalar fields, torch-harmonics
            convention (orders m >= 0, no Condon-Shortley phase, m = 0 real). A
            spectral state (B, 3, L, L) is rotated field by field.
        alpha: Last rotation, about the z axis [rad].
        beta: Middle rotation, about the y axis [rad].
        gamma: First rotation, about the z axis [rad].

    Returns:
        Coefficients of the rotated fields, same shape and device.
    """
    device = coefficients.device
    c = coefficients.to("cpu", torch.complex128)
    lmax = c.shape[-1]
    rotated = torch.zeros_like(c)
    for degree in range(lmax):
        order = torch.arange(-degree, degree + 1, dtype=torch.float64)
        sign = (-1.0) ** torch.arange(degree + 1, dtype=torch.float64)  # (-1)^m

        # Coefficients a_m of the usual harmonics, m = -l .. l
        positive = sign * c[..., degree, : degree + 1]  # m = 0 .. l
        negative = c[..., degree, 1 : degree + 1].conj().flip(-1)  # m = -l .. -1
        a = torch.cat([negative, positive], dim=-1)

        # a'_m = exp(-i m alpha) sum_m' d[m, m'] exp(-i m' gamma) a_m'
        d = wigner_d_matrix(degree, beta).to(torch.complex128)
        a = torch.exp(-1j * order * gamma) * a
        a = a @ d.T
        a = torch.exp(-1j * order * alpha) * a

        rotated[..., degree, : degree + 1] = sign * a[..., degree:]
    rotated[..., 0] = rotated[..., 0].real  # m = 0 coefficients of a real field
    return rotated.to(device)


def rotate_fields(
    dynamics: SphereShallowWaterDynamics,
    fields: TensorLike,
    alpha: float,
    beta: float,
    gamma: float,
) -> TensorLike:
    """Rotate saved frames h, u_phi, u_theta, the wind as a spin-1 field.

    The frames are converted to the geopotential of the free surface, the vorticity
    and the divergence, which are scalars; these are rotated and converted back. The
    rotation is exact for band-limited fields, because a rotation does not mix the
    degrees of a spherical-harmonic expansion. Rotating the maps of u_phi and u_theta
    as two scalars would be wrong: the components are measured along the local east
    and south directions, which the rotation turns from point to point.

    The shallow water equations commute with rotations once the rotation pole of the
    Coriolis parameter and the orography are rotated with the state. The tests use
    this function to check that property of the solver: simulating a rotated problem
    equals rotating the simulation, to 1e-9 after 12 h
    (``test_rotated_problem_matches_rotated_simulation``). The same check applies to
    an emulator: rotating its input frames and its constant fields
    (``return_constant_fields``) by R must rotate its forecast by R if the emulator is
    SO(3)-equivariant. ``rotate_point`` gives the rotated pole and mountain summit,
    and ``rotate_coefficients`` rotates the Coriolis parameter and the orography.

    Args:
        dynamics: Fixed operators of the grid of the frames.
        fields: Frames (B, *grid.shape, 3) in the order h, u_phi, u_theta.
        alpha: Last rotation, about the z axis [rad].
        beta: Middle rotation, about the y axis [rad].
        gamma: First rotation, about the z axis [rad].

    Returns:
        Rotated frames (B, *grid.shape, 3), float64.
    """
    state = fields_to_spectral(dynamics, fields)
    rotated_state = rotate_coefficients(state, alpha, beta, gamma)
    return spectral_to_fields(dynamics, rotated_state)


def rotate_point(
    longitude: float, latitude: float, alpha: float, beta: float, gamma: float
) -> tuple[float, float]:
    """Image of a point of the sphere under the rotation.

    Used for the things that a rotated problem locates by a point: the rotation pole
    of the Coriolis parameter, the summit of a mountain.

    Args:
        longitude: Longitude of the point [rad].
        latitude: Latitude of the point [rad].
        alpha: Last rotation, about the z axis [rad].
        beta: Middle rotation, about the y axis [rad].
        gamma: First rotation, about the z axis [rad].

    Returns:
        Longitude in [0, 2 pi) and latitude of the rotated point [rad].
    """
    x = math.cos(latitude) * math.cos(longitude)
    y = math.cos(latitude) * math.sin(longitude)
    z = math.sin(latitude)

    # Rotation by gamma about z
    x, y = (
        math.cos(gamma) * x - math.sin(gamma) * y,
        math.sin(gamma) * x + math.cos(gamma) * y,
    )
    # Rotation by beta about y
    x, z = (
        math.cos(beta) * x + math.sin(beta) * z,
        -math.sin(beta) * x + math.cos(beta) * z,
    )
    # Rotation by alpha about z
    x, y = (
        math.cos(alpha) * x - math.sin(alpha) * y,
        math.sin(alpha) * x + math.cos(alpha) * y,
    )

    rotated_longitude = math.atan2(y, x) % (2 * math.pi)
    rotated_latitude = math.asin(max(-1.0, min(1.0, z)))
    return rotated_longitude, rotated_latitude
