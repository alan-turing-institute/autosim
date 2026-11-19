r"""Diagnostics of the shallow water equations on the sphere.

Every function is pure: it takes tensors and returns tensors. Two kinds of inputs are
used.

Spectral inputs are spherical-harmonic coefficients of shape ``(..., L, L)``, indexed
``[degree, order]`` with orders :math:`m \ge 0` only (torch-harmonics, orthonormal
harmonics on the unit sphere, :math:`m = 0` coefficients real). A coefficient of order
:math:`m > 0` stands for the two orders :math:`\pm m`, so for a real field :math:`X`

.. math::

    \int X^2 \, d\Omega
        = \sum_{l} \Big( |\hat X_l^0|^2 + 2 \sum_{m > 0} |\hat X_l^m|^2 \Big).

Grid inputs are fields of shape ``(..., *grid.shape)`` with the quadrature weights of
the grid, ``dynamics.quad_weights``, which sum to :math:`4\pi`.

Saved frames ``[h, u_phi, u_theta]`` are converted to a spectral state with
``fields_to_spectral`` (module ``dynamics``).
"""

from __future__ import annotations

import math

import torch

from autosim.types import TensorLike

from .dynamics import SphereShallowWaterDynamics


def degree_power(coefficients: TensorLike) -> TensorLike:
    r"""Sum of :math:`|\hat X_l^m|^2` over all orders :math:`-l \le m \le l`.

    Args:
        coefficients: Coefficients (..., L, L) of a real field.

    Returns:
        One value per degree, shape (..., L). Its sum over the degrees is the integral
        of the squared field over the unit sphere.
    """
    lmax = coefficients.shape[-1]
    order_weights = torch.full(
        (lmax,), 2.0, dtype=torch.float64, device=coefficients.device
    )
    order_weights[0] = 1.0  # m = 0 is stored once, m > 0 stands for +m and -m
    return (coefficients.abs() ** 2 * order_weights).sum(dim=-1)


def height_power_spectrum(state: TensorLike, gravity: float) -> TensorLike:
    r"""Power of the layer depth :math:`\Phi / g` per degree.

    Args:
        state: Spectral state (B, 3, L, L).
        gravity: Gravitational acceleration [m/s^2].

    Returns:
        Power per degree [m^2], shape (B, L). The sum over the degrees is the global
        mean of the squared depth; degree 0 holds the squared mean depth.
    """
    depth = state[:, 0] / gravity
    return degree_power(depth) / (4 * math.pi)


def kinetic_energy_spectrum(state: TensorLike, radius: float) -> TensorLike:
    r"""Kinetic energy per unit mass and per degree, rotational and divergent parts.

    With :math:`\mathbf{u} = \hat{\mathbf{k}}\times\nabla\psi + \nabla\chi`, the global
    mean of :math:`\tfrac12|\mathbf{u}|^2` is

    .. math::

        \frac{1}{4\pi}\sum_{l \ge 1} \frac{a^2}{2\,l(l+1)}
            \sum_{m=-l}^{l} \big( |\hat\zeta_l^m|^2 + |\hat\delta_l^m|^2 \big).

    The vorticity term is the rotational part and the divergence term the divergent
    part. Their sum per degree equals :math:`\sum_m |{}_1 w_{lm}|^2` of the spin-1
    field :math:`u + i v`, up to the same normalisation.

    Args:
        state: Spectral state (B, 3, L, L).
        radius: Planet radius a [m].

    Returns:
        Energy per degree [m^2/s^2], shape (B, 2, L): index 0 is the rotational part,
        1 the divergent part. The sum over both parts and all degrees is the global
        mean of :math:`\tfrac12|\mathbf{u}|^2`.
    """
    lmax = state.shape[-1]
    degree = torch.arange(lmax, dtype=torch.float64, device=state.device)
    factor = torch.zeros_like(degree)
    factor[1:] = radius**2 / (8 * math.pi * degree[1:] * (degree[1:] + 1))
    rotational = factor * degree_power(state[:, 1])
    divergent = factor * degree_power(state[:, 2])
    return torch.stack([rotational, divergent], dim=1)


def non_zonal_energy(
    state: TensorLike, radius: float, gravity: float, mean_depth: float
) -> TensorLike:
    r"""Energy per unit mass in the non-zonal coefficients (orders :math:`m \ne 0`).

    The energy is the linearised one, the global mean of
    :math:`\tfrac12|\mathbf{u}|^2 + \tfrac12 (g/H)\,h^2`. A zonally symmetric flow has
    zero non-zonal energy, so this measures how far a balanced zonal jet has left its
    initial symmetry.

    Args:
        state: Spectral state (B, 3, L, L).
        radius: Planet radius a [m].
        gravity: Gravitational acceleration g [m/s^2].
        mean_depth: Mean depth H [m].

    Returns:
        Non-zonal energy [m^2/s^2], shape (B,).
    """
    lmax = state.shape[-1]
    degree = torch.arange(lmax, dtype=torch.float64, device=state.device)
    kinetic_factor = torch.zeros_like(degree)
    kinetic_factor[1:] = radius**2 / (8 * math.pi * degree[1:] * (degree[1:] + 1))
    potential_factor = 1.0 / (8 * math.pi * gravity * mean_depth)

    # Orders m > 0 only; each stands for +m and -m, hence the factor 2
    non_zonal = 2 * state[:, :, :, 1:].abs() ** 2  # (B, 3, L, L - 1)
    power = non_zonal.sum(dim=-1)  # (B, 3, L): per field and degree
    kinetic = (kinetic_factor * (power[:, 1] + power[:, 2])).sum(dim=-1)
    potential = potential_factor * power[:, 0].sum(dim=-1)
    return kinetic + potential


def global_integral(field: TensorLike, quad_weights: TensorLike) -> TensorLike:
    r"""Integral over the unit sphere, the operator :math:`I` of W92 eq. (81).

    Args:
        field: Grid values (..., *grid.shape).
        quad_weights: Quadrature weights of the grid, shape grid.shape, summing to
            :math:`4\pi`.

    Returns:
        The integral, shape (...).
    """
    grid_dims = tuple(range(-quad_weights.ndim, 0))
    return (field * quad_weights).sum(dim=grid_dims)


def height_error_norms(
    height: TensorLike, height_true: TensorLike, quad_weights: TensorLike
) -> tuple[TensorLike, TensorLike, TensorLike]:
    r"""Normalised :math:`l_1`, :math:`l_2` and :math:`l_\infty` errors of the height.

    W92 eqs. (82)-(84):
    :math:`l_1 = I[|h - h_T|] / I[|h_T|]`,
    :math:`l_2 = \sqrt{I[(h - h_T)^2]} / \sqrt{I[h_T^2]}` and
    :math:`l_\infty = \max|h - h_T| / \max|h_T|`.

    Args:
        height: Height (..., *grid.shape).
        height_true: Reference height, same shape.
        quad_weights: Quadrature weights of the grid, shape grid.shape.

    Returns:
        (l1, l2, linf), each of shape (...).
    """
    grid_dims = tuple(range(-quad_weights.ndim, 0))
    error = height - height_true
    l1 = global_integral(error.abs(), quad_weights) / global_integral(
        height_true.abs(), quad_weights
    )
    l2 = torch.sqrt(global_integral(error**2, quad_weights)) / torch.sqrt(
        global_integral(height_true**2, quad_weights)
    )
    linf = error.abs().amax(dim=grid_dims) / height_true.abs().amax(dim=grid_dims)
    return l1, l2, linf


def wind_error_norms(
    wind: TensorLike, wind_true: TensorLike, quad_weights: TensorLike
) -> tuple[TensorLike, TensorLike, TensorLike]:
    r"""Normalised :math:`l_1`, :math:`l_2` and :math:`l_\infty` errors of the wind.

    W92 eqs. (97)-(99), with the length of the error vector
    :math:`e = \sqrt{(u - u_T)^2 + (v - v_T)^2}` and of the reference wind
    :math:`s_T = \sqrt{u_T^2 + v_T^2}`:
    :math:`l_1 = I[e] / I[s_T]`, :math:`l_2 = \sqrt{I[e^2]} / \sqrt{I[s_T^2]}` and
    :math:`l_\infty = \max e / \max s_T`. The lengths are those of vectors, so the
    norms are the same in any rotated frame.

    Args:
        wind: Wind (..., 2, *grid.shape), any two orthogonal components.
        wind_true: Reference wind, same shape and components.
        quad_weights: Quadrature weights of the grid, shape grid.shape.

    Returns:
        (l1, l2, linf), each of shape (...).
    """
    grid_dims = tuple(range(-quad_weights.ndim, 0))
    component_dim = -quad_weights.ndim - 1
    error_squared = ((wind - wind_true) ** 2).sum(dim=component_dim)
    true_squared = (wind_true**2).sum(dim=component_dim)
    l1 = global_integral(torch.sqrt(error_squared), quad_weights) / global_integral(
        torch.sqrt(true_squared), quad_weights
    )
    l2 = torch.sqrt(global_integral(error_squared, quad_weights)) / torch.sqrt(
        global_integral(true_squared, quad_weights)
    )
    linf = torch.sqrt(error_squared.amax(dim=grid_dims)) / torch.sqrt(
        true_squared.amax(dim=grid_dims)
    )
    return l1, l2, linf


def invariants(
    dynamics: SphereShallowWaterDynamics,
    state: TensorLike,
    coriolis: TensorLike | None = None,
    surface_geopotential: TensorLike | None = None,
) -> TensorLike:
    r"""Mass, total energy and potential enstrophy, integrated over the unit sphere.

    With the fluid depth :math:`h^* = \Phi / g`, the orography :math:`h_s` and the free
    surface :math:`h = h^* + h_s`, the three integrands are

    .. math::

        h^*, \qquad
        \tfrac12 h^* |\mathbf{u}|^2 + \tfrac12 g\,(h^2 - h_s^2), \qquad
        \frac{(\zeta + f)^2}{2 h^*}.

    These are the quantities conserved by the equations. The ORNL print of W92 eq.
    (146) reads :math:`h\,(g h + \mathbf{v}\cdot\mathbf{v}/2)`, which the equations do
    not conserve, so the conserved energy is used here. W92 eq. (144) normalises each
    invariant as :math:`(I(t) - I(0)) / I(0)`, its relative change since time 0.

    Args:
        dynamics: Fixed operators.
        state: Spectral state (B, 3, L, L).
        coriolis: Coriolis parameter (B, *grid.shape); None is the polar one.
        surface_geopotential: Coefficients of :math:`g h_s`, (B, L, L); None is a flat
            bottom.

    Returns:
        Tensor (B, 3): mass [m], energy [m^3/s^2] and potential enstrophy [1/(m s^2)].
    """
    gravity = dynamics.gravity
    if coriolis is None:
        coriolis = dynamics.coriolis

    depth = dynamics.to_grid(state[:, 0]) / gravity
    vorticity = dynamics.to_grid(state[:, 1])
    wind = dynamics.wind(state[:, 1], state[:, 2])
    speed_squared = wind[:, 0] ** 2 + wind[:, 1] ** 2

    if surface_geopotential is None:
        surface_height = torch.zeros_like(depth)
    else:
        surface_height = dynamics.to_grid(surface_geopotential) / gravity
    free_surface = depth + surface_height

    mass = depth
    energy = 0.5 * depth * speed_squared + 0.5 * gravity * (
        free_surface**2 - surface_height**2
    )
    potential_enstrophy = (vorticity + coriolis) ** 2 / (2 * depth)

    quad_weights = dynamics.quad_weights
    return torch.stack(
        [
            global_integral(mass, quad_weights),
            global_integral(energy, quad_weights),
            global_integral(potential_enstrophy, quad_weights),
        ],
        dim=1,
    )
