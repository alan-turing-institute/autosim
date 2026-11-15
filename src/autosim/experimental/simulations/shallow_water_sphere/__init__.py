r"""Shallow water equations on the rotating sphere, spectral method (torch-harmonics).

The state of ``B`` simulations is one complex tensor of shape ``(B, 3, L, L)``:
index 0 is the geopotential :math:`\Phi = g h`, 1 the relative vorticity
:math:`\zeta` and 2 the divergence :math:`\delta`, all as spherical-harmonic
coefficients with :math:`m \le l < L`. The grid is only used to form the nonlinear
products:

.. math::

    \partial_t \Phi = -\nabla\cdot(\Phi\,\mathbf{u}), \qquad
    \partial_t \zeta = -\nabla\cdot\big((\zeta+f)\,\mathbf{u}\big), \qquad
    \partial_t \delta = \hat{\mathbf{k}}\cdot\nabla\times\big((\zeta+f)\,\mathbf{u}\big)
        - \nabla^2\big(\Phi + \tfrac12|\mathbf{u}|^2\big).

Time stepping is third-order Adams-Bashforth with the hyperdiffusion as an integrating
factor. The initial condition is the barotropically unstable jet of Galewsky, Scott and
Polvani (2004, Tellus A 56, 429-440), with sampled parameters and a small height
noise that seeds the chaos. The tendencies equal torch-harmonics'
``ShallowWaterSolver.dudtspec``.

Requires the optional ``spherical`` extra (torch-harmonics with grid descriptors).
"""

from .dynamics import (
    SphereShallowWaterDynamics,
    adams_bashforth_step,
    shallow_water_tendencies,
    spectral_to_fields,
)
from .initial_conditions import GALEWSKY_RANGES, galewsky_initial_condition
from .simulator import SphericalShallowWater

__all__ = [
    "GALEWSKY_RANGES",
    "SphereShallowWaterDynamics",
    "SphericalShallowWater",
    "adams_bashforth_step",
    "galewsky_initial_condition",
    "shallow_water_tendencies",
    "spectral_to_fields",
]
