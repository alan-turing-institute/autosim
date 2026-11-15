r"""Shallow water equations on the rotating sphere, spectral method (torch-harmonics).

The state of ``B`` simulations is one complex tensor of shape ``(B, 3, L, L)``:
index 0 is the geopotential :math:`\Phi = g h^*` of the fluid depth :math:`h^*`, 1 the
relative vorticity :math:`\zeta` and 2 the divergence :math:`\delta`, all as
spherical-harmonic coefficients with :math:`m \le l < L`. Over an orography
:math:`h_s` the free surface is :math:`h^* + h_s`. The grid is only used to form the
nonlinear products:

.. math::

    \partial_t \Phi = -\nabla\cdot(\Phi\,\mathbf{u}), \qquad
    \partial_t \zeta = -\nabla\cdot\big((\zeta+f)\,\mathbf{u}\big), \qquad
    \partial_t \delta = \hat{\mathbf{k}}\cdot\nabla\times\big((\zeta+f)\,\mathbf{u}\big)
        - \nabla^2\big(\Phi + g h_s + \tfrac12|\mathbf{u}|^2\big).

The Coriolis parameter :math:`f = 2\Omega\,(\hat{\mathbf{n}}\cdot\hat{\mathbf{r}})` is
a field on the grid, for a rotation axis :math:`\hat{\mathbf{n}}` of any direction.
Time stepping is third-order Adams-Bashforth with the linear damping as an
integrating factor: the hyperdiffusion, and on request a linear drag on
:math:`\zeta` and :math:`\delta` and a Newtonian relaxation of :math:`\Phi` towards an
equilibrium. Without orography and with the geographic axis the tendencies equal
torch-harmonics' ``ShallowWaterSolver.dudtspec``. A stochastic forcing can be added
after each step by operator splitting (modules ``forcing`` and ``integration``).

Modules:

- ``dynamics``: operators, tendencies, time step, energy, conversion to and from
  saved fields;
- ``initial_conditions``: one function per family (Galewsky jet, W92 cases 2, 5, 6,
  random balanced flow);
- ``forcing``: stochastic kinetic-energy forcing, backscatter, mass pulses;
- ``integration``: the time loop of a batch and its energy budget;
- ``diagnostics``: spectra, invariants, W92 error norms, non-zonal energy;
- ``simulator``: ``SphericalShallowWater``, the autosim interface.

References: Galewsky, Scott and Polvani (2004, Tellus A 56, 429-440); Williamson,
Drake, Hack, Jakob and Swarztrauber (1992, J. Comput. Phys. 102, 211-224), "W92";
Berner, Shutts, Leutbecher and Palmer (2009, J. Atmos. Sci. 66, 603-626), "B09".

Requires the optional ``spherical`` extra (torch-harmonics with grid descriptors).
"""

from .diagnostics import (
    degree_power,
    global_integral,
    height_error_norms,
    height_power_spectrum,
    invariants,
    kinetic_energy_spectrum,
    non_zonal_energy,
    normalised_invariants,
    wind_error_norms,
)
from .dynamics import (
    SphereShallowWaterDynamics,
    adams_bashforth_step,
    coriolis_parameter,
    fields_to_spectral,
    increment_to_fields,
    make_grid,
    shallow_water_tendencies,
    spectral_to_fields,
    step_damping_factors,
    total_energy,
)
from .forcing import (
    ForcingSettings,
    advance_pattern,
    autoregressive_parameter,
    complex_noise,
    dissipation_taper,
    global_dissipation_rate,
    kinetic_forcing_increment,
    local_dissipation_rate,
    mass_pulse_increment,
    sample_pulse_events,
    stationary_pattern,
    unit_rate_amplitudes,
)
from .initial_conditions import (
    GALEWSKY_RANGES,
    GALEWSKY_UNPERTURBED_RANGES,
    MOUNTAIN_RANGES,
    RANDOM_BALANCED_RANGES,
    ROSSBY_HAURWITZ_RANGES,
    WILLIAMSON2_RANGES,
    balanced_geopotential,
    galewsky_initial_condition,
    mountain_initial_condition,
    mountain_orography,
    random_balanced_initial_condition,
    rossby_haurwitz_initial_condition,
    williamson2_initial_condition,
    zonal_jet,
)
from .integration import ENERGY_BUDGET_NAMES, integrate_trajectories
from .simulator import (
    FORCING_TYPES,
    INITIAL_CONDITIONS,
    SAMPLED_SCALARS,
    SphericalShallowWater,
    sample_generators,
)

__all__ = [
    "ENERGY_BUDGET_NAMES",
    "FORCING_TYPES",
    "GALEWSKY_RANGES",
    "GALEWSKY_UNPERTURBED_RANGES",
    "INITIAL_CONDITIONS",
    "MOUNTAIN_RANGES",
    "RANDOM_BALANCED_RANGES",
    "ROSSBY_HAURWITZ_RANGES",
    "SAMPLED_SCALARS",
    "WILLIAMSON2_RANGES",
    "ForcingSettings",
    "SphereShallowWaterDynamics",
    "SphericalShallowWater",
    "adams_bashforth_step",
    "advance_pattern",
    "autoregressive_parameter",
    "balanced_geopotential",
    "complex_noise",
    "coriolis_parameter",
    "degree_power",
    "dissipation_taper",
    "fields_to_spectral",
    "galewsky_initial_condition",
    "global_dissipation_rate",
    "global_integral",
    "height_error_norms",
    "height_power_spectrum",
    "increment_to_fields",
    "integrate_trajectories",
    "invariants",
    "kinetic_energy_spectrum",
    "kinetic_forcing_increment",
    "local_dissipation_rate",
    "make_grid",
    "mass_pulse_increment",
    "mountain_initial_condition",
    "mountain_orography",
    "non_zonal_energy",
    "normalised_invariants",
    "random_balanced_initial_condition",
    "rossby_haurwitz_initial_condition",
    "sample_generators",
    "sample_pulse_events",
    "shallow_water_tendencies",
    "spectral_to_fields",
    "stationary_pattern",
    "step_damping_factors",
    "total_energy",
    "unit_rate_amplitudes",
    "williamson2_initial_condition",
    "wind_error_norms",
    "zonal_jet",
]
