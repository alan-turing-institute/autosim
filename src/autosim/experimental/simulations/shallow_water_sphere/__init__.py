r"""Shallow water equations on the rotating sphere, spectral method (torch-harmonics).

**State.** The state of ``B`` simulations is one complex tensor of shape
``(B, 3, L, L)``: index 0 is the geopotential :math:`\Phi = g h^*` of the fluid depth
:math:`h^*`, 1 the relative vorticity :math:`\zeta` and 2 the divergence
:math:`\delta`, all as spherical-harmonic coefficients with :math:`m \le l < L`. Over
an orography :math:`h_s` the free surface is :math:`h^* + h_s`. The grid is only used
to form the nonlinear products.

**Equations.**

.. math::

    \partial_t \Phi &= -\nabla\cdot(\Phi\,\mathbf{u}) + F_\Phi
        - (\Phi - \Phi_\text{eq}) / \tau_\text{relax}, \\
    \partial_t \zeta &= -\nabla\cdot\big((\zeta+f)\,\mathbf{u}\big) + F_\zeta
        - \zeta / \tau_\text{drag} + \text{hyperdiffusion}, \\
    \partial_t \delta &= \hat{\mathbf{k}}\cdot\nabla\times
        \big((\zeta+f)\,\mathbf{u}\big)
        - \nabla^2\big(\Phi + g h_s + \tfrac12|\mathbf{u}|^2\big) + F_\delta
        - \delta / \tau_\text{drag} + \text{hyperdiffusion}.

The Coriolis parameter :math:`f = 2\Omega\,(\hat{\mathbf{n}}\cdot\hat{\mathbf{r}})` is
a field on the grid, for a rotation axis :math:`\hat{\mathbf{n}}` of any direction.
Without orography, forcing, drag and relaxation, and with the geographic axis, the
tendencies equal torch-harmonics' ``ShallowWaterSolver.dudtspec``.

**Time step.** Third-order Adams-Bashforth on the conservative terms, with the linear
terms (hyperdiffusion, drag, relaxation) as an exact integrating factor. The forcing
:math:`F` is stochastic and is added after each step by operator splitting: its
increment is damped from the next step on and never enters the history of the
Adams-Bashforth scheme (module ``integration``).

**Forcing** (module ``forcing``). ``"vortical"`` is a streamfunction pattern
:math:`\psi'` whose coefficients follow the AR(1) process of Berner et al. (2009),
:math:`\psi'(t+\Delta t) = (1-\alpha)\,\psi'(t) + g_l\sqrt{\alpha}\,\epsilon(t)` with
:math:`g_l = b\,l^p` and :math:`\alpha = \Delta t/\tau`, and
:math:`F_\zeta\,\Delta t = \nabla^2\psi'`. ``"momentum"`` adds a velocity-potential
pattern, :math:`F_\delta\,\Delta t = \nabla^2\chi'`. ``"mass_pulses"`` adds Gaussian
bumps to :math:`\Phi` at Poisson times. The amplitude :math:`b` follows from the
energy rate :math:`\varepsilon` (kinetic energy per unit mass and time gained by a
free random walk of the increments): :math:`b^2 = 8\pi a^2 \alpha\,\varepsilon\,
\Delta t / \Gamma`, :math:`\Gamma = \sum_l l(l+1)(2l+1)\,l^{2p}`. A backscatter adds
a fraction of the kinetic energy lost to the hyperdiffusion (and to the drag, with
``backscatter_include_drag``) to :math:`\varepsilon`, as its global mean or as a
field on the grid (Berner et al., eq. 13).

**Latent forcing state.** The pattern has memory that the saved fields do not show.
Where a forecast branches (a restart, or the first saved frame of a generated run),
``branch_forcing_state="redraw"`` draws the pattern from its stationary distribution,
so the forecast distribution is :math:`p(x_{n+1} \mid x_n)`. ``"carry"`` keeps the
pattern, which ``return_forcing_state`` saves as an extra input: the oracle setting.

**Decisions.**

- *Layout.* One module per job (below), plain functions on tensors, and one class for
  the autosim interface. Families and forcings are selected by strings, as in the
  planar ``ShallowWater2D``, whose argument names are reused.
- *State with orography.* :math:`\Phi` is the geopotential of the fluid depth, so the
  mass equation is unchanged, and the saved ``h`` is the free surface above
  ``mean_depth``.
- *Energy norm.* The forcing is calibrated in the specific-energy norm of the planar
  simulator, the global mean of :math:`\tfrac12|\mathbf{u}|^2`. The energy budget
  uses the energy of the layer,
  :math:`\tfrac12 h^*|\mathbf{u}|^2 + \tfrac12 g (h - H)^2`.
- *Seeds.* Every random draw of a simulation comes from generators seeded from the
  sample seed and the row of the simulation, so it does not depend on the batch.
- *Rotation.* The scalar fields of the state are rotated with Wigner D-matrices,
  computed as matrix exponentials. The wind follows as a spin-1 field.
- *Aliasing.* Quadratic products are exact when ``2 nlat >= 3 lmax - 2``
  (``lmax = 86`` on 128 x 256), checked against a grid twice as fine.

Modules:

- ``dynamics``: operators, tendencies, time step, energy, conversion to and from
  saved fields;
- ``initial_conditions``: one function per family (Galewsky jet, W92 cases 2, 5, 6,
  random balanced flow);
- ``forcing``: stochastic kinetic-energy forcing, backscatter, mass pulses;
- ``integration``: the time loop of a batch and its energy budget;
- ``diagnostics``: spectra, invariants, W92 error norms, non-zonal energy;
- ``rotation``: rotation of states and saved frames by an element of SO(3);
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
from .rotation import (
    rotate_coefficients,
    rotate_fields,
    rotate_point,
    wigner_d_matrix,
)
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
    "random_balanced_initial_condition",
    "rossby_haurwitz_initial_condition",
    "rotate_coefficients",
    "rotate_fields",
    "rotate_point",
    "sample_generators",
    "sample_pulse_events",
    "shallow_water_tendencies",
    "spectral_to_fields",
    "stationary_pattern",
    "step_damping_factors",
    "total_energy",
    "unit_rate_amplitudes",
    "wigner_d_matrix",
    "williamson2_initial_condition",
    "wind_error_norms",
    "zonal_jet",
]
