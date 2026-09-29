autosim.experimental.simulations.shallow_water
==============================================

Rotation choices
----------------

``f0=None`` retains the default reference Coriolis parameter
``sqrt(g * h_mean) / 8``. Set ``f0=0`` with ``coriolis_mode="f_plane"``
to disable rotation. If ``beta`` is also left unset, ``f0=0`` makes the
``periodic_beta`` profile zero as well. An explicitly nonzero ``beta`` still
produces spatially varying Coriolis acceleration in ``periodic_beta`` mode.

At ``f0=0``, ``balanced`` and ``pv_balanced`` forcing have no height
component and reduce to ``vortical`` forcing.

Initial conditions and restarts
-------------------------------

``initial_condition="random"`` retains the original randomized jet-like
state. ``"balanced_random_pv"`` samples an isotropic Gaussian ring of
potential-vorticity modes, applies deformation-radius-aware inversion, and
normalizes the RMS speed to ``amp``. ``initial_wavenumber`` and
``initial_bandwidth`` control its central scale and spectral width; their
defaults prefer modes 4 and 1.5 on the longest domain side. On a small grid
that cannot retain mode 4 isotropically, the default centre is clamped to the
largest isotropically retained mode.
Either value may instead be included in ``parameters_range``. It is then
sampled independently for every trajectory and returned in
``constant_scalars``, allowing emulator datasets to span multiple resolved
eddy scales.

``"balanced_double_jet"`` creates a smooth, periodic zonal double jet with
zero net transport and a small configurable wave perturbation. Both new
generated states construct height in constant-``f`` geostrophic balance,
which is approximate when evolved with ``periodic_beta``. ``"restart"``
accepts a finite ``[nx, ny, 3]`` tensor in ``[h, u, v]`` order, which is useful
for branching deterministic and stochastic runs from exactly the same
spun-up physical state. A restart does not preserve the latent OU forcing
tendency, so correlated forcing is initialized anew rather than continuing an
interrupted stochastic path exactly. Because the supplied state already fixes
its amplitude, restart datasets omit ``amp`` from ``parameters_range``.

Differentiable forcing-free forecasts
-------------------------------------

``autosim.experimental.simulations.advance_swe_2d`` advances physical
``[h, u, v]`` fields shaped ``(..., nx, ny, 3)`` without stochastic forcing.
It shares the generator's deterministic RHS, RK4 stages, spectral projection
and hyperviscosity, but preserves the input's float32/float64 precision,
device and autograd graph. Leading batch and ensemble axes evolve independently.
With ``dealias=True``, both interfaces project each RK-stage state onto the
retained Fourier band before evaluating nonlinear products. This prevents
unresolved modes in restarts or learned corrections from aliasing into resolved
modes. The differentiable interface validates both the raw and projected
states; a projection that creates nonphysical values raises an error. A zero
forecast interval remains an exact identity, without projection.

Supply the dataset's physical parameters explicitly and choose a fixed
``n_substeps`` by convergence testing. For example, for a valid 32 by 32 state
on a periodic domain with side length ``2*pi``:

.. code-block:: python

   import math
   from autosim.experimental.simulations import advance_swe_2d

   forecast = advance_swe_2d(
       state, 0.25, n_substeps=64,
       Lx=2 * math.pi, Ly=2 * math.pi,
       g=9.81, h_mean=1.0, nu=1e-4, drag=0.05,
       f0=1.0, beta=0.5, coriolis_mode="periodic_beta",
   )

Do not pass normalized model channels directly: convert to physical units
before the solve and convert back afterwards. Physics parameters and the
fixed time schedule are Python scalars, not learnable tensors. Gradients
propagate through the incoming state, including when it contains a learned
correction from an earlier forecast. An additive residual model can therefore
use ``advance_swe_2d(state, ...) + learned_correction`` and backpropagate a
later rollout loss through the intervening physics solves. This correction
represents the net finite-interval residual, not an instantaneous forcing.

The interface rejects non-finite states, states needing the generator's
clipping, and steps exceeding the wave/advection CFL or the combined linear
viscosity/drag/Coriolis bound. The generator caps its adaptive steps with the
same linear bound, including the maximum absolute periodic Coriolis value.
The differentiable interface does not silently detach, sanitize or adapt a
learned state. These checks
are safeguards, not a general stability guarantee; their control flow is not
differentiated and currently synchronizes on CUDA. Fixed-step and adaptive
forecasts need not be bitwise identical over a full forecast interval.
Long differentiable rollouts retain intermediate FFT graphs, so measure memory
and runtime before training. ``torch.compile`` and ``vmap`` are not currently
supported contracts. No noise or data-generation settings change automatically.

.. autofunction:: autosim.experimental.simulations.shallow_water_stepper.advance_swe_2d

Zero-gravity limit
------------------

Setting ``g=0`` removes height-gradient feedback from the momentum equations
and initializes ``h`` uniformly. With ``f0=beta=0``, velocity follows forced,
damped two-dimensional vector-Burgers dynamics:

.. math::

   \partial_t \mathbf{u} + \mathbf{u}\cdot\nabla\mathbf{u}
   = \nu\nabla^2\mathbf{u} - r\mathbf{u} + \mathbf{F}.

The continuity equation is still integrated, so ``h`` is a passive field rather
than a constant. This limit is not incompressible Navier--Stokes: even when
``vortical`` forcing is divergence-free, nonlinear evolution can generate
velocity divergence. Smaller initial amplitudes and stronger viscosity and drag
than the gravity-supported defaults help prevent compressive steepening.

Forcing choices
---------------

``ShallowWater2D`` supports four stochastic forcing geometries in addition
to the deterministic ``none`` mode:

* ``vortical`` adds divergence-free velocity increments and is useful for
  unresolved rotational eddy stirring or wind-stress curl. It is the closest
  option to spectral stochastic kinetic-energy backscatter schemes.
* ``balanced`` adds vortical velocity together with its constant-Coriolis
  geostrophic height perturbation. This experimental joint perturbation can
  reduce immediate imbalance, although its balance is approximate on the
  periodic beta-plane.
* ``pv_balanced`` samples a potential-vorticity anomaly, inverts the
  deformation-radius Helmholtz operator, and constructs geostrophically
  balanced velocity and height increments. The inversion is physically
  motivated, but its repeated use as additive forcing is an experimental
  construction rather than a standard atmospheric backscatter scheme.
* ``momentum`` adds unconstrained horizontal-velocity increments containing
  rotational and divergent components. It is the most direct idealization of
  stochastic wind stress for ocean-atmosphere coupling.

All modes use a configurable Gaussian ring in spatial Fourier space. For
``vortical`` and ``balanced`` forcing, the ring filters a sampled vorticity
field before streamfunction inversion; for ``pv_balanced`` it filters a
sampled potential-vorticity field before Helmholtz inversion; and for
``momentum`` it filters two sampled velocity components directly.
``forcing_energy_rate`` is a diffusion scale in the linearized SWE energy
norm. For white noise it sets the expected increment energy per unit time; for
OU forcing it sets the long-time diffusion rate. It does not prescribe the
realized energy transferred to the nonlinear flow. Individual Gaussian
draws fluctuate around the target rather than being rescaled to identical
energy. Neither the total flow energy nor the energy of each forcing
realization is held constant.

``forcing_wavenumber`` is an angular wavenumber. On a square domain of side
``L``, Fourier mode ``m`` has ``k = 2 * pi * m / L`` and wavelength ``L / m``.
For example, on the 64 by 64 example with ``L=64``, mode 4 has
``k=0.392699...`` and a 16-grid-point wavelength. Lower modes are useful for
coherent weather or wind-stress patterns; higher modes represent smaller-scale
eddy stirring. ``forcing_bandwidth`` controls how many neighbouring scales are
excited. Spectral forcing is a standard choice on this periodic FFT grid and
also makes the vortical constraint exact up to numerical precision.
The ring centre must fit inside the largest circular band retained in every
Fourier direction. Internally selected defaults are clamped to that band on
small grids. Fixed values and sampled ranges that place the peak only in the
rectangular mask's diagonal corners are rejected before generation.

The Fourier ring is an idealized homogeneous, direction-neutral covariance:
it controls the injected scale, not the location of storms, coastlines, or
other spatially varying sources. Spectral stochastic streamfunction patterns
have direct weather-model precedent; see `Berner et al. (2009)
<https://doi.org/10.1175/2008JAS2677.1>`_ for their use in the ECMWF ensemble
prediction system.

``forcing_correlation_time=0`` selects independent white-in-time increments.
A positive value selects an Ornstein-Uhlenbeck forcing tendency with that
e-folding time in simulator time units. Increasing it produces longer-lived
temporal correlations without changing the target long-time energy diffusion
rate. Conditional on the incoming tendency and the effective diffusion rate
held over a step, the OU time integral is sampled exactly. It converges to the
white-noise increment as the correlation time tends to zero.
Set ``return_additional_input_fields=True`` to store the realized
``[dh, du, dv]`` impulse accumulated over every saved transition as an opt-in
diagnostic. It is not included in the default forced dataset or its
normalization statistics.

Set ``forcing_backscatter_fraction`` above zero to add that fraction of the
diagnosed Laplacian-viscosity and exact numerical hyperviscosity loss to the
forcing diffusion rate. Depth-weighted loss diagnostics are divided by
``h_mean`` to match the forcing's specific-energy convention. The default zero
keeps forcing fixed. Linear-drag loss is excluded because drag normally
represents a physical large-scale sink; set
``backscatter_include_drag=True`` to include it for controlled experiments.
This is an idealized energy calibration, not a closure derived from the
resolved flow.

OU forcing is a finite-persistence, Gaussian red-noise model rather than a
claim that real weather follows a single correlation time. First-order
autoregressive spectral coefficients, the discrete-time analogue of this OU
correlation, are used in atmospheric stochastic-backscatter schemes; see
`Berner et al. (2009) <https://doi.org/10.1175/2008JAS2677.1>`_ and the WRF
application by `Duda et al. (2016)
<https://doi.org/10.1175/MWR-D-15-0092.1>`_.

Stochastic height increments that leave the generator's output bounds cause
an error, even when only one grid cell is affected. Reduce the forcing strength
or adjust the initial state rather than relying on clipping to repair such a
trajectory: clipping would alter mass and the saved state's energy budget.

Energy diagnostics
------------------

``return_energy_budget=True`` records total shallow-water energy and the exact
energy changes across the deterministic RK4 step, spectral hyperviscosity,
and stochastic increment for each saved transition. Their sum closes the
recorded total-energy change up to floating-point precision. Positive
viscosity and drag loss estimates accumulated over the saved transition and
the interval-mean effective forcing diffusion rate are also returned. These
estimates explain the source used by dissipation-linked forcing but are not
extra terms in the exact closure.

CRPS spatial-coherence dataset presets
---------------------------------------

The ``shallow_water2d_crps_32`` and ``shallow_water2d_crps_64`` simulator
presets generate independent trajectories with ``balanced_random_pv`` initial
conditions and a coherent vortical forcing ring centred on Fourier mode 3.
All physical parameters are fixed within each dataset, including ``amp=0.1``.

The forcing is white in time (``forcing_correlation_time=0``), so the returned
``[h, u, v]`` fields form the complete state for a one-step conditional model.
After spin-up, each trajectory contains 128 states from time 40.0 through 71.75
at intervals of 0.25. Downstream data loaders can extract adjacent one-step
pairs while retaining the complete trajectory for rollout evaluation.

Run a small 32 by 32 pilot before generating either full dataset:

.. code-block:: console

   uv run autosim --config-name=generate_data_swe_crps_32 \
     dataset.n_train=8 dataset.n_valid=2 dataset.n_test=2

The production commands are:

.. code-block:: console

   uv run autosim --config-name=generate_data_swe_crps_32
   uv run autosim --config-name=generate_data_swe_crps_64

Both resolutions use 64/8/8 trajectories for the train/validation/test splits.
Generation is serial, so the 64 by 64 dataset should only be started after
inspecting the 32 by 32 output. ``forcing_energy_rate`` controls the forcing
diffusion scale; it should not be interpreted as a direct target for the ratio
of stochastic to deterministic forecast spread. Nor should these forced,
dissipative presets be expected to exhibit a ``k^-3`` inertial range.

The ``shallow_water2d_crps_deterministic_32`` preset provides a matched
deterministic control. It keeps the 32 by 32 physical parameters, initial-state
distribution, sampling interval, trajectory length, split sizes, and seed, but
sets ``forcing_type=none``. An unforced trajectory decays under the retained
drag and viscosity, so the control uses a shorter spin-up and returns 128 states
from time 5.0 through 36.75; the original time-40 sampling window would be
almost static. Generate it with:

.. code-block:: console

   uv run autosim --config-name=generate_data_swe_crps_deterministic_32

.. automodule:: autosim.experimental.simulations.shallow_water
   :members:
   :undoc-members:
   :show-inheritance:
