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
forcing diffusion rate. The default zero keeps forcing fixed. Linear-drag loss
is excluded because drag normally represents a physical large-scale sink; set
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

.. automodule:: autosim.experimental.simulations.shallow_water
   :members:
   :undoc-members:
   :show-inheritance:
