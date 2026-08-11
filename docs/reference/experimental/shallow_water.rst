autosim.experimental.simulations.shallow_water
==============================================

Rotation choices
----------------

``f0=None`` retains the default reference Coriolis parameter
``sqrt(g * h_mean) / 8``. Set ``f0=0`` with ``coriolis_mode="f_plane"``
to disable rotation. If ``beta`` is also left unset, ``f0=0`` makes the
``periodic_beta`` profile zero as well. An explicitly nonzero ``beta`` still
produces spatially varying Coriolis acceleration in ``periodic_beta`` mode.

At ``f0=0``, ``balanced`` forcing has no height component and therefore
reduces to ``vortical`` forcing.

Forcing choices
---------------

``ShallowWater2D`` supports three stochastic forcing geometries in addition
to the deterministic ``none`` mode:

* ``vortical`` adds divergence-free velocity increments and is useful for
  unresolved rotational eddy stirring or wind-stress curl.
* ``balanced`` adds vortical velocity together with its constant-Coriolis
  geostrophic height perturbation. It is the preferred idealization for
  large-scale weather-like flow, although its balance is approximate on the
  periodic beta-plane.
* ``momentum`` adds unconstrained horizontal-velocity increments containing
  rotational and divergent components. It is the most direct idealization of
  stochastic wind stress for ocean-atmosphere coupling.

All modes use a configurable Gaussian ring in spatial Fourier space.
``forcing_energy_rate`` is a diffusion scale in the linearized SWE energy
norm. For white noise it sets the expected increment energy per unit time; for
OU forcing it sets the resolved long-time diffusion rate. It does not prescribe
the realized energy transferred to the nonlinear flow. Individual Gaussian
draws fluctuate around the target rather than being rescaled to identical
energy.

``forcing_wavenumber`` is an angular wavenumber. On a square domain of side
``L``, Fourier mode ``m`` has ``k = 2 * pi * m / L`` and wavelength ``L / m``.
For example, on the 64 by 64 example with ``L=64``, mode 4 has
``k=0.392699...`` and a 16-grid-point wavelength. Lower modes are useful for
coherent weather or wind-stress patterns; higher modes represent smaller-scale
eddy stirring. ``forcing_bandwidth`` controls how many neighbouring scales are
excited. Spectral forcing is a standard choice on this periodic FFT grid and
also makes the vortical constraint exact up to numerical precision.

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
rate when the correlation time is resolved by the adaptive timestep. Set
``return_additional_input_fields=True`` to store the realized ``[dh, du, dv]``
impulse accumulated over every saved transition.

OU forcing is a finite-persistence, Gaussian red-noise model rather than a
claim that real weather follows a single correlation time. First-order
autoregressive spectral coefficients, the discrete-time analogue of this OU
correlation, are used in atmospheric stochastic-backscatter schemes; see
`Berner et al. (2009) <https://doi.org/10.1175/2008JAS2677.1>`_ and the WRF
application by `Duda et al. (2016)
<https://doi.org/10.1175/MWR-D-15-0092.1>`_. Those schemes can scale forcing
with diagnosed dissipation; this simulator deliberately keeps a homogeneous,
fixed-amplitude ring for controlled benchmark experiments.

.. automodule:: autosim.experimental.simulations.shallow_water
   :members:
   :undoc-members:
   :show-inheritance:
