autosim.experimental.simulations.shallow_water_sphere
=====================================================

Shallow water equations on the rotating sphere, solved with spherical harmonics
(torch-harmonics, the ``spherical`` extra). The package is the data generator of a
benchmark for probabilistic emulators on the sphere.

Initial conditions
------------------

``initial_condition`` selects a family, and ``parameters_range`` gives the bounds of
its parameters. Equal bounds at the published values give the published case.

=========================  ===================================================
``galewsky``               unstable jet with a height bump (Galewsky et al. 2004)
``mountain``               zonal flow over a Gaussian mountain (W92 case 5, with
                           the mountain of DCMIP-2008)
``era5``                   ERA5 at 500 hPa over the orography of the Earth,
                           balanced (W92 case 7; ``era5`` extra and
                           network access)
=========================  ===================================================

Each family exercises a different part of the dynamics. W92 case 2 is not used,
because a spectral model represents it exactly (Jakob et al. 1993). The Rossby-Haurwitz
wave of W92 case 6 is not used: it is not a solution of the shallow water equations and
contains no gravity waves (Shamir and Paldor 2016).
``galewsky`` has a gravity-wave adjustment and the roll-up of a shear instability.
``mountain`` replaces the cone of W92, whose truncated series rings, by a smooth
mountain. ``era5`` gives real atmospheric states. Its orography can be compressed as
in The Well's PlanetSWE, :math:`\operatorname{sign}(h_s)|h_s|^{15/16}` shifted to a
minimum of 0 (``era5=Era5Settings(orography_exponent=15/16, orography_shift=True)``).

Physics and forcing
-------------------

The Coriolis parameter is a grid field for a rotation axis of any direction
(``rotation_axis``). An orography enters through the free surface. With
``return_constant_fields=True`` both are returned in ``constant_fields``.

Each process is one argument of the simulator, a dataclass of the module ``settings``
in user units (hours, m/hour); ``None`` leaves it out. The processes can be used alone
or together:

- ``damping=DampingSettings(...)``: ``relaxation_time`` relaxes the height towards an
  equilibrium and ``drag_time`` damps the wind. With ``relaxation_target="jet"`` the
  equilibrium is the balanced height of a zonal jet, and the flow reaches a
  statistically steady state with a jet and eddies (Scott and Polvani 2007). With
  ``"era5_climatology"`` it is the ERA5 1990-2019 climatology of the start date's day
  and hour.
- ``skebs=SkebsSettings(...)``: the stochastic kinetic-energy backscatter (SKEBS) of
  Berner et al. (2009), a pattern in spherical harmonics whose coefficients are AR(1)
  processes, with ``energy_rate`` the kinetic energy injected per unit mass and time.
  ``degree_min`` and ``degree_max`` restrict the pattern to a band of degrees.
- ``global_backscatter=GlobalBackscatterSettings(fraction)`` and
  ``local_backscatter=LocalBackscatterSettings(fraction)`` add a fraction of the
  diagnosed dissipation to the SKEBS rate, as its global mean or as a field (B09 eq.
  13); the noise then depends on the flow. The two can be combined.
- ``sppt=SpptSettings(...)``: stochastically perturbed physics tendencies (Buizza et
  al. 1999; Palmer et al. 2009; Leutbecher et al. 2017). The increment of the drag and
  the relaxation is multiplied by ``1 + r``, with ``r`` a sum of AR(1) Gaussian fields,
  clipped. SKEBS is additive and acts on the wind; SPPT is multiplicative and acts
  mostly on the height. The two are complementary (Arnold et al. 2013; Christensen
  2020).
- ``periodic=PeriodicForcingSettings()``: the daily and yearly height source of The
  Well's PlanetSWE (Ohana et al. 2024), as in its generation code: the run is no
  longer autonomous. ``return_periodic_forcing=True`` returns the forcing at each saved
  frame, a known input of the emulator.

The fields of ``settings.SAMPLED_FIELDS`` can be sampled per simulation: their dotted
name, such as ``"skebs.energy_rate"`` or ``"damping.drag_time"``, is then a key of
``parameters_range``. The random patterns have memory: ``return_forcing_state``
returns the SKEBS pattern as an extra input, and ``return_additional_input_fields``
the increments added by all the forcings between frames.

Coarse-graining
---------------

``output_lmax`` saves the run truncated to a lower band limit, and
``return_baseline=True`` adds the run of the solver at that band limit from the
truncated first saved frame.

Datasets
--------

The configs ``shallow_water_sphere_ds1_*.yaml`` to ``ds6_*.yaml`` define the six
datasets of the benchmark, at L = 256 on the 384 x 768 Gauss-Legendre grid. Every run is
an independent trajectory, and every dataset is forced.

=========================  ==================================================  =================
Config                     Family and forcing                                  Length, saved
=========================  ==================================================  =================
``ds1_galewsky``           Galewsky jet, relaxation, drag, vortical, SPPT      10 days, 1 h
``ds2_galewsky_large``     Galewsky jet, relaxation, drag, vortical on         10 days, 1 h
                           degrees 1-20, no SPPT
``ds3_galewsky_small``     Galewsky jet, relaxation, drag, vortical on         10 days, 1 h
                           degrees 60-255, no SPPT
``ds4_mountain``           Gaussian mountain, local backscatter                15 days, 6 h
``ds5_era5_weather``       ERA5, relaxation to the climatology, drag,          10 days, 6 h
                           vortical, SPPT
``ds6_era5_climate``       ERA5, The Well's periodic forcing                   3 years, 1 h
=========================  ==================================================  =================

Dataset 6 follows PlanetSWE, with 40 trajectories (``dataset.n_train=32
dataset.n_valid=4 dataset.n_test=4``).

The notebook ``examples/experimental/01_05_spherical_shallow_water_benchmark.ipynb``
runs every family and forcing at a small size.

.. automodule:: autosim.experimental.simulations.shallow_water_sphere
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: autosim.experimental.simulations.shallow_water_sphere.simulator
   :members:
   :show-inheritance:

.. automodule:: autosim.experimental.simulations.shallow_water_sphere.dynamics
   :members:

.. automodule:: autosim.experimental.simulations.shallow_water_sphere.initial_conditions
   :members:

.. automodule:: autosim.experimental.simulations.shallow_water_sphere.settings
   :members:

.. automodule:: autosim.experimental.simulations.shallow_water_sphere.forcing
   :members:

.. automodule:: autosim.experimental.simulations.shallow_water_sphere.integration
   :members:

.. automodule:: autosim.experimental.simulations.shallow_water_sphere.diagnostics
   :members:

.. automodule:: autosim.experimental.simulations.shallow_water_sphere.rotation
   :members:
