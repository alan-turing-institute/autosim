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
``williamson2``            steady flow about a tilted axis (W92 case 2)
``rossby_haurwitz``        Rossby-Haurwitz wave (W92 case 6)
``mountain``               zonal flow over an isolated mountain (W92 case 5)
``random_balanced``        random flow with its balanced height
``restart``                a supplied state: branching and ensembles
``from_fields``            supplied fields after a balance step (W92 case 7)
=========================  ===================================================

Physics
-------

The Coriolis parameter is a grid field for a rotation axis of any direction
(``rotation_axis``). An orography enters through the free surface. With
``return_constant_fields=True`` both are returned in ``constant_fields``.
``relaxation_time`` relaxes the height towards the balanced height of a zonal jet and
``drag_time`` damps the wind; together they give a statistically steady flow with a
jet and eddies.

Stochastic forcing
------------------

``forcing_type`` is ``"none"``, ``"vortical"``, ``"momentum"`` or ``"mass_pulses"``.
The first two follow Berner et al. (2009): a pattern in spherical harmonics whose
coefficients are AR(1) processes, with ``forcing_energy_rate`` the kinetic energy
injected per unit mass and time. ``forcing_backscatter_fraction`` adds a fraction of
the diagnosed dissipation to that rate, as its global mean or, with
``forcing_backscatter_mode="local"``, as a field: the noise then depends on the flow.

The pattern has memory. ``branch_forcing_state="redraw"`` draws it anew where a
forecast branches, so the forecast depends on the saved fields alone. ``"carry"``
keeps it, and ``return_forcing_state=True`` returns it as an extra input.

Ensembles and coarse-graining
-----------------------------

A restart with ``n`` simulations gives ``n`` members from one state.
``reference_members`` branches members from the first saved frame of every generated
simulation. ``output_lmax`` saves the run truncated to a lower band limit, and
``return_baseline=True`` adds the run of the solver at that band limit.

Datasets
--------

The configs ``shallow_water_sphere_d*.yaml`` and ``shallow_water_sphere_t*.yaml``
define the datasets of the benchmark:

.. code-block:: console

   uv run autosim --config-name generate_data_swe_sphere simulator=experimental/shallow_water_sphere_d1_galewsky
   uv run autosim --config-name generate_data_swe_sphere_t1

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

.. automodule:: autosim.experimental.simulations.shallow_water_sphere.forcing
   :members:

.. automodule:: autosim.experimental.simulations.shallow_water_sphere.integration
   :members:

.. automodule:: autosim.experimental.simulations.shallow_water_sphere.diagnostics
   :members:

.. automodule:: autosim.experimental.simulations.shallow_water_sphere.rotation
   :members:
