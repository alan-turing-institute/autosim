autosim.experimental.simulations.shallow_water_1d
=================================================

Governing equations
-------------------

The one-dimensional shallow-water equations on a flat, periodic bed are

.. math::

   \begin{aligned}
   \partial_t h + \partial_x(hu) &= 0, \\
   \partial_t(hu)
       + \partial_x\!\left(hu^2+\frac{1}{2}gh^2\right) &= 0,
   \end{aligned}
   \qquad x\in[0,L),\quad t\in[0,T].

Here :math:`h(x,t)>0` is water depth, :math:`u(x,t)` is horizontal velocity,
:math:`g>0` is the gravity coefficient, and :math:`L` is the domain length.
The initial fields specify :math:`h(x,0)=h_0(x)` and
:math:`u(x,0)=u_0(x)`. Both fields are periodic in space. There is no Coriolis
term, forcing, drag, or explicit viscosity.

The first equation conserves water; the second conserves momentum per unit
density :math:`hu`. The simulator exposes ``[h, u]`` while evolving conserved
``[h, h*u]`` internally. For smooth, positive-depth solutions, an equivalent
velocity equation is

.. math::

   \partial_t u + u\partial_xu + g\partial_xh = 0.

The conservative equations above define the evolution across shocks.

Generate a dataset
------------------

Run the reference preset, which samples periodic Gaussian height bumps over
resting water and keeps gravity fixed at ``g=1``:

.. code-block:: console

   uv run autosim simulator=experimental/shallow_water_1d \
     dataset.n_train=4 dataset.n_valid=1 dataset.n_test=1 seed=7

For stronger nonlinear wave transients, this preset samples ``g`` from
``[0.5, 2.0]`` and height increments from ``[0.3, 0.8]`` above baseline depth
``1``, with pulse widths from ``[0.03, 0.08]`` of the domain length:

.. code-block:: console

   uv run autosim simulator=experimental/shallow_water_1d_nonlinear \
     dataset.n_train=4 dataset.n_valid=1 dataset.n_test=1 seed=7

Relative height increment controls the initial nonlinearity. At fixed height
field and zero initial velocity, changing gravity rescales time and velocity
by ``sqrt(g)``. These presets cover wave steepening and interactions; they do
not establish sustained chaos.

Each split's ``data.pt`` contains ``data`` shaped
``[batch, time, nx, 1, 2]``, with channels ``[h, u]`` and the actual initial
fields in the first frame. It also contains coordinates ``x`` and ``times``,
each with a batch axis, and ``constant_scalars`` containing gravity. With
visualization enabled, the CLI automatically writes space-time PNGs under
``examples/<split>/batch_<index>.png``.
See :doc:`../../tutorials/visualize` for output paths, parameter overrides,
and the initial-condition distributions.

Evolve explicit initial fields
------------------------------

Initial-field sampling is separate from deterministic evolution:

.. code-block:: python

   from autosim.experimental.simulations import ShallowWater1D
   from autosim.utils import plot_spatiotemporal_1d

   sim = ShallowWater1D(nx=256, T=1.0, dt_save=0.01, g=1.0)
   initial_state = sim.sample_initial_conditions(1, random_seed=7)
   trajectory = sim.rollout(initial_state, g=1.0)
   figure = plot_spatiotemporal_1d(
       trajectory,
       x=sim.x,
       times=sim.times,
       channel_names=sim.output_names,
       save_path="shallow_water.png",
   )

Replace ``initial_state`` with finite float32/float64 fields of shape
``(batch, nx, 2)`` in ``[h, u]`` order. Fields represent cell averages on
centers ``x_j=(j+0.5)*L/nx``; depth must be strictly positive. The result has
shape ``(batch, time, nx, 2)`` and includes the supplied initial fields. If
``parameters_range`` varies, ``rollout`` requires an explicit ``g``.

The solver uses minmod MUSCL reconstruction, Rusanov fluxes and SSP-RK2.
Its numerical dissipation and entropy-satisfying shocks can cause decay even
though explicit viscosity is absent. Dry states are rejected; wetting and
drying are unsupported. ``dt_save`` controls saved frames; ``dt_max`` and
``cfl`` control internal steps. Check spatial and temporal convergence when
changing pulse amplitude, width, gravity or grid resolution.

API reference
-------------

.. automodule:: autosim.experimental.simulations.shallow_water_1d
   :members:
   :undoc-members:
   :show-inheritance:
