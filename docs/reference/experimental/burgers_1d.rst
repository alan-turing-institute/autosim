autosim.experimental.simulations.burgers_1d
===========================================

Governing equation
------------------

The periodic viscous Burgers equation is

.. math::

   \partial_t u + \partial_x\!\left(\frac{u^2}{2}\right)
       = \nu\,\partial_{xx}u,
   \qquad x\in[0,L),\quad t\in[0,T].

Here :math:`u(x,t)` is velocity, :math:`\nu>0` is kinematic viscosity,
and :math:`L` is the domain length. The supplied initial field defines
:math:`u(x,0)=u_0(x)`; the field and its derivatives are periodic in space.
For smooth fields the advection term is :math:`u\partial_xu`, so velocity
transports itself. There is no forcing or process noise. Unforced trajectories
eventually approach their constant spatial mean.

Generate a dataset
------------------

Run the reference preset, which uses Gaussian random initial velocity fields
and fixed viscosity ``nu=0.1``:

.. code-block:: console

   uv run autosim simulator=experimental/burgers_1d \
     dataset.n_train=4 dataset.n_valid=1 dataset.n_test=1 seed=7

For stronger nonlinear transients, the following preset samples viscosity
from ``[0.005, 0.02]`` and eight-mode Fourier initial fields with peak speeds
in ``[0.5, 1.0]``:

.. code-block:: console

   uv run autosim simulator=experimental/burgers_1d_nonlinear \
     dataset.n_train=4 dataset.n_valid=1 dataset.n_test=1 seed=7

Each split's ``data.pt`` contains ``data`` shaped
``[batch, time, nx, 1, 1]``, with channel ``u`` and the actual initial field
in the first frame. It also contains coordinates ``x`` and ``times``, each
with a batch axis, and ``constant_scalars`` containing viscosity. With
visualization enabled, the CLI automatically writes space-time PNGs under
``examples/<split>/batch_<index>.png``.
See :doc:`../../tutorials/visualize` for output paths, parameter overrides,
and the initial-condition distributions.

Evolve an explicit initial field
--------------------------------

Initial-field sampling is separate from evolution. This example samples a
field once and then advances it deterministically at the supplied viscosity:

.. code-block:: python

   from autosim.experimental.simulations import Burgers1D
   from autosim.utils import plot_spatiotemporal_1d

   sim = Burgers1D(nx=256, T=1.0, dt_save=0.01, nu=0.01)
   initial_state = sim.sample_initial_conditions(1, random_seed=7)
   trajectory = sim.rollout(initial_state, nu=0.01)
   figure = plot_spatiotemporal_1d(
       trajectory,
       x=sim.x,
       times=sim.times,
       channel_names=sim.output_names,
       save_path="burgers.png",
   )

Replace ``initial_state`` with any finite float32/float64 field of shape
``(batch, nx, 1)`` on nodes ``x_j=j*L/nx``. The result has shape
``(batch, time, nx, 1)`` and includes the supplied initial state. If
``parameters_range`` varies, ``rollout`` requires an explicit ``nu``.

The solver uses exact Fourier diffusion half-steps around a dealiased RK4
advection step, with second-order Strang splitting. ``dt_save`` controls saved
frames; ``dt_max`` and ``cfl`` control internal steps. Check spatial and
temporal convergence when lowering viscosity or sharpening the initial field.

API reference
-------------

.. automodule:: autosim.experimental.simulations.burgers_1d
   :members:
   :undoc-members:
   :show-inheritance:
