autosim.experimental.simulations.kuramoto_sivashinsky_1d
========================================================

Governing equation and regimes
------------------------------

The deterministic periodic Kuramoto-Sivashinsky (KS) equation is

.. math::

   \partial_t u + \partial_x\!\left(\frac{u^2}{2}\right)
       + \partial_{xx}u + \nu\,\partial_{xxxx}u = 0,
   \qquad x\in[0,L),\quad \nu>0.

The initial field specifies :math:`u(x,0)=u_0(x)`. The field and its derivatives
are periodic. There is no external forcing or process noise. Here ``nu`` is
the coefficient of fourth-order dissipation, with a different meaning from
Burgers viscosity. The standard nondimensional equation uses ``nu=1``.

The Fourier linear growth rate is :math:`k^2-\nu k^4`: long wavelengths are
unstable and short wavelengths are damped. Nonlinear advection couples the
modes. Their balance can sustain chaotic evolution, and the spatial mean is
conserved. When :math:`L<2\pi\sqrt{\nu}`, all nonconstant modes are linearly
stable and the energy of the mean-subtracted field decreases.

The ``L=22, nu=1`` case targets the established chaotic benchmark of
`Cvitanovic, Davidchack and Siminos (2010)
<https://arxiv.org/abs/0709.2944>`_. This is a model of instability and
spatiotemporal chaos; it complements the nonlinear fluid transients of
Burgers and shallow water. Changing ``nu`` changes the effective domain
size ``L/sqrt(nu)`` and the time scale, so wider parameter ranges require
fresh regime checks.

Generate a dataset
------------------

The reference preset records evolution from raw sampled initial fields,
without warm-up or chaos screening:

.. code-block:: console

   uv run autosim simulator=experimental/kuramoto_sivashinsky_1d \
     dataset.n_train=4 dataset.n_valid=1 dataset.n_test=1 seed=7

The chaotic preset uses ``L=22``, ``nu=1``, ``nx=128`` and ``dt_max=0.1``.
It warms each field for 100 time units, checks activity and estimates
perturbation growth over 200 time units, then records a 100-unit forecast:

.. code-block:: console

   uv run autosim simulator=experimental/kuramoto_sivashinsky_1d_chaotic \
     dataset.n_train=4 dataset.n_valid=1 dataset.n_test=1 seed=7

The stable control uses ``L=5`` and records relaxation from the initial field:

.. code-block:: console

   uv run autosim simulator=experimental/kuramoto_sivashinsky_1d_stable \
     dataset.n_train=4 dataset.n_valid=1 dataset.n_test=1 seed=7

All presets sample generic zero-mean eight-mode Fourier fields with coefficient
decay ``k^-1.5`` and peak amplitudes in ``[0.5, 1.0]``. Both sine and cosine
coefficients are drawn, avoiding an imposed reflection-symmetric subspace.
These are dataset choices; coefficients remain fixed at the standard value.

Each split's ``data.pt`` includes ``data`` shaped ``[batch, time, nx, 1, 1]``,
channel ``u``, coordinates ``x`` and ``times`` with batch axes, and
``constant_scalars`` containing fourth-order dissipation ``nu``.
The first frame is the actual forecast initial field. With warm-up enabled,
recorded time zero is the end of warm-up. ``seed_fields`` retains the raw
sampled fields, shaped ``[batch, nx, 1]``, and ``warmup_times`` records the
duration for each sample. The CLI automatically saves space-time PNGs under
``examples/<split>/batch_<index>.png``.

Chaos screening and its limits
-------------------------------

The chaotic preset accepts a sample only if both checks pass:

* The RMS change of its warmed field after one time unit exceeds ``1e-6``.
  This rejects quiescent states, including unstable equilibria whose
  perturbations can grow without the reference trajectory being chaotic.
* Its estimated largest Lyapunov exponent exceeds ``0.01`` per time unit.
  Float64 paired trajectories are perturbed at RMS distance ``1e-7``;
  the perturbation is renormalized every time unit and logarithmic growth
  rates are averaged over the validation window.

``activity_estimates`` and ``lyapunov_estimates`` retain these diagnostics,
each shaped ``[batch, 1]``. Failed checks raise with the sample index; no
sample is silently replaced. Redraw explicitly to obtain a different sample.
Setting ``chaos_validation_time=0`` disables screening.

These are finite-time acceptance criteria, not a proof of asymptotic chaos
for every sample. Longer windows, several initial fields, perturbation-size
checks, and spatial/time refinement are needed when changing the preset.
Chaotic trajectories can temporarily contract, and regular parameter windows
exist: complexity or positive growth over a short transient is insufficient.
See `Edson et al.'s Lyapunov-spectrum study
<https://arxiv.org/abs/1902.09651>`_.

Evolve and diagnose an explicit initial field
---------------------------------------------

Sampling, warm-up, direct rollout and chaos diagnostics are separate:

.. code-block:: python

   from autosim.experimental.simulations import (
       KuramotoSivashinsky1D,
       estimate_ks_lyapunov,
   )
   from autosim.utils import plot_spatiotemporal_1d

   sim = KuramotoSivashinsky1D(warmup_time=100.0)
   raw = sim.sample_initial_conditions(1, random_seed=7)
   initial_state = sim.warmup_state(raw, nu=1.0)
   trajectory = sim.rollout(initial_state, nu=1.0)
   exponent = estimate_ks_lyapunov(
       initial_state, nu=1.0, L=sim.L, warmup_time=0.0, T=200.0,
   )
   figure = plot_spatiotemporal_1d(
       trajectory, x=sim.x, times=sim.times,
       channel_names=sim.output_names, save_path="kuramoto_sivashinsky.png",
   )

``rollout`` always starts with the supplied field and applies neither warm-up
nor screening. Legacy ``forward(parameters)`` likewise evolves the supplied
constructor reference field, or the sampler's seed-zero raw field. Dataset
warm-up and screening apply to ``forward_samples_spatiotemporal``.

Numerical method
----------------

The solver uses Fourier differentiation, two-thirds-dealiased conservative
advection and fourth-order exponential time differencing Runge-Kutta (ETDRK4).
Contour evaluation avoids cancellation near zero eigenvalues, following
`Kassam and Trefethen's KS example
<https://people.maths.ox.ac.uk/trefethen/tda05.pdf>`_. This integrates the stiff
linear term exponentially. ``dt_max`` must still resolve nonlinear dynamics;
``dt_save`` independently controls saved output times. Fourier modes above the
dealiasing cutoff evolve linearly. Input float32/float64 precision, device,
leading batch axes and state gradients are retained; diagnostics use float64.

Check convergence over short forecasts and compare statistics and diagnostic
growth over long runs. Chaotic sensitivity prevents indefinite pointwise
agreement between numerically refined trajectories.

API reference
-------------

.. automodule:: autosim.experimental.simulations.kuramoto_sivashinsky_1d
   :members:
   :undoc-members:
   :show-inheritance:
