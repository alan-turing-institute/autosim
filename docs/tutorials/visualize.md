# Visualization

As seen on [the previous page](quickstart.md), the `autosim` command generates visualizations of the simulation results by default.

## Disabling visualizations

To disable the generation of visualizations, you can specify the extra argument

```bash
uv run autosim [...] ++visualize.enabled=false
```

## Changing which trajectories are visualized

You can generate visualizations for the 1st and 3rd training trajectories by specifying:

```bash
uv run autosim [...] ++visualize.split=train '++visualize.batch_indices=[0,2]'
```

## Changing the file output

To generate GIFs instead of videos, you can specify

```bash
uv run autosim [...] ++visualize.file_ext=gif
```

## One-dimensional fields

For trajectories with one singleton spatial axis, the CLI automatically saves
space-time heatmaps as `examples/<split>/batch_<index>.png`. This applies to the
new 1D simulators and the existing 1D field simulators. The `file_ext` setting
continues to control 2D videos; 1D examples are PNG images and need no video encoder.

Generate small examples with:

```bash
uv run autosim simulator=experimental/burgers_1d simulator.nx=64 \
  dataset.n_train=4 dataset.n_valid=1 dataset.n_test=1 seed=7

uv run autosim simulator=experimental/shallow_water_1d simulator.nx=64 \
  dataset.n_train=4 dataset.n_valid=1 dataset.n_test=1 seed=7
```

Both simulators save `[batch, time, nx, 1, channels]` fields, including the actual
initial state at time zero. Burgers has channel `u`; SWE has channels `[h, u]`.
The payload also includes physical coordinates `x` and `times`, each with a batch
axis, and the sampled physical parameters in `constant_scalars`.

Use the plotting utility directly, including physical axes:

```python
from autosim.experimental.simulations import Burgers1D
from autosim.utils import plot_spatiotemporal_1d

sim = Burgers1D(nx=128)
payload = sim.forward_samples_spatiotemporal(2, random_seed=7)
figure = plot_spatiotemporal_1d(
    payload["data"],
    batch_idx=0,
    x=payload["x"][0],
    times=payload["times"][0],
    channel_names=sim.output_names,
    save_path="burgers.png",
)
```

`plot_spatiotemporal_1d` also accepts compact `[batch, time, nx, channels]`
trajectories. Optional `pred` and `pred_uq` tensors add prediction, difference
and uncertainty panels. It returns a Matplotlib figure for further customization.

### Initial conditions and physical parameters

Field sampling is separate from evolution. `sample_initial_conditions(n, seed)`
draws fields; `rollout(initial_state, nu=...)` for Burgers or
`rollout(initial_state, g=...)` for SWE evolves supplied fields without drawing
random numbers. Legacy `forward(parameters)` evolves a fixed reference field:
the supplied constructor `initial_state`, or the field generated with seed zero.
This reference is distinct from the independently sampled fields used in datasets.
If physical parameter bounds vary, `rollout` requires that parameter explicitly.
Otherwise it uses the single value in `parameters_range`.

The default Burgers preset uses the Gaussian covariance
`625 * (-Delta + 25 I)^(-2)` from the
[FNO Burgers benchmark](https://arxiv.org/html/2010.08895v3#A3.SS1), on the unit
periodic domain with viscosity `0.1`. The constant mode is removed by default;
set `initial_condition_kwargs.zero_mean=false` to retain it. Sampling uses all
available paired Fourier modes, without per-sample amplitude normalization.
The grid, zero-mean projection and full-trajectory output should be accounted for
when comparing with published benchmark results. `sine` and `fourier` initial
condition families are available through the Python API with their own sampler
settings.

The SWE preset samples a Gaussian height bump over resting water, inspired by the
[Clawpack perturbation example](https://www.clawpack.org/gallery/pyclaw/gallery/dam_break.html).
Periodic Gaussian images avoid a seam at the boundary. Height increment is sampled
from `[0.05, 0.2]`, width from `[0.05, 0.15]` of the domain length, and center from
the whole domain. These ranges are our dataset choices. The baseline depth is `1`,
initial velocity is zero and gravity is fixed at `1`. The bed is flat, with no
Coriolis, forcing, drag or explicit viscosity. A conservative finite-volume solver
evolves depth and momentum internally but exposes `[h, u]`. Strictly positive
depth is required; wetting/drying is unsupported.

To sample viscosity, override only the physical bounds:

```bash
uv run autosim simulator=experimental/burgers_1d \
  'simulator.parameters_range.nu=[0.05,0.2]' seed=7
```

Changing physical bounds does not change the initial-field distribution. `dt_save`
controls output sampling; `dt_max` and `cfl` control smaller internal steps. Check
convergence when changing viscosity, wave amplitude, grid resolution or timestep.

### Sampling stronger nonlinear dynamics

Two additional presets sample more nonlinear, unforced trajectories:

```bash
uv run autosim simulator=experimental/burgers_1d_nonlinear \
  dataset.n_train=4 dataset.n_valid=1 dataset.n_test=1 seed=7

uv run autosim simulator=experimental/shallow_water_1d_nonlinear \
  dataset.n_train=4 dataset.n_valid=1 dataset.n_test=1 seed=7
```

Both use `nx=512`, `T=1`, `dt_save=0.01` and `dt_max=0.001`. Their ranges
are dataset choices intended to cover nonlinear steepening and wave interactions,
not guarantees that every sampled trajectory forms a shock or exhibits chaos.
The reference presets above retain their original initial-condition laws and
fixed physical parameters.

| Preset | Sampled physical parameters | Sampled initial conditions |
| --- | --- | --- |
| `burgers_1d_nonlinear` | Viscosity `nu` in `[0.005, 0.02]` | Eight Fourier modes, coefficient decay `k^-1.5`, peak speed in `[0.5, 1.0]` |
| `shallow_water_1d_nonlinear` | Gravity `g` in `[0.5, 2.0]` | Gaussian height increment in `[0.3, 0.8]` above depth `1`, width in `[0.03, 0.08]` of the domain, uniform center, zero initial velocity |

Burgers' initial peak-speed Reynolds number `U_peak * L / nu` spans `25` to
`200` across these bounds. Low viscosity allows sharper, longer-lived transient
structures, but periodic unforced viscous Burgers ultimately approaches its
constant spatial mean. This is a regime of
[decaying Burgers turbulence](https://arxiv.org/abs/1208.5241), not sustained chaos.

For SWE, relative height increment `amplitude / h_mean` controls the strength of
the initial perturbation. At fixed height field and zero initial velocity,
changing `g` rescales time by `sqrt(g)` and velocity by `sqrt(g)`; it does not
independently change this relative nonlinearity. The larger, narrower height bumps
produce stronger steepening and subsequent wave interactions. Shocks and
numerical dissipation can still cause decay. Neither preset adds process noise,
forcing, or a claim of positive Lyapunov exponents.

Physical scalars use the existing Latin hypercube sampler; initial-field draws
use their own sampler. The actual initial field is saved in the first frame,
and `constant_scalars` records the viscosity or gravity used for each trajectory.
Override either set of bounds independently, for example:

```bash
uv run autosim simulator=experimental/burgers_1d_nonlinear \
  'simulator.parameters_range.nu=[0.008,0.015]' seed=7

uv run autosim simulator=experimental/shallow_water_1d_nonlinear \
  'simulator.initial_condition_kwargs.amplitude_range=[0.4,0.6]' seed=7
```

Lower viscosity or narrower initial features require spatial convergence checks;
a smaller timestep alone cannot compensate for insufficient grid resolution.
