# autosim
Lots of Simulations

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

### Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install the package

```bash
uv pip install -e .
```

### Install development dependencies (includes pytest)

```bash
uv sync --extra dev
```

## Running tests

Once dev dependencies are installed:

```bash
uv run pytest
```

## Generate training data (Hydra CLI)

Use the tiny CLI to generate train/valid/test splits from any simulator that
inherits `SpatioTemporalSimulator`:

```bash
uv run autosim
```

List available simulator configs:

```bash
uv run autosim list
```

Simulator defaults now live in package configs under
`src/autosim/configs/simulator` and can be selected via config groups.
Nested groups are supported, so you can select configs such as
`simulator=gpe/laser_only_wake`.

Available simulator config names include:
`advection_diffusion`, `advection_diffusion_multichannel`, `compressible_fluid_2d`,
`conditioned_navier_stokes_2d`, `epidemic`, `flow_problem`,
`gpe/laser_only_wake`, `gpe/rotating_box_lattice`,
`gray_scott`,
`hydrodynamics_2d`, `lattice_boltzmann`, `projectile`,
`projectile_multioutput`, `reaction_diffusion`, `seir_simulator`,
`shallow_water2d`.

Additional exploratory GPE configs are available under `gpe/exploratory/` for
reference and in-progress work (note: `high_complexity` and `low_complexity`
start from an arbitrary Gaussian, not a physical ground state).

Override simulator and dataset settings from the command line via Hydra:

```bash
uv run autosim \
	simulator=shallow_water2d \
	simulator.nx=32 \
	simulator.ny=32 \
	simulator.T=10.0 \
	dataset.n_train=50 dataset.n_valid=10 dataset.n_test=10 \
	dataset.output_dir=examples/experimental/generated_datasets/shallow_water_small \
	seed=123 overwrite=true
```

Use a faster built-in simulator config:

```bash
uv run autosim \
	simulator=advection_diffusion \
	simulator.n=16 simulator.T=0.2 simulator.dt=0.1 \
	dataset.n_train=1 dataset.n_valid=1 dataset.n_test=1
```

Optionally save example rollout videos for selected batch indices after generation:

```bash
uv run autosim \
	simulator=advection_diffusion_multichannel \
	dataset.n_train=4 dataset.n_valid=1 dataset.n_test=1 \
	visualize.enabled=true \
	visualize.split=train \
	visualize.batch_indices=[0,2] \
	visualize.file_ext=gif
```

By default videos are written under
`<dataset.output_dir>/examples/<split>/batch_<idx>.<ext>`.
Use `visualize.file_ext=mp4` if ffmpeg is available.

	Generate one combined dataset from ordered strata values (single sweep key):

	```bash
	uv run autosim \
		simulator=gray_scott \
		stratify.enabled=true \
		stratify.key=simulator.pattern \
		stratify.values=[gliders,bubbles,maze,worms,spirals,spots] \
		dataset.n_train=240 dataset.n_valid=24 dataset.n_test=24 \
		dataset.output_dir=outputs/gray_scott_combined
	```

	When stratification is enabled, each split size is divided equally across strata,
	and results are concatenated in the exact order of `stratify.values`.
	If a split size is not divisible by the number of strata, an error is raised.

Bring your own simulator subclass (no registry needed):

```bash
uv run autosim \
	simulator._target_=my_package.my_module.MySimulator \
	++simulator.my_arg=42 \
	simulator.log_level=warning
```
