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

This installs `autosim` in editable mode along with its runtime dependencies:
- `numpy>=1.24`
- `scipy>=1.10`
- `tqdm>=4.65`
- `torch>=2.0`

### Install development dependencies (includes pytest)

```bash
uv sync --group dev
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

Available simulator config names:
`advection_diffusion`, `advection_diffusion_multichannel`, `compressible_fluid_2d`,
`conditioned_navier_stokes_2d`, `epidemic`, `flow_problem`, `gray_scott`,
`gross_pitaevskii_equation_2d`, `hydrodynamics_2d`, `lattice_boltzmann`,
`projectile`, `projectile_multioutput`, `reaction_diffusion`, `seir_simulator`,
`shallow_water2d`.

Override simulator and dataset settings from the command line via Hydra:

```bash
uv run autosim \
	simulator=shallow_water2d \
	simulator.nx=32 \
	simulator.ny=32 \
	simulator.T=10.0 \
	dataset.n_train=50 dataset.n_valid=10 dataset.n_test=10 \
	dataset.output_dir=examples/experimental/generated_datasets/shallow_water_small \
	run.seed=123 run.overwrite=true
```

Use a faster built-in simulator config:

```bash
uv run autosim \
	simulator=advection_diffusion \
	simulator.n=16 simulator.T=0.2 simulator.dt=0.1 \
	dataset.n_train=1 dataset.n_valid=1 dataset.n_test=1
```

Bring your own simulator subclass (no registry needed):

```bash
uv run autosim \
	simulator._target_=my_package.my_module.MySimulator \
	++simulator.my_arg=42 \
	simulator.log_level=warning
```
