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
implements `forward_samples_spatiotemporal`:

```bash
uv run autosim
```

Override simulator and dataset settings from the command line via Hydra:

```bash
uv run autosim \
	simulator.name=ShallowWater2D \
	simulator.kwargs.nx=32 \
	simulator.kwargs.ny=32 \
	simulator.kwargs.T=10.0 \
	dataset.n_train=50 dataset.n_valid=10 dataset.n_test=10 \
	dataset.output_dir=examples/experimental/generated_datasets/shallow_water_small \
	run.seed=123 run.overwrite=true
```

You can also use fully-qualified class paths:

```bash
uv run autosim simulator.name=autosim.experimental.simulations.ShallowWater2D
```
