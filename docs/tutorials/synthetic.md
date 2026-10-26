# Synthetic performance datasets

The `synthetic` category generates reproducible Gaussian fields for software
checks and performance measurements. Shapes and correlations are configurable;
the fields have no physical interpretation. The output uses the usual
`[trajectory, time, height, width, channels]` layout and AutoSim export pipeline.

Generate a small dataset:

```bash
uv run autosim --config-name=generate_data_synthetic \
  dataset.output_dir=outputs/gaussian_fields
```

This preset produces two training trajectories and one each for validation and
test, with four frames on a **16×32 grid and four channels**. Channel names default
to `field_0`, `field_1`, etc. Override `simulator.height`, `simulator.width`,
`simulator.n_channels` and `simulator.n_steps` to match the workload. The generator
preserves the requested shape; padding or cropping belongs in the consuming model.

The default fields are independent across space, time and channels. Optional
correlations can make training checks less trivial:

```bash
uv run autosim --config-name=generate_data_synthetic \
  dataset.output_dir=outputs/gaussian_correlated \
  simulator.rho=0.8 \
  simulator.spatial_length_scale=2.0 \
  simulator.channel_correlation=0.4
```

- `rho` sets the lag-one temporal correlation using a stationary AR(1) process.
  Its magnitude must be below one; negative values are supported.
- `spatial_length_scale` is the Gaussian smoothing width in grid cells. It uses
  periodic boundaries along **both** axes, with normalization that preserves
  unit marginal variance.
- `channel_correlation` gives every distinct channel pair the same correlation
  through a shared Gaussian component. It lies in `[0, 1)`; channels remain
  independent when zero.

The data retain unit marginal variance in expectation; individual finite
datasets need not have exactly zero mean or unit variance. Correlations are
separable across space, time and channels. The seed is local to the generator,
does not reset the global Torch RNG, and is recorded in each split's provenance.
The existing CLI assigns separate split seeds and writes training-derived
`stats.yml`, `resolved_config.yaml` and `train/valid/test/data.pt`.

For a different shape and trajectory count:

```bash
uv run autosim --config-name=generate_data_synthetic \
  simulator.height=16 simulator.width=24 simulator.n_channels=8 \
  simulator.n_steps=6 dataset.n_train=4 dataset.n_valid=1 dataset.n_test=1 \
  dataset.output_dir=outputs/gaussian_custom
```

## Memory and timing

The output tensor uses `trajectories × frames × height × width × channels ×
bytes_per_value` bytes (four bytes for FP32, eight for FP64). Generation is
CPU-only and in memory; it is not a streaming dataset implementation.
`max_output_bytes` defaults to 512 MiB and rejects an oversized **single output
tensor** before allocation. It does not cap total process memory: the CLI holds
all splits and computes additional delta tensors for normalisation. Prefer the
`generate_data_synthetic` preset over the generic CLI's default trajectory counts.

Generate datasets before timing training. Synthetic inputs can establish model
memory, training throughput and communication costs only when the measured
architecture, patching, ensemble size, loss, precision, batch, backward pass
and optimiser match the target workload. They do not establish real-data loading
performance, required training duration, convergence or predictive quality.

For distributed timing, use enough samples for the global batch and sampler;
the default tiny dataset is for smoke tests. Report the timings as **synthetic
workloads**, including their shape and whether input data were resident in memory.

## Example: matching an ERA5 input layout

An optional configuration supplies a **121×240 grid and 69 generic channels**,
matching a possible 1.5-degree ERA5 input layout with 65 pressure-level and four
surface fields. Adjust the channel count to match the actual model inputs.

```bash
uv run autosim --config-name=generate_data_era5_shape \
  dataset.output_dir=outputs/era5_shape
```

This configuration inherits the generic preset's split counts, frame count and
seed. It only changes the grid and channel count; it does not read ERA5 or model
atmospheric statistics or dynamics. Each FP32 frame is about 7.64 MiB, giving
about 122 MiB across the default splits before statistics and temporary buffers.
