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
The default CLI assigns separate split seeds and writes training-derived
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
CPU-only. The default `.pt` export holds the complete dataset in memory.
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

## Large datasets with Zarr

The optional Zarr exporter writes continuous trajectories in bounded batches and
computes training statistics incrementally. It requires Python 3.11 or later:

```bash
uv run --extra zarr autosim --config-name=generate_data_synthetic_zarr \
  dataset.output_dir=outputs/gaussian_zarr
```

Each split contains a `data.zarr` group with a `data` array in the same
`[trajectory, time, height, width, channels]` order. Group attributes record the
`autosim.spatiotemporal` schema (version 1), axes, field names, provenance and
completion status. `stats.yml` and `resolved_config.yaml` remain at dataset root.

Two independent settings control storage:

- `dataset.chunk_steps`: frames read independently, normally a small number
  suitable for training windows.
- `dataset.shard_steps`: frames packed into one storage file; this must be a
  multiple of `chunk_steps`. Generation writes one complete shard at a time.

Storage boundaries do not reset the random generator or temporal correlations,
and windows can cross shards without duplicated frames. Trajectories and splits
remain independent. `dataset.steps_per_split` sets each split's trajectory
length separately; the usual `n_train`, `n_valid` and `n_test` set trajectory
counts. Model context and forecast lengths are chosen by the reader, not stored
as fixed training windows.

The allocation limit applies to one shard payload. Generation, codecs and
float64 statistics use additional working memory proportional to shard size.
Statistics include temporal differences crossing shard boundaries and exclude
differences between independent trajectories. Arrays, training statistics and
resolved configuration are staged together; stores become complete only after
publication finishes. Zarr export only creates new datasets: choose a fresh
output directory for each run, and leave `overwrite=false`. Existing dataset
stores or metadata are rejected before generation. This route currently supports
Gaussian fields with per-channel normalization and serial generation;
visualization and stratification remain available through the default `.pt` route.

Tune read chunks using the actual DataLoader's window lengths and access pattern.
Larger shards reduce file count, while larger read chunks can increase bytes
read for short windows. Measure worker startup separately from steady-state
loading, and distinguish warm filesystem-cache results from storage throughput.

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

For a full-size **synthetic size reference**, use:

```bash
uv run --extra zarr autosim --config-name=generate_data_era5_full \
  dataset.output_dir=outputs/era5_full
```

This creates one continuous trajectory per split: 58,440 training frames,
1,460 validation frames and 1,464 test frames, matching six-hourly counts for
1979–2018, 2019 and 2020. Raw FP32 storage is approximately **468.4 GB for training
and 491.8 GB overall**. These are synthetic sequences, without dates or
meteorological variable identities. With two context frames and 12 target
frames, the training sequence supplies 58,427 stride-one windows regardless of
storage layout.

The starting layout uses one-frame read chunks (about 7.64 MiB) and 64-frame
shards (about 489 MiB). These are configurable starting values for DataLoader
measurements, not an established optimum. A small check of the same preset can
override `dataset.steps_per_split.train`, `.valid` and `.test` and the simulator
dimensions before allocating a full dataset.
