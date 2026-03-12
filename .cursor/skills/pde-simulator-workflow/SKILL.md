---
name: pde-simulator-workflow
description: Standardize implementing/refactoring PDE-style SpatioTemporalSimulator simulators in autosim. Use when adding or changing PDE simulators, rollouts, solver time-stepping, stability checks, PDE residual validation, invariant/diagnostic checks, benchmarking, or adding pytest coverage for simulators.
---

# PDE Simulator Workflow (autosim)

This skill guides a repeatable workflow for making `SpatioTemporalSimulator` PDE simulators **simpler, more accurate, more tested, and faster**.

The workflow assumes this repo’s conventions:
- Simulators inherit `SpatioTemporalSimulator` in `src/autosim/simulations/base.py`.
- Data-generation path uses `forward_samples_spatiotemporal()` returning `{"data","constant_scalars","constant_fields"}`.
- Shapes for spatiotemporal data are `[batch,time,x,y,channels]`.
- Simulator configs live in `src/autosim/configs/simulator/*.yaml` and are used by `src/autosim/cli.py`.
- Tests are `pytest` under `tests/`.

## Quick start (choose one)

### A) Validate a simulator rollout now

Run:

```bash
uv run python scripts/pde_sim/validate_rollout.py \
  --target autosim.simulations.AdvectionDiffusion \
  --kwargs return_timeseries=true n=16 L=4.0 T=0.2 dt=0.1 log_level=warning \
  --n 2 --seed 42
```

Optionally add a residual function:

```bash
uv run python scripts/pde_sim/validate_rollout.py \
  --target autosim.experimental.simulations.ShallowWater2D \
  --kwargs return_timeseries=true nx=32 ny=32 T=2.0 dt_save=0.2 log_level=warning \
  --n 1 --seed 123 \
  --residual autosim.pde_residuals:shallow_water_residual \
  --diagnostics autosim.pde_residuals:shallow_water_diagnostics
```

### B) Benchmark a simulator

```bash
uv run python scripts/pde_sim/benchmark_simulator.py \
  --target autosim.simulations.AdvectionDiffusion \
  --kwargs return_timeseries=true n=16 L=4.0 T=0.2 dt=0.1 log_level=warning \
  --n 1 --seed 0 --warmup 1 --runs 5
```

### C) Add a new PDE simulator or refactor an existing one

Use the templates in `TEMPLATES.md`, then follow the workflow below (validation → tests → perf).

## Core workflow (follow in order)

### 1) Identify the “simulation contract”

For the simulator under work, write down:
- **PDE & boundary conditions** (periodic is common in this repo via FFT).
- **State channels** and `output_names`.
- **Parameterization**: entries in `parameters_range` and how they’re interpreted (by index vs by name).
- **Return mode**: `return_timeseries` True/False and what time axis means (saved frames vs solver internal steps).

If you can’t express the PDE clearly, you can’t validate it well. Use `examples/experimental/*.ipynb` for canonical PDE descriptions.

### 2) Standardize I/O and shapes (must-pass invariants)

Ensure:
- `_forward()` accepts exactly one sample (`x.shape[0] == 1`) and returns `(1, out_dim_flat)` tensor.
- `forward_samples_spatiotemporal()` reshapes to `[batch,time,x,y,channels]` and returns `constant_scalars` containing the sampled parameters used.
- `output_names` are explicit and non-generic (see `tests/simulations/test_all_simulator_output_names.py`).

### 3) Add layered validation (PDE residual is useful, not sufficient)

Use these layers:
- **Numerical sanity**: finite values, reasonable ranges, no sudden overflow.
- **Stability / smoothness**: per-frame delta statistics (mean/std/max) per channel.
- **PDE residual** (optional but recommended): compute a discrete residual norm against the simulator’s discretization or an agreed stencil.
- **Invariants / diagnostics** (when applicable): e.g., mass conservation, divergence-free velocity, energy/enstrophy trends.
- **Performance**: ms/frame or ms/step budget on a small canonical config.

Run `scripts/validate_rollout.py` and capture the JSON output (useful for regression tests and CI).

### 4) Turn validations into tests (fast + deterministic)

Add or extend tests so they are:
- **Fast**: small grids, short T, minimal samples.
- **Deterministic**: use fixed seeds and stable configs.
- **Actionable**: failures should indicate what metric broke (finite, delta explode, residual too large, too slow).

Preferred test patterns in this repo:
- “unit” tests around base behaviour: `tests/simulations/test_base_simulator.py`
- “integration” tests via CLI/Hydra: `tests/test_cli.py`

### 5) Optimize safely (don’t trade correctness for speed silently)

Common safe wins:
- Precompute spectral grids/operators and reuse them.
- Vectorize inner loops; avoid Python loops per timestep when possible.
- Use torch FFT on GPU where appropriate (if simulation is torch-native).
- Reduce allocations: reuse buffers; avoid repeated reshape/copies.

After any optimization, re-run the validation + tests + benchmark scripts.

## Decision points (pick defaults, document escapes)

### PDE residual: what is “good enough”?

There is no universal threshold. Use:
- a **relative** residual (e.g., residual norm divided by state norm + eps), and
- compare against **dt/dx changes** (residual should not worsen unexpectedly when you refine).

See `REFERENCE.md` for residual strategies and choosing tolerances.

## Additional resources (one level deep)

- Residual/invariant/stability guidance: `REFERENCE.md`
- Templates for new simulators, validation plans, tests: `TEMPLATES.md`

