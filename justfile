set shell := ["zsh", "-cu"]

# Prefer running everything inside uv's environment.
uv_run := "uv run"

# Show available recipes.
default:
  @just --list

# -----------------------------
# Validate rollouts (sanity/deltas/residuals)
# -----------------------------

validate target *kwargs:
  {{uv_run}} python scripts/pde_sim/validate_rollout.py \
    --target {{target}} \
    --kwargs {{kwargs}}

validate_advection:
  {{uv_run}} python scripts/pde_sim/validate_rollout.py \
    --target autosim.simulations.AdvectionDiffusion \
    --kwargs return_timeseries=true n=16 L=4.0 T=0.2 dt=0.1 log_level=warning \
    --n 1 --seed 0 --ensure-exact-n

validate_swe:
  {{uv_run}} python scripts/pde_sim/validate_rollout.py \
    --target autosim.experimental.simulations.ShallowWater2D \
    --kwargs return_timeseries=true nx=16 ny=16 Lx=16.0 Ly=16.0 T=0.4 dt_save=0.2 log_level=warning \
    --n 1 --seed 0 --ensure-exact-n \
    --residual autosim.pde_residuals:shallow_water_residual \
    --diagnostics autosim.pde_residuals:shallow_water_diagnostics

# -----------------------------
# Benchmark
# -----------------------------

bench target *kwargs:
  {{uv_run}} python scripts/pde_sim/benchmark_simulator.py \
    --target {{target}} \
    --kwargs {{kwargs}}

bench_advection:
  {{uv_run}} python scripts/pde_sim/benchmark_simulator.py \
    --target autosim.simulations.AdvectionDiffusion \
    --kwargs return_timeseries=true n=16 L=4.0 T=0.2 dt=0.1 log_level=warning \
    --n 1 --seed 0 --warmup 1 --runs 5

# -----------------------------
# Regression fixtures
# -----------------------------

fixture out target *kwargs:
  {{uv_run}} python scripts/pde_sim/make_regression_fixture.py \
    --out {{out}} \
    --target {{target}} \
    --kwargs {{kwargs}}

