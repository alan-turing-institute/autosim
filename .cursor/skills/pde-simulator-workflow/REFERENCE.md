# Reference: PDE simulator validations

This file is intentionally “one level deep” from `SKILL.md` to keep the main skill concise.

## Common solver patterns in this repo

- **Pseudo-spectral periodic PDEs**
  - Examples: advection-diffusion variants, reaction-diffusion, shallow-water.
  - Spatial derivatives and Poisson solves via FFT; periodic BC implied.
- **Method-of-lines**
  - Build RHS \( \partial_t u = F(u, \theta) \) on a grid, then integrate in time.
  - Used with `scipy.integrate.solve_ivp` or custom RK stepping.
- **Parameter sampling**
  - `Simulator.sample_inputs()` uses QMC (LHS/Sobol) and supports constant params by equal min=max.

## Why “PDE residual check” is not sufficient

Residual checks help catch obvious bugs (wrong sign, swapped derivatives, dt mismatch), but:
- A residual computed with the *wrong discretization* can look “good” while the solver is wrong.
- Many solvers include filters/hyperviscosity/clipping; residual of the “pure PDE” may not be small.
- Boundary conditions, constraints, and invariants can be wrong while residual is small.

Treat residual as a **layer** within a validation stack.

## Validation stack (recommended default)

### 1) Numerical sanity

Always check:
- no NaNs/Infs
- finite norms (L2/L∞)
- any simulator-specific physical bounds (e.g., SWE depth \(h>0\))

### 2) Smoothness / “good per-frame change”

Compute channel-wise deltas:
- \( \Delta_t u_t = u_{t+1}-u_t \)

Track:
- mean/std/max of \(|\Delta_t|\)
- mean/std of \(|u|\)
- ratio \( \max|\Delta_t| / (\max|u|+\epsilon) \)

Use these to catch:
- “numerical blow up” (exploding deltas)
- “too static” outputs (near-zero deltas across all frames)

### 3) PDE residual (when feasible)

Residual is generally:

\[
R(u) \approx \frac{u_{t+1}-u_t}{\Delta t} - F(u_t)
\]

Guidelines:
- Use the **same operators** as the solver when possible.
- Prefer **relative** norms (divide by state scale).
- Compare residual behaviour under refinements (dt halved should not worsen unexpectedly).

Implementation approach for the skill:
- `validate_rollout.py` accepts `--residual module:function`
  - residual function signature:
    - `residual_fn(payload: dict, *, dt: float | None = None, dx: float | None = None, dy: float | None = None) -> dict[str, float]`
  - It should return scalar metrics like `{"residual_l2": ..., "residual_rel_l2": ...}`.

### 4) Invariants / diagnostics (PDE-dependent)

Examples (not universal):
- **Mass conservation**: \( \int u \, dx \) constant (advection without sources)
- **Divergence-free velocity**: \( \nabla\cdot \mathbf{u} \approx 0 \) (incompressible flows)
- **Energy/enstrophy trends**: should decay under viscosity/drag (monotonic-ish, not exploding)
- **SWE depth positivity**: \(h \ge h_{\min}\) (or clip fraction is low)

Like residuals, implement as a plug-in function:
- `--diagnostics module:function`

### 5) Performance budgets

Measure:
- wall time per call to `forward_samples_spatiotemporal(n=1)`
- wall time per frame (`time / n_time`)

Keep budgets simulator-specific and configurable in scripts/tests.

## Picking thresholds (practical defaults)

There is no single correct threshold across PDEs. Use:
- **Absolute caps** for safety (e.g., reject NaN, reject |u| > huge value).
- **Relative caps** for stability (delta-to-state ratio).
- **Regression thresholds** based on a canonical small config per simulator.

For “good smooth change per frame”:
- Use max delta ratio:
  - warn if \( \max|\Delta_t| / (\max|u|+\epsilon) \) is too large (blow-up) or too small (frozen).
- Use per-channel thresholds because channels have different scales.

## Debug playbook: all trajectories failed

When you hit repeated errors like:
- `ODE solver failed: Required step size is less than spacing between numbers`
- `All <Simulator> trajectories failed`

use this sequence:

1. **Fixed-point preflight**
   - Set `parameters_range` to fixed values (`min=max`) at conservative settings.
   - Run one sample first.

2. **Shorten and coarsen**
   - Reduce horizon `T`.
   - Increase saved step size (fewer frames).
   - Lower grid resolution for diagnosis.

3. **Increase damping**
   - Raise viscosity / drag / damping.
   - Lower forcing and IC amplitude.

4. **Solver strategy**
   - For interactive speed: start with `RK45` + fallback.
   - For stiff cases: try `BDF`/`Radau` (expect slower runtime).
   - Relax tolerances for preview runs.

5. **Environment sanity**
   - Check notebook uses current source path (`inspect.getsourcefile`).
   - If mismatch: `%pip install -e .` then restart kernel.

6. **Only then broaden sampling**
   - Restore wider ranges gradually.
   - Keep retries (`ensure_exact_n=True`) only after single-sample success.

## Parameter regimes and interesting dynamics

Simulation parameters should be chosen so that runs are *physically interesting*, not just numerically stable:

- **Time horizon $T$**: Long enough for the PDE to show its characteristic evolution (e.g. coarsening in Allen–Cahn, vortex formation in 2D fluids, pattern formation in reaction–diffusion). Short $T$ is fine for a "stable preview"; add a second, longer run for "interesting dynamics".
- **Domain size and resolution**: Larger domains allow more structure (e.g. multiple phase domains, larger vortices). Balance $n$ and $L$ so that key scales are resolved.
- **Parameter ranges**: Document which ranges yield "easy" (smooth, stable) vs "hard" (stiff, chaotic, or rich) dynamics. See the Gross–Pitaevskii notebook for low-complexity (linear limit, sloshing/breathing) vs high-complexity (turbulence, vortices) examples.
- **Notebook structure**: After a short stable preview and its video, add an **"Interesting dynamics"** section with a config that produces visible, characteristic behaviour, then a second video. Optionally add a "What you will see" list per regime.

