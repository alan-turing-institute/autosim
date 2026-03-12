# Templates: simulator + validation + tests

## Template: example notebook checklist (recommended)

Each simulator should have an example/comparison notebook (see `examples/experimental/*.ipynb`) whose *first markdown cells* include:

- **PDE**: the governing equation(s) in math form.
- **Physics**: what phenomenon it models and why it’s interesting.
- **Symbols**: define all state variables, parameters, operators (e.g. \(u,v,\omega,\psi,\nu,\mu\)).
- **Initial conditions**: how they’re generated, plus rationale and any randomness/seed story.
- **Boundary conditions**: periodic/Dirichlet/Neumann and how they’re enforced numerically.
- **Assumptions**: dimensionality, incompressibility, nondimensionalization, forcing, filtering/hyperviscosity/clipping, solver tolerances.
- **What distinguishes this PDE**: key regimes, parameters, outputs/channels, failure modes, what makes it different from existing simulators.

Optional but very helpful:
- **Numerics**: spatial discretization + time integrator (and any CFL/stability constraints).
- **Validation notes**: which residual/diagnostics are meaningful and how to run them.
- **Runtime notes**: a small “fast config” for quick iteration.

Notebook execution order (recommended):
1. **Import cell**
2. **Fast/stable preview config** (fixed conservative params; runs first)
3. **Visualization cell**
4. **Exploration config** (wider ranges; clearly marked as may fail)

## Template: new `SpatioTemporalSimulator` skeleton

Copy and adapt:

```python
import torch

from autosim.simulations.base import SpatioTemporalSimulator
from autosim.types import TensorLike


class MyPDESimulator(SpatioTemporalSimulator):
    def __init__(
        self,
        parameters_range: dict[str, tuple[float, float]] | None = None,
        output_names: list[str] | None = None,
        return_timeseries: bool = False,
        log_level: str = "progress_bar",
        # grid / domain / time params...
    ) -> None:
        if parameters_range is None:
            parameters_range = {
                # "param": (min, max),
            }
        if output_names is None:
            output_names = [
                # "channel1", "channel2", ...
            ]
        super().__init__(parameters_range, output_names, log_level)
        self.return_timeseries = return_timeseries
        # assign config...

    def _forward(self, x: TensorLike) -> TensorLike:
        if x.shape[0] != 1:
            raise ValueError("Simulator._forward expects a single input (batch size 1)")
        # parse params (prefer by name if optional params exist)
        # run solver -> produce tensor y_flat shape (features,)
        return y_flat.unsqueeze(0)

    def forward_samples_spatiotemporal(
        self,
        n: int,
        random_seed: int | None = None,
        ensure_exact_n: bool = False,
    ) -> dict:
        y, x = self._forward_batch_with_optional_retries(
            n=n, random_seed=random_seed, ensure_exact_n=ensure_exact_n
        )
        # reshape to [batch,time,x,y,channels]
        y = y.reshape(y.shape[0], n_time, nx, ny, n_channels)
        return {"data": y, "constant_scalars": x, "constant_fields": None}
```

## Template: per-simulator validation plan

Fill this out next to your simulator implementation (in docs or PR description):

```text
Simulator: <ClassName>
PDE: <equation(s)>
Boundary conditions: <periodic/dirichlet/...>
State channels (output_names): [...]
Parameters (parameters_range): <name -> meaning>

Sanity checks:
- [ ] finite
- [ ] bounds (if any)

Smoothness/stability checks:
- [ ] delta stats per channel
- thresholds: <what and why>

Residual checks:
- residual definition: <discrete form>
- operator matches solver: <yes/no>
- thresholds: <relative metric, rationale>

Invariants/diagnostics:
- [ ] mass / energy / enstrophy / div-free / positivity...
- thresholds/trends: <what and why>

Performance:
- canonical config: <small grid, short T>
- budget: <ms/call or ms/frame>

Preflight:
- [ ] single-sample fixed conservative parameters succeeds at target horizon
- [ ] only after preflight, broaden parameter ranges
- [ ] notebook kernel points to repo source (`inspect.getsourcefile`)
```

## Template: pytest smoke test for a simulator rollout

Use small configs and deterministic seeds.

```python
import torch


def test_<sim>_rollout_smoke() -> None:
    from autosim.simulations import AdvectionDiffusion

    sim = AdvectionDiffusion(
        return_timeseries=True,
        log_level="warning",
        n=8,
        L=4.0,
        T=0.1,
        dt=0.1,
    )
    batch = sim.forward_samples_spatiotemporal(n=1, random_seed=0, ensure_exact_n=True)
    data = batch["data"]
    assert torch.isfinite(data).all()
    assert data.ndim == 5  # [batch,time,x,y,channels]
    assert data.shape[0] == 1
    assert data.shape[-1] == len(sim.output_names)
```

## Template: regression fixture generation (optional)

If you want strict regression:
- generate a tiny rollout fixture once
- check against it in CI with a tolerance

Use `scripts/make_regression_fixture.py` to generate fixtures.

