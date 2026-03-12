# Templates: simulator + validation + tests

## Template: example notebook structure (recommended)

**Template notebooks**: Copy structure and patterns from `examples/experimental/00_00_reaction_diffusion.ipynb` or `01_00_advection_diffusion.ipynb`. Use **two introductory markdown cells**:

**First cell** — short title and overview:
- One paragraph describing the simulator and the PDE in plain language.
- Bullet list: **State variables**, **Conditioning variables**, **Dynamics**, **Boundary conditions**.
- A **Why this is useful** subsection (one short paragraph).

**Second cell** — governing equations and parameters:
- **Governing equations**: the PDE(s) in display math (see math formatting below).
- **Boundary conditions**: how they are enforced (with equations if helpful).
- **Parameters**: one bullet per parameter with a short explanation.
- Optionally: **Initial conditions**, **Numerics**, or **Execution order** (stable preview first, then viz, then exploration).

**Math in notebooks**: Use Jupyter-friendly LaTeX so equations render correctly in Cursor and Jupyter:
- **Inline math**: `$...$` (e.g. `$\nu$`, `$[0,L]^2$`).
- **Display (block) math**: `$$...$$` on its own lines.
- Avoid `\(...\)` and `\[...\]`; they often do not render in notebook markdown.

Notebook execution order (recommended):
1. **Import cell** — include `from IPython.display import HTML` and `from autosim.utils import plot_spatiotemporal_video`.
2. **Stable preview config** (fixed conservative parameters; run this first)
3. **Visualization cell** — plot a **spatiotemporal video** using `plot_spatiotemporal_video(batch["data"], batch_idx=0, channel_names=<your output_names>, preserve_aspect=True)` and display it with `HTML(anim.to_jshtml())`. See the template notebook for the exact pattern.
4. **Exploration config** (wider parameter ranges; mark as optional or may fail for stiff PDEs)

**Regimes and interesting dynamics**: To avoid short, uninteresting runs, document how parameters map to *dynamics* and *complexity*:

- **Parameter regimes**: Describe what different parameter choices do (e.g. small $\varepsilon$ = sharp interfaces vs large $\varepsilon$ = diffuse; low viscosity = turbulent vs high = laminar). Optionally use **low vs high complexity** configs (see `03_01_gross_pitaevskii.ipynb`).
- **Time and domain**: Choose $T$ and domain size so the PDE exhibits its characteristic behaviour (e.g. coarsening for Allen–Cahn, breathing/sloshing for GPE, pattern formation for reaction–diffusion). A "stable preview" can use short $T$; add a separate **"Interesting dynamics"** run with longer $T$ and/or larger domain.
- **"What you will see"**: For each regime or example run, add 2–3 bullet points describing the visible dynamics (e.g. "Phase separation: domains of $u \approx \pm 1$ form and coarsen"; "Interference fringes as the packet overlaps with itself"). This helps readers and ML workflows know what to expect.

Reference: `examples/experimental/03_01_gross_pitaevskii.ipynb` (Low Complexity / High Complexity examples and parameter explanations).

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

