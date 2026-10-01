# Add a custom simulator

AutoSim simulators are ordinary Python classes. To add one, subclass
`Simulator` for tabular outputs or `SpatioTemporalSimulator` when you want to
use the dataset-generation CLI.

This tutorial builds a small spatiotemporal simulator whose single field decays
exponentially over time. The same pattern applies to a numerical solver or an
external simulation package.

## Implement the simulator

Create a Python module containing a `SpatioTemporalSimulator` subclass. A
spatiotemporal simulator implements `_forward` for one sampled parameter vector
and `forward_samples_spatiotemporal` to reshape a batch into AutoSim's standard
`[batch, time, x, y, channels]` layout.

```python
import torch

from autosim.simulations.base import SpatioTemporalSimulator
from autosim.types import TensorLike


class DecayField(SpatioTemporalSimulator):
    def __init__(
        self,
        parameters_range=None,
        output_names=None,
        log_level="warning",
        n=8,
        steps=5,
    ):
        if parameters_range is None:
            parameters_range = {"decay": (0.1, 1.0)}
        if output_names is None:
            output_names = ["field"]
        super().__init__(parameters_range, output_names, log_level)
        self.n = n
        self.steps = steps

    def _forward(self, x: TensorLike) -> TensorLike:
        assert x.shape == (1, self.in_dim)
        decay = x[0, self.get_parameter_idx("decay")]
        time = torch.arange(self.steps, dtype=x.dtype, device=x.device)
        values = torch.exp(-decay * time)
        field = values[:, None, None].expand(self.steps, self.n, self.n)
        return field.reshape(1, -1)

    def forward_samples_spatiotemporal(
        self,
        n: int,
        random_seed: int | None = None,
        ensure_exact_n: bool = False,
    ) -> dict:
        y, x = self._forward_batch_with_optional_retries(
            n=n,
            random_seed=random_seed,
            ensure_exact_n=ensure_exact_n,
        )
        return {
            "data": y.reshape(y.shape[0], self.steps, self.n, self.n, 1),
            "constant_scalars": x,
            "constant_fields": None,
        }
```

`parameters_range` controls the parameter sampler used by `sample_inputs`.
Parameters whose lower and upper bounds are equal are treated as constants.
`output_names` names the final channel dimension and is also used when AutoSim
writes normalization statistics.

The base class provides `_forward_batch_with_optional_retries`, so failed
single simulations can be retried when dataset generation requests an exact
number of samples.

## Test it directly

You do not need Hydra to develop a simulator. Instantiate the class and check
its output first:

```python
sim = DecayField(n=4, steps=3)
samples = sim.forward_samples_spatiotemporal(n=2, random_seed=7)

assert samples["data"].shape == (2, 3, 4, 4, 1)
assert samples["constant_scalars"].shape == (2, 1)
```

For a non-spatiotemporal simulator, subclass `Simulator` instead, implement only
`_forward`, and use `sample_inputs`, `forward`, or `forward_batch` directly.

## Add a Hydra configuration

To make a simulator available through `autosim`, add a YAML file under
`src/autosim/configs/simulator/`. For example,
`src/autosim/configs/simulator/spatiotemporal/decay_field.yaml`:

```yaml
_target_: autosim.simulations.spatiotemporal.decay_field.DecayField
log_level: warning
n: 8
steps: 5
parameters_range:
  decay: [0.1, 1.0]
```

If the simulator is being added to AutoSim itself, put its implementation under
`src/autosim/simulations/` (or `src/autosim/experimental/simulations/` for an
experimental implementation) and export the class from the corresponding
`__init__.py` module.

The new config will then appear in:

```bash
uv run autosim list
```

Generate a small dataset with:

```bash
uv run autosim \
  simulator=spatiotemporal/decay_field \
  dataset.n_train=4 dataset.n_valid=2 dataset.n_test=2 \
  visualize.enabled=false
```

The CLI requires `SpatioTemporalSimulator` because generated datasets need a
well-defined time/spatial/channel layout. The resulting `data.pt` files use the
same structure described in the [quickstart](quickstart.md).

## What to test before contributing

A simulator contribution should at minimum check that sampled parameters stay
inside their configured ranges, `_forward` returns `(1, out_dim)` worth of data,
`forward_samples_spatiotemporal` returns the documented five-dimensional data
layout, and a tiny Hydra-configured dataset can be generated successfully.

Keep simulator-specific numerical tests close to the implementation. Expensive
or optional solver dependencies should remain optional where possible, with a
clear error when the simulator is selected without them.
