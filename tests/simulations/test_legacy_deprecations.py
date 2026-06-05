import pytest
import torch

from autosim.simulations import AdvectionDiffusion
from autosim.simulations.spatiotemporal import (
    AdvectionDiffusionMultichannel,
)


def test_deprecated_advection_diffusion_matches_vorticity_channel() -> None:
    fixed_params = {"nu": (0.001, 0.001), "mu": (1.0, 1.0)}

    with pytest.warns(DeprecationWarning, match="AdvectionDiffusion"):
        legacy = AdvectionDiffusion(
            parameters_range=fixed_params,
            return_timeseries=False,
            n=8,
            L=4.0,
            T=0.25,
            dt=0.25,
            log_level="warning",
        )
    canonical = AdvectionDiffusionMultichannel(
        parameters_range=fixed_params,
        output_indices=[0],
        return_timeseries=False,
        n=8,
        L=4.0,
        T=0.25,
        dt=0.25,
        log_level="warning",
    )

    legacy_out = legacy.forward_samples_spatiotemporal(n=1, random_seed=7)
    canonical_out = canonical.forward_samples_spatiotemporal(n=1, random_seed=7)

    assert legacy.output_names == ["vorticity"]
    assert legacy_out["data"].shape == (1, 1, 8, 8, 1)
    assert torch.allclose(legacy_out["data"], canonical_out["data"])
