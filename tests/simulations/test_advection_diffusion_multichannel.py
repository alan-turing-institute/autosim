from __future__ import annotations

import pytest
import torch

from autosim.simulations import AdvectionDiffusionMultichannel


def test_output_indices_validation() -> None:
    with pytest.raises(ValueError, match="at least one"):
        AdvectionDiffusionMultichannel(output_indices=[])

    with pytest.raises(ValueError, match="must not contain duplicate"):
        AdvectionDiffusionMultichannel(output_indices=[0, 0])

    with pytest.raises(ValueError, match=r"range \[0, 3\]"):
        AdvectionDiffusionMultichannel(output_indices=[4])


def test_output_indices_select_and_order_channels() -> None:
    fixed_params = {"nu": (0.001, 0.001), "mu": (1.0, 1.0)}

    sim_full = AdvectionDiffusionMultichannel(
        parameters_range=fixed_params,
        output_indices=[0, 1, 2, 3],
        return_timeseries=False,
        n=8,
        L=4.0,
        T=0.25,
        dt=0.25,
        log_level="warning",
    )
    sim_subset = AdvectionDiffusionMultichannel(
        parameters_range=fixed_params,
        output_indices=[0, 2],
        return_timeseries=False,
        n=8,
        L=4.0,
        T=0.25,
        dt=0.25,
        log_level="warning",
    )

    full = sim_full.forward_samples_spatiotemporal(n=1, random_seed=7)
    subset = sim_subset.forward_samples_spatiotemporal(n=1, random_seed=7)

    assert sim_subset.output_names == ["vorticity", "v"]
    assert full["data"].shape == (1, 1, 8, 8, 4)
    assert subset["data"].shape == (1, 1, 8, 8, 2)

    expected_subset = full["data"][..., [0, 2]]
    assert torch.allclose(subset["data"], expected_subset)
