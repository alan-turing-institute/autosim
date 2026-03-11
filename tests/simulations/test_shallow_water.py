import pytest
import torch

from autosim.experimental.simulations import ShallowWater2D
from autosim.experimental.simulations.shallow_water import simulate_swe_2d


def test_full_swe_timeseries_shape_and_finite() -> None:
    sim = ShallowWater2D(
        return_timeseries=True,
        log_level="warning",
        nx=32,
        ny=32,
        Lx=32.0,
        Ly=32.0,
        T=10.0,
        dt_save=1.0,
        parameters_range={"amp": (0.1, 0.1)},
    )

    out = sim.forward_samples_spatiotemporal(n=1, random_seed=0)
    data = out["data"]

    expected_frames = int(sim.T / sim.dt_save) + 1
    assert data.shape == (1, expected_frames, sim.nx, sim.ny, 3)
    assert torch.isfinite(data).all()

    h = data[0, ..., 0]
    assert (h[-1] - h[0]).abs().max().item() > 1e-4


def test_full_swe_skip_nt_reduces_timeseries_length() -> None:
    sim = ShallowWater2D(
        return_timeseries=True,
        log_level="warning",
        nx=24,
        ny=24,
        Lx=24.0,
        Ly=24.0,
        T=5.0,
        dt_save=1.0,
        skip_nt=2,
        parameters_range={"amp": (0.1, 0.1)},
    )

    out = sim.forward_samples_spatiotemporal(n=1, random_seed=0)
    data = out["data"]

    expected_frames = int(sim.T / sim.dt_save) + 1 - sim.skip_nt
    assert data.shape == (1, expected_frames, sim.nx, sim.ny, 3)


def test_full_swe_skip_nt_too_large_raises() -> None:
    with pytest.raises(ValueError, match="skip_nt is too large"):
        simulate_swe_2d(
            amp=0.12,
            return_timeseries=True,
            nx=24,
            ny=24,
            Lx=24.0,
            Ly=24.0,
            T=1.0,
            dt_save=1.0,
            cfl=0.12,
            g=9.81,
            h_mean=1.0,
            nu=5e-4,
            drag=2e-3,
            skip_nt=2,
        )
