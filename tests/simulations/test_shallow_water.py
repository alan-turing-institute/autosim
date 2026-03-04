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


def test_full_swe_initial_condition_is_balanced_and_non_trivial() -> None:
    """IC should have non-trivial velocity, height anomaly, and 2D structure."""
    out = simulate_swe_2d(
        amp=0.12,
        return_timeseries=True,
        nx=48,
        ny=48,
        Lx=48.0,
        Ly=48.0,
        T=0.0,
        dt_save=1.0,
        cfl=0.12,
        g=9.81,
        H=1.0,
        nu=5e-4,
        drag=2e-3,
    )

    h0, u0, v0 = out[0, ..., 0], out[0, ..., 1], out[0, ..., 2]

    # velocity has spatial structure
    assert u0.std().item() > 1e-4
    assert v0.std().item() > 1e-4
    # height deviates from rest
    assert (h0 - 1.0).abs().max().item() > 1e-5
    # 2D structure: variance differs between rows (not pure zonal stripes)
    row_vars = torch.stack([u0[i].var() for i in range(u0.shape[0])])
    assert row_vars.std().item() > 0
