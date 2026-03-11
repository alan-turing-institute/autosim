import pytest
import torch

from autosim.experimental.simulations import ShallowWater2D
from autosim.experimental.simulations.shallow_water import simulate_swe_2d


def _robust_zscore(values: torch.Tensor) -> torch.Tensor:
    median = values.median()
    mad = (values - median).abs().median()
    scale = 1.4826 * mad + 1e-12
    return (values - median) / scale


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


def test_full_swe_posthoc_outlier_check_no_extreme_runs() -> None:
    """Posthoc sanity check: no extreme per-run mean/variance outliers.

    This test is intentionally lightweight (small grid and horizon), but mirrors
    the posthoc dataset check by computing robust z-scores on trajectory-level
    mean and variance.
    """
    sim = ShallowWater2D(
        return_timeseries=True,
        log_level="warning",
        nx=20,
        ny=20,
        Lx=20.0,
        Ly=20.0,
        T=6.0,
        dt_save=0.5,
        cfl=0.12,
        parameters_range={
            "amp": (0.08, 0.12),
            "h_mean": (0.9, 1.2),
            "drag": (1.2e-3, 3.0e-3),
            "nu": (3.0e-4, 7.0e-4),
        },
    )

    out = sim.forward_samples_spatiotemporal(n=24, random_seed=0, ensure_exact_n=True)
    data = out["data"].float()

    per_run_mean = data.mean(dim=(1, 2, 3, 4))
    per_run_var = data.var(dim=(1, 2, 3, 4), unbiased=False)

    z_mean = _robust_zscore(per_run_mean)
    z_var = _robust_zscore(per_run_var)
    flagged = (z_mean.abs() > 3.5) | (z_var.abs() > 3.5)

    assert flagged.sum().item() == 0
