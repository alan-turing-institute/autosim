import numpy as np
import pytest
import torch

from autosim.experimental.simulations import GrayScott
from autosim.experimental.simulations.gray_scott import (
    _compute_snapshot_count,
    _nonlinear_terms,
    _normalize_field,
    _validate_gaussian_spec,
    simulate_spectral_gray_scott,
)

# --- helper function tests ---


def test_validate_gaussian_spec_valid() -> None:
    spec = {"count": (10, 50), "amplitude": (1.0, 2.0), "width": (100.0, 200.0)}
    result = _validate_gaussian_spec(spec)
    assert result == spec


def test_validate_gaussian_spec_missing_key() -> None:
    with pytest.raises(ValueError, match="missing 'count'"):
        _validate_gaussian_spec({"amplitude": (1.0, 2.0), "width": (100.0, 200.0)})


def test_validate_gaussian_spec_inverted_range() -> None:
    with pytest.raises(ValueError, match="min <= max"):
        _validate_gaussian_spec(
            {"count": (50, 10), "amplitude": (1.0, 2.0), "width": (100.0, 200.0)}
        )


def test_normalize_field_range() -> None:
    field = np.array([[1.0, 3.0], [5.0, 9.0]])
    normed = _normalize_field(field)
    assert float(np.min(normed)) == pytest.approx(0.0)
    assert float(np.max(normed)) == pytest.approx(1.0)


def test_normalize_field_constant() -> None:
    field = np.full((4, 4), 7.0)
    normed = _normalize_field(field)
    assert np.all(normed == 0.0)


def test_compute_snapshot_count() -> None:
    # 100 steps, stride 10 → snapshots at 0,10,20,...,100 = 11
    assert _compute_snapshot_count(total_time=100.0, dt=1.0, snapshot_dt=10.0) == 11


def test_nonlinear_terms_zero_v() -> None:
    u = np.ones((4, 4))
    v = np.zeros((4, 4))
    F, k = 0.04, 0.06
    nu, nv = _nonlinear_terms(u, v, F, k)
    # With v=0: nu = F*(1-u) = 0, nv = -(F+k)*v = 0
    np.testing.assert_allclose(nu, 0.0, atol=1e-15)
    np.testing.assert_allclose(nv, 0.0, atol=1e-15)


# --- constructor validation tests ---


def test_invalid_pattern_raises() -> None:
    with pytest.raises(ValueError, match="Unknown Gray-Scott pattern"):
        GrayScott(pattern="nonexistent")


def test_invalid_initial_condition_raises() -> None:
    with pytest.raises(ValueError, match="initial_condition must be one of"):
        GrayScott(initial_condition="invalid")


def test_negative_n_raises() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        GrayScott(n=-1)


def test_negative_T_raises() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        GrayScott(T=-1.0)


def test_negative_dt_raises() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        GrayScott(dt=-0.1)


# --- simulate_spectral_gray_scott tests ---


def test_simulate_final_frame_shape_and_finite() -> None:
    u, v = simulate_spectral_gray_scott(
        params={"F": 0.04, "k": 0.06},
        return_timeseries=False,
        n=16,
        L=1.0,
        T=5.0,
        dt=1.0,
        snapshot_dt=1.0,
        initial_condition="gaussians",
        gaussian_spec={"count": (2, 5), "amplitude": (1.0, 2.0), "width": (10.0, 50.0)},
        n_fourier_modes=8,
        dealias=True,
        random_seed=42,
    )
    assert u.shape == (16, 16)
    assert v.shape == (16, 16)
    assert np.all(np.isfinite(u))
    assert np.all(np.isfinite(v))


def test_simulate_timeseries_shape() -> None:
    u, v = simulate_spectral_gray_scott(
        params={"F": 0.04, "k": 0.06},
        return_timeseries=True,
        n=16,
        L=1.0,
        T=10.0,
        dt=1.0,
        snapshot_dt=5.0,
        initial_condition="fourier",
        gaussian_spec={"count": (2, 5), "amplitude": (1.0, 2.0), "width": (10.0, 50.0)},
        n_fourier_modes=8,
        dealias=False,
        random_seed=0,
    )
    expected_snapshots = _compute_snapshot_count(10.0, 1.0, 5.0)
    assert u.shape == (expected_snapshots, 16, 16)
    assert v.shape == (expected_snapshots, 16, 16)


def test_simulate_missing_params_raises() -> None:
    with pytest.raises(KeyError, match="must include 'F' and 'k'"):
        simulate_spectral_gray_scott(
            params={"F": 0.04},
            return_timeseries=False,
            n=8,
            L=1.0,
            T=1.0,
            dt=1.0,
            snapshot_dt=1.0,
            initial_condition="gaussians",
            gaussian_spec={
                "count": (2, 5),
                "amplitude": (1.0, 2.0),
                "width": (10.0, 50.0),
            },
            n_fourier_modes=4,
            dealias=False,
        )


# --- GrayScott class parameter range tests ---


def test_pattern_overrides_default_ranges_only() -> None:
    sim = GrayScott(pattern="gliders", fixed_parameters_given_pattern=False)
    assert sim.parameters_range["F"] == (0.013, 0.015)
    assert sim.parameters_range["k"] == (0.053, 0.055)


def test_pattern_does_not_override_user_ranges() -> None:
    custom_range = {
        "F": (0.02, 0.03),
        "k": (0.055, 0.06),
        "delta_u": (2.0e-5, 2.0e-5),
        "delta_v": (1.0e-5, 1.0e-5),
    }
    sim = GrayScott(
        pattern="gliders",
        parameters_range=custom_range,
        fixed_parameters_given_pattern=False,
    )
    assert sim.parameters_range["F"] == custom_range["F"]
    assert sim.parameters_range["k"] == custom_range["k"]


def test_pattern_fixed_params_mode() -> None:
    sim = GrayScott(pattern="gliders", fixed_parameters_given_pattern=True)
    assert sim.parameters_range["F"] == (0.014, 0.014)
    assert sim.parameters_range["k"] == (0.054, 0.054)


def test_defaults_used_when_pattern_and_range_missing() -> None:
    sim = GrayScott()
    assert sim.parameters_range["F"] == (0.014, 0.1)
    assert sim.parameters_range["k"] == (0.051, 0.065)


# --- GrayScott._forward test ---


def test_forward_single_frame() -> None:
    sim = GrayScott(
        n=16,
        L=1.0,
        T=5.0,
        dt=1.0,
        snapshot_dt=1.0,
        return_timeseries=False,
        log_level="warning",
        initial_condition="gaussians",
        random_seed=42,
        parameters_range={
            "F": (0.04, 0.04),
            "k": (0.06, 0.06),
            "delta_u": (2e-5, 2e-5),
            "delta_v": (1e-5, 1e-5),
        },
    )
    x = torch.tensor([[0.04, 0.06, 2e-5, 1e-5]])
    result = sim._forward(x)
    assert result.shape == (1, 16 * 16 * 2)
    assert torch.isfinite(result).all()


def test_forward_min_std_rejects_flat_trajectory() -> None:
    """A very high min_std threshold should cause rejection."""
    sim = GrayScott(
        n=16,
        L=1.0,
        T=5.0,
        dt=1.0,
        snapshot_dt=1.0,
        return_timeseries=False,
        log_level="warning",
        initial_condition="gaussians",
        random_seed=42,
        min_std=999.0,
        parameters_range={
            "F": (0.04, 0.04),
            "k": (0.06, 0.06),
            "delta_u": (2e-5, 2e-5),
            "delta_v": (1e-5, 1e-5),
        },
    )
    x = torch.tensor([[0.04, 0.06, 2e-5, 1e-5]])
    with pytest.raises(ValueError, match="std below threshold"):
        sim._forward(x)


def test_forward_min_std_accepts_valid_trajectory() -> None:
    """A very low min_std threshold should accept any non-trivial trajectory."""
    sim = GrayScott(
        n=16,
        L=1.0,
        T=5.0,
        dt=1.0,
        snapshot_dt=1.0,
        return_timeseries=False,
        log_level="warning",
        initial_condition="gaussians",
        random_seed=42,
        min_std=1e-12,
        parameters_range={
            "F": (0.04, 0.04),
            "k": (0.06, 0.06),
            "delta_u": (2e-5, 2e-5),
            "delta_v": (1e-5, 1e-5),
        },
    )
    x = torch.tensor([[0.04, 0.06, 2e-5, 1e-5]])
    result = sim._forward(x)
    assert result.shape == (1, 16 * 16 * 2)
    assert torch.isfinite(result).all()


def test_forward_timeseries_min_std_rejects() -> None:
    """min_std rejection should also work with timeseries (3D) output."""
    sim = GrayScott(
        n=16,
        L=1.0,
        T=10.0,
        dt=1.0,
        snapshot_dt=5.0,
        return_timeseries=True,
        log_level="warning",
        initial_condition="fourier",
        random_seed=0,
        min_std=999.0,
        parameters_range={
            "F": (0.04, 0.04),
            "k": (0.06, 0.06),
            "delta_u": (2e-5, 2e-5),
            "delta_v": (1e-5, 1e-5),
        },
    )
    x = torch.tensor([[0.04, 0.06, 2e-5, 1e-5]])
    with pytest.raises(ValueError, match="std below threshold"):
        sim._forward(x)


def test_forward_timeseries() -> None:
    sim = GrayScott(
        n=16,
        L=1.0,
        T=10.0,
        dt=1.0,
        snapshot_dt=5.0,
        return_timeseries=True,
        log_level="warning",
        initial_condition="fourier",
        random_seed=0,
        parameters_range={
            "F": (0.04, 0.04),
            "k": (0.06, 0.06),
            "delta_u": (2e-5, 2e-5),
            "delta_v": (1e-5, 1e-5),
        },
    )
    x = torch.tensor([[0.04, 0.06, 2e-5, 1e-5]])
    result = sim._forward(x)
    timesteps = _compute_snapshot_count(10.0, 1.0, 5.0)
    assert result.shape == (1, 2 * timesteps * 16 * 16)
    assert torch.isfinite(result).all()
