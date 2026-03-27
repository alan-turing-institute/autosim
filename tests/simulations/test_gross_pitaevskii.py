import pytest
import torch

import autosim.experimental.simulations.gross_pitaevskii as gpe_module
from autosim.experimental.simulations import GrossPitaevskiiEquation2D


@pytest.fixture
def base_simulator():
    """Returns a basic configured GPE simulator for testing."""
    return GrossPitaevskiiEquation2D(
        n=16,
        L=5.0,
        T=0.1,
        dt=0.01,
        log_level="error",
        return_timeseries=False,
    )


def test_initialization():
    """Test simulator initialization and default parameters."""
    sim = GrossPitaevskiiEquation2D(n=32, L=10.0, T=1.0)
    assert sim.n == 32
    assert sim.L == 10.0
    assert sim.T == 1.0


def test_single_forward(base_simulator):
    """Test single sample forward pass."""
    x = torch.tensor(
        [[10.0, 0.5, 0.5, 1.0, 1.0]]  # g, disorder_strength, spoon_strength, wx, wy
    )
    y = base_simulator._forward(x)

    # Output shape should be (1, n*n*channels)
    # channels = 3 (density, real, imag)
    assert y is not None
    assert y.shape == (1, base_simulator.n * base_simulator.n * 3)

    field = y.reshape(base_simulator.n, base_simulator.n, 3)
    density = field[..., 0]
    real = field[..., 1]
    imag = field[..., 2]
    assert torch.allclose(density, real**2 + imag**2, atol=1e-5, rtol=1e-5)


def test_batch_forward(base_simulator):
    """Test batch forward pass."""
    n_samples = 3
    x = base_simulator.sample_inputs(n_samples)
    y, x_valid = base_simulator.forward_batch(x)

    # Expected channels: 3 (density, real, imag)
    assert y.shape == (n_samples, base_simulator.n * base_simulator.n * 3)
    assert x_valid.shape == (n_samples, base_simulator.in_dim)


def test_timeseries_forward():
    """Test spatio-temporal timeseries generation."""
    sim = GrossPitaevskiiEquation2D(
        n=16,
        L=5.0,
        T=0.05,
        dt=0.01,
        snapshot_dt=0.02,
        log_level="error",
        return_timeseries=True,
    )

    res = sim.forward_samples_spatiotemporal(n=2)
    data = res["data"]

    # Shape should be [batch(2), time, x(16), y(16), channels(3)]
    assert data.shape[0] == 2
    assert data.shape[2] == 16
    assert data.shape[3] == 16
    assert data.shape[4] == 3
    assert data.shape[1] > 1  # Should have multiple timesteps (t=0, t=0.02, t=0.04)

    density = data[..., 0]
    real = data[..., 1]
    imag = data[..., 2]
    assert torch.allclose(density, real**2 + imag**2, atol=1e-5, rtol=1e-5)


def test_invalid_parameters():
    """Test initialization parameter validation."""
    with pytest.raises(ValueError, match="Unsupported parameters"):
        GrossPitaevskiiEquation2D(parameters_range={"invalid_param": (0, 1)})


def test_fringe_score_distinguishes_fringing_from_smooth():
    """Fringed fields should score higher than smooth fields."""
    sim = GrossPitaevskiiEquation2D(
        n=64,
        L=10.0,
        T=0.1,
        dt=0.01,
        return_timeseries=False,
        log_level="error",
    )

    x = torch.linspace(-1.0, 1.0, 64)
    X, Y = torch.meshgrid(x, x, indexing="ij")
    smooth = torch.exp(-4.0 * (X**2 + Y**2))
    stripes = smooth + 0.2 * torch.sin(2.0 * torch.pi * 20.0 * X)

    smooth_score = sim._fringe_score_density(smooth)
    stripes_score = sim._fringe_score_density(stripes)

    assert stripes_score > smooth_score


def test_artifact_validation_rejects_high_fringe_score_trajectory(
    monkeypatch: pytest.MonkeyPatch,
):
    """Artifact validation should reject trajectories with strong fringing."""

    def _fake_simulate_gpe_2d(**kwargs):
        n = int(kwargs["n"])
        t = 6
        x = torch.linspace(-1.0, 1.0, n)
        X, _Y = torch.meshgrid(x, x, indexing="ij")
        density = 0.1 + 0.05 * torch.sin(2.0 * torch.pi * 18.0 * X)
        density = density.clamp(min=1e-6)
        real = torch.sqrt(density)
        imag = torch.zeros_like(real)
        frame = torch.stack([density, real, imag], dim=-1)
        return torch.stack([frame] * t, dim=0)

    monkeypatch.setattr(gpe_module, "simulate_gpe_2d", _fake_simulate_gpe_2d)

    sim = GrossPitaevskiiEquation2D(
        n=64,
        L=10.0,
        T=0.1,
        dt=0.01,
        return_timeseries=True,
        log_level="error",
        artifact_validation_enabled=True,
        artifact_validation_threshold=0.01,
        artifact_validation_warmup_frames=0,
        artifact_validation_tail_frames=6,
    )

    x = sim.sample_inputs(1, random_seed=0)
    with pytest.raises(ValueError, match="fringe artifact score"):
        sim._forward(x)
