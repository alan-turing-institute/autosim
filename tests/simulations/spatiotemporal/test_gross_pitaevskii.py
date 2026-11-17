import math
from pathlib import Path

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

import autosim.simulations.spatiotemporal.gross_pitaevskii as gpe_module
from autosim.simulations.spatiotemporal import GrossPitaevskiiEquation2D


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
    """Fringed box-trapped fields should score higher than clean ones."""
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
    r = torch.sqrt(X**2 + Y**2)
    # Box-trapped BEC: flat interior, sharp wall, dead zone outside
    bec = 1.0 / (1.0 + torch.exp((r - 0.8) / 0.03))
    fringed = bec + 0.03 * torch.sin(2.0 * torch.pi * 6.0 * X)

    clean_score = sim._fringe_score_density(bec)
    fringed_score = sim._fringe_score_density(fringed)

    assert fringed_score > clean_score


@pytest.mark.parametrize(
    ("shape", "modes", "expected"),
    [
        ((64, 64), (25, 0), 0.0),
        ((64, 64), (26, 0), 1.0),
        ((64, 64), (18, 18), 0.0),
        ((64, 64), (19, 19), 1.0),
        ((33, 33), (13, 0), 0.0),
        ((33, 33), (14, 0), 1.0),
        ((64, 32), (25, 0), 0.0),
        ((64, 32), (0, 13), 1.0),
    ],
)
def test_fringe_score_classifies_fourier_modes(shape, modes, expected):
    rows = torch.arange(shape[0], dtype=torch.float64) / shape[0]
    cols = torch.arange(shape[1], dtype=torch.float64) / shape[1]
    phase = 2 * math.pi * (modes[0] * rows[:, None] + modes[1] * cols[None, :])
    density = 1.0 + 0.1 * torch.cos(phase)

    score = GrossPitaevskiiEquation2D._fringe_score_density(density)

    assert score == pytest.approx(expected, abs=1e-6)


def test_fringe_score_measures_spectral_energy_fraction():
    rows = torch.arange(64, dtype=torch.float64) / 64
    low_mode = 0.1 * torch.cos(2 * math.pi * 3 * rows)
    high_mode = 0.2 * torch.cos(2 * math.pi * 28 * rows)
    density = (1.0 + low_mode + high_mode)[:, None].expand(64, 64)

    score = GrossPitaevskiiEquation2D._fringe_score_density(density)

    # Orthogonal cosine modes have energies proportional to amplitude squared.
    assert score == pytest.approx(0.8, abs=1e-6)


def test_artifact_validation_rejects_high_fringe_score_trajectory(
    monkeypatch: pytest.MonkeyPatch,
):
    """Artifact validation should reject trajectories with strong fringing."""

    def _fake_simulate_gpe_2d(**kwargs):
        n = int(kwargs["n"])
        t = 6
        x = torch.linspace(-1.0, 1.0, n)
        X, _Y = torch.meshgrid(x, x, indexing="ij")
        r = torch.sqrt(X**2 + _Y**2)
        # Box-trapped BEC with high-frequency grid aliasing noise (near Nyquist)
        bec = 1.0 / (1.0 + torch.exp((r - 0.8) / 0.03))
        density = (bec + 0.5 * torch.sin(2.0 * torch.pi * 15.0 * X)).clamp(min=1e-6)
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
        artifact_validation_threshold=0.005,
        artifact_validation_warmup_frames=0,
        artifact_validation_tail_frames=6,
    )

    x = sim.sample_inputs(1, random_seed=0)
    with pytest.raises(ValueError, match="fringe artifact score"):
        sim._forward(x)


@pytest.mark.parametrize(
    ("n", "dtype"), [(64, torch.float32), (65, torch.float32), (64, torch.float64)]
)
@pytest.mark.parametrize("angle", [0.0, 0.001, -0.1, math.pi])
def test_rotation_preserves_angular_momentum_eigenstate(n, dtype, angle):
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        sim = gpe_module.GPESimulator2D(N=n, L=16.0, device=torch.device("cpu"))
        psi = ((sim.X + 1j * sim.Y) * torch.exp(-(sim.X**2 + sim.Y**2) / 2)).to(
            torch.complex64
        )
        rotated = sim._apply_rotation(psi, angle)
        expected = psi * complex(math.cos(angle), -math.sin(angle))
        torch.testing.assert_close(rotated, expected, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(
            rotated.abs().square().sum(), psi.abs().square().sum(), rtol=2e-6, atol=0
        )
    finally:
        torch.set_default_dtype(original_dtype)


@pytest.mark.parametrize("dt", [0.01, 0.005, 0.0025])
def test_rotating_harmonic_oscillator_matches_analytic_density(dt):
    sim = gpe_module.GPESimulator2D(N=64, L=10.0, dt=dt, device=torch.device("cpu"))
    ground = torch.exp(-(sim.X**2 + sim.Y**2) / 2).to(torch.complex64)
    vortex = (sim.X + 1j * sim.Y) * ground
    psi = ground + vortex
    psi /= torch.sqrt(psi.abs().square().sum() * sim.dx**2)
    potential = (sim.X**2 + sim.Y**2) / 2
    omega = 0.85
    final_time = 0.2
    for _ in range(round(final_time / dt)):
        psi = sim.step(psi, potential, g=0.0, Omega=omega)

    # Oscillator energies are 1 and 2, with Lz eigenvalues 0 and 1.
    # Keep the existing real-time convention H = H_osc + Omega * Lz.
    expected = ground * complex(math.cos(final_time), -math.sin(final_time))
    phase = (2 + omega) * final_time
    expected += vortex * complex(math.cos(phase), -math.sin(phase))
    expected /= torch.sqrt(expected.abs().square().sum() * sim.dx**2)
    relative_error = torch.linalg.vector_norm(
        psi.abs().square() - expected.abs().square()
    )
    relative_error /= torch.linalg.vector_norm(expected.abs().square())
    assert relative_error < 1e-4


@pytest.mark.parametrize(
    ("snapshot_dt", "times"),
    [
        (0.005, [0.0, 0.005, 0.01, 0.015, 0.02, 0.025]),
        (0.015, [0.0, 0.015, 0.025]),
        (0.05, [0.0, 0.025]),
    ],
)
def test_snapshots_match_requested_times(snapshot_dt, times):
    history = gpe_module.simulate_gpe_2d(
        {"g": 0.0},
        n=64,
        L=10.0,
        T=0.025,
        dt=0.01,
        snapshot_dt=snapshot_dt,
        return_timeseries=True,
        random_seed=0,
    )
    assert history.shape[0] == len(times)
    initial = torch.complex(history[0, ..., 1], history[0, ..., 2])
    for frame, time in zip(history, times, strict=True):
        expected = initial * complex(math.cos(time), -math.sin(time))
        actual = torch.complex(frame[..., 1], frame[..., 2])
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)


def test_final_state_stops_at_requested_time():
    sim = gpe_module.GPESimulator2D(N=64, L=10.0, device=torch.device("cpu"))
    initial = sim.initialize_state()
    final = gpe_module.simulate_gpe_2d(
        {"g": 0.0},
        n=64,
        L=10.0,
        T=0.025,
        dt=0.01,
        random_seed=0,
    )
    expected = initial * complex(math.cos(0.025), -math.sin(0.025))
    torch.testing.assert_close(
        torch.complex(final[..., 1], final[..., 2]), expected, atol=3e-6, rtol=3e-5
    )


@pytest.mark.parametrize(
    ("total_time", "dt", "snapshot_dt", "n_frames"),
    [
        (0.9, 0.1, 0.3, 4),
        (0.45, 0.05, 0.15, 4),
        (2.1, 0.1, 0.7, 4),
        (0.900001, 0.1, 0.3, 5),
    ],
)
def test_final_snapshot_count_handles_roundoff(total_time, dt, snapshot_dt, n_frames):
    config = {"g": 0.0}
    history = gpe_module.simulate_gpe_2d(
        config,
        n=16,
        L=10.0,
        T=total_time,
        dt=dt,
        snapshot_dt=snapshot_dt,
        return_timeseries=True,
        random_seed=0,
    )
    assert history.shape[0] == n_frames
    final = gpe_module.simulate_gpe_2d(
        config,
        n=16,
        L=10.0,
        T=total_time,
        dt=dt,
        random_seed=0,
    )
    torch.testing.assert_close(history[-1], final)


@pytest.mark.parametrize("trap_omega", [None, -0.3])
def test_exact_trap_rotation_preserves_reusable_config(trap_omega):
    config = {"g": 0.0, "wx": 1.0, "wy": 2.0, "Omega": 0.8, "exact_trap_rotation": True}
    if trap_omega is not None:
        config["trap_Omega"] = trap_omega
    original = config.copy()
    gpe_module.simulate_gpe_2d(config, n=32, T=0.1, dt=0.01, random_seed=0)
    assert config == original

    config["exact_trap_rotation"] = False
    actual = gpe_module.simulate_gpe_2d(config, n=32, T=0.1, dt=0.01, random_seed=0)
    expected = gpe_module.simulate_gpe_2d(
        {**original, "exact_trap_rotation": False},
        n=32,
        T=0.1,
        dt=0.01,
        random_seed=0,
    )
    torch.testing.assert_close(actual, expected)


@pytest.fixture
def prepared_state():
    config = {"g": 0.0, "width": 1.5, "imaginary_time": True, "initial_noise": 0.0}
    field = gpe_module.simulate_gpe_2d(
        config,
        n=32,
        L=10.0,
        T=0.04,
        dt=0.02,
        random_seed=0,
    )
    return config, field


def test_initial_snapshot_is_prepared_state(prepared_state):
    config, field = prepared_state
    history = gpe_module.simulate_gpe_2d(
        {
            **config,
            "imaginary_time": False,
            "imaginary_time_steps": 2,
            "imaginary_time_dt": 0.02,
        },
        n=32,
        L=10.0,
        T=0.01,
        dt=0.01,
        return_timeseries=True,
        random_seed=0,
    )
    torch.testing.assert_close(history[0], field)


@pytest.mark.parametrize("imaginary_time", [False, True])
def test_evolution_after_preparation_uses_requested_timestep(
    monkeypatch, prepared_state, imaginary_time
):
    config, field = prepared_state
    config = {**config, "imaginary_time": imaginary_time}
    actual = gpe_module.simulate_gpe_2d(
        {**config, "imaginary_time_steps": 2, "imaginary_time_dt": 0.02},
        n=32,
        L=10.0,
        T=0.01,
        dt=0.01,
        random_seed=0,
    )
    psi = torch.complex(field[..., 1], field[..., 2])
    monkeypatch.setattr(
        gpe_module.GPESimulator2D,
        "initialize_state",
        lambda _self, **kwargs: psi.clone(),
    )
    expected = gpe_module.simulate_gpe_2d(
        config,
        n=32,
        L=10.0,
        T=0.01,
        dt=0.01,
        random_seed=0,
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(("wx", "wy"), [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0)])
def test_unconfined_disorder_has_unwindowed_automatic_radius(wx, wy):
    sim = gpe_module.GPESimulator2D(N=32, L=10.0, device=torch.device("cpu"))
    config = {"wx": wx, "wy": wy, "disorder_strength": 1.0}
    actual = gpe_module.generate_static_disorder(
        sim.X, sim.Y, config, torch.Generator().manual_seed(42)
    )
    expected = gpe_module.generate_static_disorder(
        sim.X,
        sim.Y,
        {**config, "disorder_radius": 1e12},
        torch.Generator().manual_seed(42),
    )
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected)


def test_laser_only_config_bypasses_wavefunction_rotation(monkeypatch):
    config_path = Path(gpe_module.__file__).parents[2] / (
        "configs/simulator/spatiotemporal/gpe/laser_only_wake.yaml"
    )
    config = OmegaConf.load(config_path)
    config.parameters_range.imaginary_time_steps = [2, 2]
    sim = instantiate(
        config,
        n=32,
        T=0.025,
        snapshot_dt=0.01,
        skip_nt=0,
        artifact_validation_enabled=False,
        log_level="error",
    )

    def unexpected_rotation(*args, **kwargs):
        pytest.fail("Laser-only configuration must not rotate the wavefunction")

    monkeypatch.setattr(
        gpe_module.GPESimulator2D, "_apply_rotation", unexpected_rotation
    )
    result = sim.forward_samples_spatiotemporal(1, random_seed=42, ensure_exact_n=True)
    data = result["data"]
    assert data.shape == (1, 3, 32, 32, 3)
    norms = data[..., 0].sum(dim=(-1, -2)) * (sim.L / sim.n) ** 2
    torch.testing.assert_close(norms, torch.ones_like(norms))


@pytest.mark.parametrize(
    ("return_timeseries", "total_time", "snapshot_dt", "skip_nt", "n_time"),
    [
        (False, 0.025, 0.015, 0, 1),
        (True, 0.025, 0.015, 0, 2),
        (True, 0.025, 0.015, 1, 1),
        (True, 0.9, 0.3, 0, 3),
        (True, 0.900001, 0.3, 0, 4),
    ],
)
def test_all_rejected_samples_return_empty_spatiotemporal_data(
    monkeypatch, return_timeseries, total_time, snapshot_dt, skip_nt, n_time
):
    sim = GrossPitaevskiiEquation2D(
        n=16,
        L=10.0,
        T=total_time,
        dt=0.01,
        snapshot_dt=snapshot_dt,
        skip_nt=skip_nt,
        return_timeseries=return_timeseries,
        artifact_validation_enabled=True,
        log_level="error",
    )
    monkeypatch.setattr(sim, "_max_fringe_score", lambda _sol: float("inf"))
    result = sim.forward_samples_spatiotemporal(2, random_seed=0)
    assert result["data"].shape == (0, n_time, 16, 16, 3)
    assert result["constant_scalars"].shape == (0, sim.in_dim)
    assert result["constant_fields"] is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.0, False), (0.1, False), (0.49, False), (0.51, True), (0.9, True), (1.0, True)],
)
def test_sampled_disorder_flag_uses_boolean_conversion(value, expected):
    sim = GrossPitaevskiiEquation2D(
        parameters_range={"disorder_time_dependent": (0.0, 1.0)},
        n=16,
        T=0.01,
        dt=0.01,
        log_level="error",
    )
    config = sim._extract_run_parameters(torch.tensor([[value]]))
    assert config["disorder_time_dependent"] is expected
