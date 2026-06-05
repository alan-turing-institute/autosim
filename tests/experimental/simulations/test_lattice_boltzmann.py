import pytest
import torch

from autosim.experimental.simulations.lattice_boltzmann import (
    LatticeBoltzmann,
    simulate_lbm_cylinder,
)

DEFAULT_RANGE = {
    "viscosity": (0.02, 0.02),
    "u_in": (0.06, 0.06),
    "oscillation_frequency": (1.0, 1.0),
}


def test_default_output_names_ordering() -> None:
    sim = LatticeBoltzmann()
    assert sim.output_names == ["vorticity", "velocity_x", "velocity_y", "rho"]


def test_n_saved_frames_validation() -> None:
    with pytest.raises(ValueError, match="n_saved_frames must be positive"):
        LatticeBoltzmann(n_saved_frames=0)


def test_dt_validation() -> None:
    with pytest.raises(ValueError, match="dt must be positive"):
        LatticeBoltzmann(dt=0.0)


def test_skip_nt_validation() -> None:
    with pytest.raises(ValueError, match="skip_nt must be non-negative"):
        LatticeBoltzmann(skip_nt=-1)


def test_oscillation_frequency_validation() -> None:
    with pytest.raises(
        ValueError, match="oscillation_frequency range must be non-negative"
    ):
        LatticeBoltzmann(
            parameters_range={
                "viscosity": (0.01, 0.03),
                "u_in": (0.05, 0.15),
                "oscillation_frequency": (-0.1, 1.0),
            }
        )


def test_timeseries_length_changes_with_duration_when_unsampled() -> None:
    params = torch.tensor([0.02, 0.06, 1.0], dtype=torch.float32)

    traj_short = simulate_lbm_cylinder(
        params=params,
        return_timeseries=True,
        width=24,
        height=12,
        duration=0.12,
        dt=1.0 / 250.0,
        use_cylinder=False,
        oscillatory_inlet=False,
        n_saved_frames=None,
    )
    traj_long = simulate_lbm_cylinder(
        params=params,
        return_timeseries=True,
        width=24,
        height=12,
        duration=0.24,
        dt=1.0 / 250.0,
        use_cylinder=False,
        oscillatory_inlet=False,
        n_saved_frames=None,
    )

    assert traj_long.shape[0] > traj_short.shape[0]


def test_n_saved_frames_downsamples_timeseries() -> None:
    params = torch.tensor([0.02, 0.06, 1.0], dtype=torch.float32)

    traj_full = simulate_lbm_cylinder(
        params=params,
        return_timeseries=True,
        width=24,
        height=12,
        duration=0.2,
        dt=1.0 / 250.0,
        use_cylinder=False,
        oscillatory_inlet=False,
        n_saved_frames=None,
    )
    traj_sparse = simulate_lbm_cylinder(
        params=params,
        return_timeseries=True,
        width=24,
        height=12,
        duration=0.2,
        dt=1.0 / 250.0,
        use_cylinder=False,
        oscillatory_inlet=False,
        n_saved_frames=5,
    )

    assert traj_sparse.shape[0] < traj_full.shape[0]


def test_n_saved_frames_exact_count_when_available() -> None:
    params = torch.tensor([0.02, 0.06, 1.0], dtype=torch.float32)

    traj = simulate_lbm_cylinder(
        params=params,
        return_timeseries=True,
        width=24,
        height=12,
        duration=2.0,
        dt=1.0 / 250.0,
        use_cylinder=False,
        oscillatory_inlet=False,
        n_saved_frames=321,
    )

    assert traj.shape[0] == 321


def test_n_saved_frames_capped_by_available_steps() -> None:
    params = torch.tensor([0.02, 0.06, 1.0], dtype=torch.float32)

    duration = 0.2
    dt = 1.0 / 250.0
    available_steps = int(duration / dt)

    traj = simulate_lbm_cylinder(
        params=params,
        return_timeseries=True,
        width=24,
        height=12,
        duration=duration,
        dt=dt,
        use_cylinder=False,
        oscillatory_inlet=False,
        n_saved_frames=1000,
    )

    assert traj.shape[0] == available_steps


def test_snapshot_channel_order_is_vorticity_velocityx_velocityy_rho() -> None:
    params = torch.tensor([0.02, 0.06, 1.0], dtype=torch.float32)

    snapshot = simulate_lbm_cylinder(
        params=params,
        return_timeseries=False,
        width=24,
        height=12,
        duration=0.1,
        dt=1.0 / 250.0,
        use_cylinder=False,
        oscillatory_inlet=False,
        n_saved_frames=None,
    )

    assert snapshot.shape[-1] == 4
    assert not torch.allclose(snapshot[..., 0], snapshot[..., 1])
    assert not torch.allclose(snapshot[..., 0], snapshot[..., 3])


def test_smaller_dt_yields_longer_timeseries_for_fixed_duration() -> None:
    params = torch.tensor([0.02, 0.06, 1.0], dtype=torch.float32)

    traj_coarse = simulate_lbm_cylinder(
        params=params,
        return_timeseries=True,
        width=24,
        height=12,
        duration=1.2,
        dt=1.0 / 125.0,
        use_cylinder=False,
        oscillatory_inlet=False,
        n_saved_frames=None,
    )
    traj_fine = simulate_lbm_cylinder(
        params=params,
        return_timeseries=True,
        width=24,
        height=12,
        duration=1.2,
        dt=1.0 / 500.0,
        use_cylinder=False,
        oscillatory_inlet=False,
        n_saved_frames=None,
    )

    assert traj_fine.shape[0] > traj_coarse.shape[0]


def test_oscillation_frequency_changes_dynamics() -> None:
    params_low = torch.tensor([0.02, 0.08, 0.5], dtype=torch.float32)
    params_high = torch.tensor([0.02, 0.08, 2.0], dtype=torch.float32)

    traj_low = simulate_lbm_cylinder(
        params=params_low,
        return_timeseries=True,
        width=24,
        height=12,
        duration=1.0,
        dt=1.0 / 250.0,
        use_cylinder=False,
        oscillatory_inlet=True,
        n_saved_frames=40,
    )
    traj_high = simulate_lbm_cylinder(
        params=params_high,
        return_timeseries=True,
        width=24,
        height=12,
        duration=1.0,
        dt=1.0 / 250.0,
        use_cylinder=False,
        oscillatory_inlet=True,
        n_saved_frames=40,
    )

    assert not torch.allclose(traj_low, traj_high)


def test_forward_samples_spatiotemporal_applies_skip_nt(monkeypatch) -> None:
    sim = LatticeBoltzmann(
        return_timeseries=True,
        width=4,
        height=3,
        skip_nt=2,
        parameters_range=DEFAULT_RANGE,
    )

    n_samples = 1
    n_steps = 7
    channels = len(sim.output_names)
    features_per_frame = sim.width * sim.height * channels
    fake_y = torch.arange(
        float(n_samples * n_steps * features_per_frame), dtype=torch.float32
    ).reshape(n_samples, n_steps * features_per_frame)
    fake_x = torch.tensor([[0.02, 0.06, 1.0]], dtype=torch.float32)

    def _sample_inputs(*_args, **_kwargs):
        return fake_x

    def _forward(_x):
        return fake_y

    monkeypatch.setattr(sim, "sample_inputs", _sample_inputs)
    monkeypatch.setattr(sim, "_forward", _forward)

    out = sim.forward_samples_spatiotemporal(n=1)
    assert out["data"].shape == (
        1,
        n_steps - sim.skip_nt,
        sim.height,
        sim.width,
        channels,
    )


def test_forward_samples_spatiotemporal_raises_when_skip_nt_too_large(
    monkeypatch,
) -> None:
    sim = LatticeBoltzmann(
        return_timeseries=True,
        width=4,
        height=3,
        skip_nt=10,
        parameters_range=DEFAULT_RANGE,
    )

    n_samples = 1
    n_steps = 5
    channels = len(sim.output_names)
    features_per_frame = sim.width * sim.height * channels
    fake_y = torch.zeros((n_samples, n_steps * features_per_frame), dtype=torch.float32)
    fake_x = torch.tensor([[0.02, 0.06, 1.0]], dtype=torch.float32)

    def _sample_inputs(*_args, **_kwargs):
        return fake_x

    def _forward(_x):
        return fake_y

    monkeypatch.setattr(sim, "sample_inputs", _sample_inputs)
    monkeypatch.setattr(sim, "_forward", _forward)

    with pytest.raises(ValueError, match="skip_nt is too large"):
        sim.forward_samples_spatiotemporal(n=1)
