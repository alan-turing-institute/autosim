import pytest
import torch

from autosim.experimental.simulations.lattice_boltzmann import (
    LatticeBoltzmann,
    simulate_lbm_cylinder,
)


def test_default_output_names_ordering() -> None:
    sim = LatticeBoltzmann()
    assert sim.output_names == ["vorticity", "velocity_x", "velocity_y", "rho"]


def test_n_saved_frames_validation() -> None:
    with pytest.raises(ValueError, match="n_saved_frames must be positive"):
        LatticeBoltzmann(n_saved_frames=0)


def test_dt_validation() -> None:
    with pytest.raises(ValueError, match="dt must be positive"):
        LatticeBoltzmann(dt=0.0)


def test_timeseries_length_changes_with_duration_when_unsampled() -> None:
    params = torch.tensor([0.02, 0.06], dtype=torch.float32)

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
    params = torch.tensor([0.02, 0.06], dtype=torch.float32)

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


def test_snapshot_channel_order_is_vorticity_velocityx_velocityy_rho() -> None:
    params = torch.tensor([0.02, 0.06], dtype=torch.float32)

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
    params = torch.tensor([0.02, 0.06], dtype=torch.float32)

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
