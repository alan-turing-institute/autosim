from __future__ import annotations

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from autosim import cli
from autosim.experimental.simulations import ShallowWater2D
from autosim.experimental.simulations.shallow_water import _save_times
from autosim.simulations.base import SpatioTemporalSimulator


def _config(overrides=None, *, hydra=False):
    directory = Path(__file__).resolve().parents[1] / "src/autosim/configs"
    with initialize_config_dir(config_dir=str(directory), version_base=None):
        return compose(
            config_name="generate_data_swe_ae_pretraining_64",
            overrides=overrides or [],
            return_hydra_config=hydra,
        )


def test_pool_matches_baseline_physics_and_preserves_all_frames():
    cfg = _config()
    directory = Path(__file__).resolve().parents[1] / "src/autosim/configs"
    with initialize_config_dir(config_dir=str(directory), version_base=None):
        baseline = compose(
            config_name="generate_data_swe_crps_64",
            overrides=["simulator.T=120.0", "simulator.nu=1e-4"],
        )
    assert OmegaConf.to_container(cfg.simulator) == OmegaConf.to_container(
        baseline.simulator
    )
    sim = cli.build_simulator(cfg.simulator)
    assert isinstance(sim, ShallowWater2D)
    assert sim.dtype is torch.float64
    assert sim.output_names == ["h", "u", "v"]
    times = _save_times(sim.T, sim.dt_save)[sim.skip_nt :]
    assert len(times) == 321
    assert (times[0], times[-1]) == (40.0, 120.0)
    assert (cfg.dataset.n_train, cfg.dataset.n_valid, cfg.dataset.n_test) == (
        1000,
        0,
        0,
    )
    assert cfg.dataset.ensure_exact_n
    assert not cfg.overwrite
    assert not cfg.visualize.enabled
    assert not cfg.visualize.overwrite


def test_pool_seed_and_hydra_output_are_separate(tmp_path):
    cfg = _config([f"dataset.output_dir={tmp_path}"], hydra=True)
    assert cfg.seed == 20280000
    assert (
        cfg.seed > 20261014 + 3 * (SpatioTemporalSimulator._retry_budget(200) + 1) - 1
    )
    assert cfg.seed > 3 * (SpatioTemporalSimulator._retry_budget(200) + 1) - 1
    assert Path(cfg.hydra.run.dir) != tmp_path
    assert "swe_ae_pretraining_runs" in cfg.hydra.run.dir


def test_pool_empty_splits_and_seed_routing(monkeypatch, tmp_path):
    cfg = _config(["simulator.T=0.25", "simulator.skip_nt=0"])
    sim = cli.build_simulator(cfg.simulator)
    monkeypatch.setattr(sim, "_forward", lambda _: torch.ones(1, 2 * 64 * 64 * 3))
    splits = cli.generate_dataset_splits(
        sim,
        n_train=1,
        n_valid=0,
        n_test=0,
        base_seed=cfg.seed,
        ensure_exact_n=True,
    )
    for split, count in (("train", 1), ("valid", 0), ("test", 0)):
        assert splits[split]["data"].shape == (count, 2 if count else 0, 64, 64, 3)
        assert splits[split]["data"].dtype == torch.float32
        assert splits[split]["constant_scalars"].shape == (count, 1)
    cli.save_dataset_splits(splits, tmp_path)
    with pytest.raises(FileExistsError):
        cli.save_dataset_splits(splits, tmp_path)


def test_short_physical_smoke_is_finite_reproducible_and_independent():
    cfg = _config(
        [
            "simulator.T=0.25",
            "simulator.skip_nt=0",
            "simulator.nx=16",
            "simulator.ny=16",
        ]
    )
    original_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        first = cli.generate_dataset_splits(
            cli.build_simulator(cfg.simulator),
            2,
            0,
            0,
            base_seed=cfg.seed,
            ensure_exact_n=True,
        )["train"]["data"]
        repeat = cli.generate_dataset_splits(
            cli.build_simulator(cfg.simulator),
            2,
            0,
            0,
            base_seed=cfg.seed,
            ensure_exact_n=True,
        )["train"]["data"]
    finally:
        torch.set_num_threads(original_threads)
    assert first.shape == (2, 2, 16, 16, 3)
    assert torch.isfinite(first).all()
    assert (first[..., 0] > 0).all()
    assert torch.equal(first, repeat)
    assert not torch.equal(first[0, 0], first[1, 0])


@pytest.mark.parametrize("name", ["stats.yml", "resolved_config.yaml", "train/data.pt"])
def test_generation_protects_existing_metadata_before_simulating(
    tmp_path, monkeypatch, name
):
    existing = tmp_path / name
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_text("preserve this existing file")
    cfg = _config([f"dataset.output_dir={tmp_path}"])

    def forbidden(_):
        pytest.fail("Must reject existing output before simulator construction")

    monkeypatch.setattr(cli, "get_original_cwd", lambda: str(tmp_path))
    monkeypatch.setattr(cli, "build_simulator", forbidden)
    with pytest.raises(FileExistsError, match="Refusing"):
        cli._generate_main.__wrapped__(cfg)
    assert existing.read_text() == "preserve this existing file"
