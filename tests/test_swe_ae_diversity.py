from __future__ import annotations

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from autosim import cli
from autosim.cli import build_simulator, generate_dataset_splits, save_dataset_splits
from autosim.experimental.simulations import ShallowWater2D
from autosim.experimental.simulations.shallow_water import _save_times
from autosim.simulations.base import SpatioTemporalSimulator


@pytest.fixture
def diversity_config() -> DictConfig:
    config_dir = Path(__file__).resolve().parents[1] / "src/autosim/configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name="generate_data_swe_ae_diversity_64")


def test_diversity_hydra_records_do_not_reuse_dataset_directory(tmp_path: Path) -> None:
    config_dir = Path(__file__).resolve().parents[1] / "src/autosim/configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="generate_data_swe_ae_diversity_64",
            overrides=[f"dataset.output_dir={tmp_path}"],
            return_hydra_config=True,
        )
        assert cfg.dataset.output_dir == str(tmp_path)
        assert Path(cfg.hydra.run.dir) != tmp_path
        assert "swe_ae_diversity_runs/" in cfg.hydra.run.dir
        unresolved = OmegaConf.to_container(cfg.hydra.run, resolve=False)
        assert isinstance(unresolved, dict)
        assert "${shortuuid:7}" in unresolved["dir"]


def test_diversity_config_matches_original_swe64_physics(
    diversity_config: DictConfig,
) -> None:
    # Pin every resolved simulator field from the original matched dataset.
    expected = {
        "_target_": "autosim.experimental.simulations.ShallowWater2D",
        "return_timeseries": True,
        "return_additional_input_fields": False,
        "return_energy_budget": False,
        "log_level": "warning",
        "nx": 64,
        "ny": 64,
        "Lx": 6.283185307179586,
        "Ly": 6.283185307179586,
        "T": 120.0,
        "dt_save": 0.25,
        "skip_nt": 160,
        "cfl": 0.12,
        "g": 9.81,
        "h_mean": 1.0,
        "f0": 1.0,
        "beta": 0.5,
        "nu": 1.0e-4,
        "drag": 0.05,
        "coriolis_mode": "periodic_beta",
        "dealias": True,
        "initial_condition": "balanced_random_pv",
        "initial_wavenumber": 4.0,
        "initial_bandwidth": 1.5,
        "forcing_type": "vortical",
        "forcing_energy_rate": 1.5e-3,
        "forcing_wavenumber": 3.0,
        "forcing_bandwidth": 0.7,
        "forcing_correlation_time": 0.0,
        "forcing_backscatter_fraction": 0.0,
        "backscatter_include_drag": False,
        "parameters_range": {"amp": [0.1, 0.1]},
    }
    assert OmegaConf.to_container(diversity_config.simulator, resolve=True) == expected

    sim = build_simulator(diversity_config.simulator)
    assert isinstance(sim, ShallowWater2D)
    assert sim.dtype is torch.float64
    assert sim.output_names == ["h", "u", "v"]
    times = _save_times(sim.T, sim.dt_save)[sim.skip_nt :]
    assert len(times) == 321
    assert (times[0], times[-1]) == (40.0, 120.0)
    assert sim.out_dim == 3


def test_diversity_config_is_additive_and_has_fresh_seeds(
    diversity_config: DictConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = diversity_config
    assert (cfg.dataset.n_train, cfg.dataset.n_valid, cfg.dataset.n_test) == (200, 0, 0)
    assert cfg.dataset.ensure_exact_n is True
    assert cfg.overwrite is False
    assert cfg.visualize.enabled is False
    assert cfg.visualize.split == "train"
    assert cfg.visualize.overwrite is False
    assert cfg.stratify.enabled is False
    assert list(cfg.normalization.shared_core_field_groups) == []
    unresolved_dataset = OmegaConf.to_container(cfg.dataset, resolve=False)
    assert isinstance(unresolved_dataset, dict)
    unresolved_path = unresolved_dataset["output_dir"]
    assert "${shortuuid:7}" in unresolved_path
    assert "swe_stochastic_64_ae_diversity_" in cfg.dataset.output_dir
    assert not Path(cfg.dataset.output_dir).is_absolute()

    original_stride = SpatioTemporalSimulator._retry_budget(200) + 1
    assert original_stride == 4001
    assert cfg.seed == 20261014
    assert cfg.seed > 3 * original_stride - 1

    sim = build_simulator(cfg.simulator)
    calls = []

    def record_split(n, random_seed=None, ensure_exact_n=False):
        calls.append((n, random_seed, ensure_exact_n))
        return {"data": torch.zeros(n, 1, 1, 1, 3)}

    monkeypatch.setattr(sim, "forward_samples_spatiotemporal", record_split)
    generate_dataset_splits(
        sim,
        **{key: cfg.dataset[key] for key in ("n_train", "n_valid", "n_test")},
        base_seed=cfg.seed,
        ensure_exact_n=cfg.dataset.ensure_exact_n,
    )
    assert calls == [
        (200, 20261014, True),
        (0, 20265015, True),
        (0, 20269016, True),
    ]


def test_diversity_empty_splits_without_running_the_solver(
    diversity_config: DictConfig, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Exercise real sampling/reshaping/saving with a cheap forward stub.
    # Neither a physical simulation nor a production dataset is generated.
    cfg = diversity_config
    cfg.simulator.T = 0.25
    cfg.simulator.skip_nt = 0
    sim = build_simulator(cfg.simulator)
    calls = []

    def fake_forward(x):
        calls.append(x.clone())
        return torch.full((1, 2 * 64 * 64 * 3), 0.1, dtype=torch.float32)

    monkeypatch.setattr(sim, "_forward", fake_forward)
    splits = generate_dataset_splits(
        sim, n_train=1, n_valid=0, n_test=0, base_seed=cfg.seed, ensure_exact_n=True
    )
    assert len(calls) == 1
    torch.testing.assert_close(calls[0], torch.tensor([[0.1]]))
    for split, count in (("train", 1), ("valid", 0), ("test", 0)):
        # Empty splits have no trajectories or retained frames.
        assert splits[split]["data"].shape == (count, 2 if count else 0, 64, 64, 3)
        assert splits[split]["constant_scalars"].shape == (count, 1)
        assert splits[split]["data"].dtype is torch.float32
        assert splits[split]["additional_input_fields"] is None
        assert splits[split]["energy_budget"] is None

    save_dataset_splits(splits, tmp_path, overwrite=cfg.overwrite)
    original_bytes = {
        split: (tmp_path / split / "data.pt").read_bytes() for split in splits
    }
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        save_dataset_splits(splits, tmp_path, overwrite=cfg.overwrite)
    for split in splits:
        assert (tmp_path / split / "data.pt").read_bytes() == original_bytes[split]


@pytest.mark.parametrize(
    "existing_file",
    [
        "resolved_config.yaml",
        "stats.yml",
        "train/data.pt",
        "valid/data.pt",
        "test/data.pt",
    ],
)
def test_generation_refuses_existing_outputs_before_any_work(
    diversity_config: DictConfig,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    existing_file: str,
) -> None:
    cfg = diversity_config
    cfg.dataset.output_dir = str(tmp_path)
    protected = tmp_path / existing_file
    protected.parent.mkdir(parents=True, exist_ok=True)
    protected.write_text("original bytes", encoding="utf-8")

    def unexpected_call(*args, **kwargs):
        pytest.fail("Generation or file writing ran before the overwrite check")

    monkeypatch.setattr(cli, "build_simulator", unexpected_call)
    monkeypatch.setattr(cli, "save_resolved_config", unexpected_call)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        cli._generate_main.__wrapped__(cfg)
    assert protected.read_text(encoding="utf-8") == "original bytes"
