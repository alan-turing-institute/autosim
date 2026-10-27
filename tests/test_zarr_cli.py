from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from hydra import compose, initialize_config_module
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import InterpolationKeyError

from autosim import cli


def test_full_size_preset_counts_without_generating_data() -> None:
    with initialize_config_module(config_module="autosim.configs", version_base=None):
        cfg = compose(config_name="generate_data_era5_full")
    assert (cfg.dataset.n_train, cfg.dataset.n_valid, cfg.dataset.n_test) == (1, 1, 1)
    assert dict(cfg.dataset.steps_per_split) == {
        "train": 58440,
        "valid": 1460,
        "test": 1464,
    }
    frame_bytes = (
        cfg.simulator.height * cfg.simulator.width * cfg.simulator.n_channels * 4
    )
    assert frame_bytes * cfg.dataset.steps_per_split.train == 468398937600
    assert frame_bytes * cfg.dataset.shard_steps <= cfg.simulator.max_output_bytes
    assert cfg.dataset.format == "zarr"
    assert not cfg.visualize.enabled


def test_full_size_preset_can_export_small_split_overrides(tmp_path: Path) -> None:
    zarr = pytest.importorskip("zarr")
    output_dir = tmp_path / "dataset"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "autosim.cli",
            "--config-name=generate_data_era5_full",
            f"dataset.output_dir={output_dir}",
            "simulator.height=3",
            "simulator.width=5",
            "simulator.n_channels=2",
            "simulator.rho=0.8",
            "dataset.steps_per_split.train=11",
            "dataset.steps_per_split.valid=6",
            "dataset.steps_per_split.test=7",
            "dataset.shard_steps=4",
            "dataset.chunk_steps=2",
            f"hydra.run.dir={tmp_path / 'hydra'}",
            "hydra.output_subdir=null",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    for split, steps in (("train", 11), ("valid", 6), ("test", 7)):
        group = zarr.open_group(str(output_dir / split / "data.zarr"), mode="r")
        assert group.attrs["complete"]
        assert group["data"].shape == (1, steps, 3, 5, 2)
    assert (output_dir / "resolved_config.yaml").exists()
    stats = OmegaConf.load(output_dir / "stats.yml")
    assert list(stats.core_field_names) == ["field_0", "field_1"]


def _tiny_generation_config(output_dir: Path, data_format: str) -> DictConfig:
    cfg = OmegaConf.create(
        {
            "simulator": {
                "_target_": "autosim.simulations.synthetic.GaussianField",
                "height": 3,
                "width": 5,
                "n_channels": 2,
                "n_steps": 5,
                "rho": 0.8,
            },
            "dataset": {
                "output_dir": str(output_dir),
                "format": data_format,
                "n_train": 2,
                "n_valid": 1,
                "n_test": 1,
                "steps_per_split": {"train": 9, "valid": 6, "test": 4},
                "shard_steps": 4,
                "chunk_steps": 1,
            },
            "seed": 11,
            "overwrite": False,
            "visualize": {"enabled": False},
        }
    )
    assert isinstance(cfg, DictConfig)
    return cfg


def _dataset_files(output_dir: Path) -> dict[Path, bytes]:
    return {
        path.relative_to(output_dir): path.read_bytes()
        for path in output_dir.rglob("*")
        if path.is_file()
    }


def _existing_zarr_marker(output_dir: Path, split: str) -> None:
    store = output_dir / split / "data.zarr"
    store.mkdir(parents=True)
    (store / "zarr.json").write_text("existing Zarr metadata", encoding="utf-8")
    (output_dir / "stats.yml").write_text("existing statistics", encoding="utf-8")
    (output_dir / "resolved_config.yaml").write_text(
        "existing configuration", encoding="utf-8"
    )


@pytest.mark.parametrize("split", ["train", "valid", "test"])
@pytest.mark.parametrize("overwrite", [False, True])
def test_pt_cli_rejects_zarr_before_generation_or_metadata_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, split: str, overwrite: bool
) -> None:
    output_dir = tmp_path / "dataset"
    _existing_zarr_marker(output_dir, split)
    existing_files = _dataset_files(output_dir)
    cfg = _tiny_generation_config(output_dir, "pt")
    cfg.overwrite = overwrite

    def unexpected_generation(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Existing Zarr must be detected before simulator generation")

    monkeypatch.setattr(cli, "generate_dataset_splits", unexpected_generation)
    with pytest.raises(FileExistsError, match=r"[Zz]arr"):
        cli._generate_main(cfg)

    assert _dataset_files(output_dir) == existing_files


@pytest.mark.parametrize("split", ["train", "valid", "test"])
@pytest.mark.parametrize("overwrite", [False, True])
def test_save_dataset_splits_rejects_zarr_before_writing_any_pt_files(
    tmp_path: Path, split: str, overwrite: bool
) -> None:
    output_dir = tmp_path / "dataset"
    _existing_zarr_marker(output_dir, split)
    existing_files = _dataset_files(output_dir)
    payloads: dict[str, dict[str, Any]] = {
        name: {} for name in ("train", "valid", "test")
    }

    with pytest.raises(FileExistsError, match=r"[Zz]arr"):
        cli.save_dataset_splits(payloads, output_dir, overwrite=overwrite)

    assert _dataset_files(output_dir) == existing_files


def test_zarr_cli_rejects_overwrite_before_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "dataset"
    _existing_zarr_marker(output_dir, "train")
    existing_files = _dataset_files(output_dir)
    cfg = _tiny_generation_config(output_dir, "zarr")
    cfg.overwrite = True

    def unexpected_generation(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Zarr overwrite must be rejected before simulator creation")

    monkeypatch.setattr(cli, "build_simulator", unexpected_generation)
    with pytest.raises(ValueError, match="overwrite=true is unsupported"):
        cli._generate_main(cfg)
    assert _dataset_files(output_dir) == existing_files


def test_zarr_cli_config_resolution_failure_does_not_create_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "dataset"
    cfg = _tiny_generation_config(output_dir, "zarr")
    cfg.missing_ref = "${missing}"

    def unexpected_export(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Resolve configuration before starting export")

    monkeypatch.setattr(cli, "export_chunked_dataset", unexpected_export)
    with pytest.raises(InterpolationKeyError, match="missing"):
        cli._generate_main(cfg)
    assert not output_dir.exists()
