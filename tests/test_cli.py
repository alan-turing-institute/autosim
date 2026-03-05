from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from autosim.cli import (
    generate_dataset_splits,
    resolve_simulator_class,
    save_dataset_splits,
)


class DummySimulator:
    def forward_samples_spatiotemporal(
        self, n: int, random_seed: int | None = None
    ) -> dict:
        seed_value = -1 if random_seed is None else random_seed
        return {
            "data": torch.full((n, 1, 2, 2, 1), float(seed_value), dtype=torch.float32),
            "constant_scalars": torch.tensor([seed_value]),
            "constant_fields": None,
        }


def test_resolve_simulator_class_from_core_and_experimental() -> None:
    assert resolve_simulator_class("ReactionDiffusion").__name__ == "ReactionDiffusion"
    assert resolve_simulator_class("ShallowWater2D").__name__ == "ShallowWater2D"


def test_generate_dataset_splits_uses_seed_offsets() -> None:
    splits = generate_dataset_splits(
        sim=DummySimulator(),
        n_train=3,
        n_valid=2,
        n_test=1,
        base_seed=11,
    )

    assert splits["train"]["data"].shape[0] == 3
    assert splits["valid"]["data"].shape[0] == 2
    assert splits["test"]["data"].shape[0] == 1
    assert splits["train"]["constant_scalars"].item() == 11
    assert splits["valid"]["constant_scalars"].item() == 12
    assert splits["test"]["constant_scalars"].item() == 13


def test_save_dataset_splits_writes_expected_structure(tmp_path: Path) -> None:
    splits = generate_dataset_splits(
        sim=DummySimulator(),
        n_train=1,
        n_valid=1,
        n_test=1,
        base_seed=5,
    )

    output_dir = tmp_path / "dataset"
    save_dataset_splits(splits=splits, output_dir=output_dir)

    for split in ("train", "valid", "test"):
        data_path = output_dir / split / "data.pt"
        assert data_path.exists()
        payload = torch.load(data_path)
        assert "data" in payload


def test_save_dataset_splits_respects_overwrite_flag(tmp_path: Path) -> None:
    splits = generate_dataset_splits(
        sim=DummySimulator(),
        n_train=1,
        n_valid=1,
        n_test=1,
    )
    output_dir = tmp_path / "dataset"
    save_dataset_splits(splits=splits, output_dir=output_dir)

    with pytest.raises(FileExistsError):
        save_dataset_splits(splits=splits, output_dir=output_dir, overwrite=False)

    save_dataset_splits(splits=splits, output_dir=output_dir, overwrite=True)


def test_cli_generates_dataset_fast_with_advection_diffusion(tmp_path: Path) -> None:
    output_dir = tmp_path / "generated"
    hydra_run_dir = tmp_path / "hydra_run"
    repo_root = Path(__file__).resolve().parents[1]

    command = [
        sys.executable,
        "-m",
        "autosim.cli",
        f"dataset.output_dir={output_dir.as_posix()}",
        "dataset.n_train=1",
        "dataset.n_valid=1",
        "dataset.n_test=1",
        "run.overwrite=true",
        "simulator.name=AdvectionDiffusion",
        "simulator.kwargs.log_level=warning",
        "simulator.kwargs.return_timeseries=true",
        "simulator.kwargs.n=8",
        "simulator.kwargs.L=4.0",
        "simulator.kwargs.T=0.1",
        "simulator.kwargs.dt=0.1",
        f"hydra.run.dir={hydra_run_dir.as_posix()}",
        "hydra.output_subdir=null",
    ]

    subprocess.run(command, check=True, cwd=repo_root)

    for split in ("train", "valid", "test"):
        split_path = output_dir / split / "data.pt"
        assert split_path.exists()
        payload = torch.load(split_path)
        assert payload["data"].shape[0] == 1
