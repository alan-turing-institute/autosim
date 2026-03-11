from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from autosim.cli import (
    build_simulator,
    combine_stratified_splits,
    compute_normalization_stats,
    generate_dataset_splits,
    generate_normalization_stats_yaml,
    get_per_strata_counts,
    save_dataset_splits,
    save_example_videos,
)
from autosim.simulations.base import SpatioTemporalSimulator


class DummySimulator(SpatioTemporalSimulator):
    def _forward(self, x: torch.Tensor) -> torch.Tensor | None:
        msg = "DummySimulator does not implement _forward."
        raise NotImplementedError(msg)

    def forward_samples_spatiotemporal(
        self,
        n: int,
        random_seed: int | None = None,
        ensure_exact_n: bool = False,
    ) -> dict:
        del ensure_exact_n
        seed_value = -1 if random_seed is None else random_seed
        return {
            "data": torch.full((n, 1, 2, 2, 1), float(seed_value), dtype=torch.float32),
            "constant_scalars": torch.tensor([seed_value]),
            "constant_fields": None,
        }


def test_build_simulator_from_target_core_and_experimental() -> None:
    core_cfg = OmegaConf.create(
        {"_target_": "autosim.simulations.AdvectionDiffusion", "log_level": "warning"}
    )
    experimental_cfg = OmegaConf.create(
        {
            "_target_": "autosim.experimental.simulations.ShallowWater2D",
            "log_level": "warning",
        }
    )

    assert build_simulator(core_cfg).__class__.__name__ == "AdvectionDiffusion"
    assert build_simulator(experimental_cfg).__class__.__name__ == "ShallowWater2D"


def test_generate_dataset_splits_uses_non_overlapping_seed_namespaces() -> None:
    splits = generate_dataset_splits(
        sim=DummySimulator({}, []),
        n_train=3,
        n_valid=2,
        n_test=1,
        base_seed=11,
    )

    assert splits["train"]["data"].shape[0] == 3
    assert splits["valid"]["data"].shape[0] == 2
    assert splits["test"]["data"].shape[0] == 1
    assert splits["train"]["constant_scalars"].item() == 11
    assert splits["valid"]["constant_scalars"].item() == 112
    assert splits["test"]["constant_scalars"].item() == 213


def test_build_simulator_rejects_non_spatiotemporal() -> None:
    non_spatiotemporal_cfg = OmegaConf.create(
        {"_target_": "autosim.simulations.Projectile", "log_level": "warning"}
    )

    with pytest.raises(TypeError, match="SpatioTemporalSimulator"):
        build_simulator(non_spatiotemporal_cfg)


def test_save_dataset_splits_writes_expected_structure(tmp_path: Path) -> None:
    splits = generate_dataset_splits(
        sim=DummySimulator({}, []),
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
        sim=DummySimulator({}, []),
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
        "overwrite=true",
        "simulator=advection_diffusion",
        "simulator.log_level=warning",
        "simulator.return_timeseries=true",
        "simulator.n=8",
        "simulator.L=4.0",
        "simulator.T=0.1",
        "simulator.dt=0.1",
        "visualize.enabled=false",
        f"hydra.run.dir={hydra_run_dir.as_posix()}",
        "hydra.output_subdir=null",
    ]

    subprocess.run(command, check=True, cwd=repo_root)

    for split in ("train", "valid", "test"):
        split_path = output_dir / split / "data.pt"
        assert split_path.exists()
        payload = torch.load(split_path)
        assert payload["data"].shape[0] == 1


def test_cli_list_subcommand_outputs_simulator_names() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "autosim.cli", "list"],
        check=True,
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    output_lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    assert "advection_diffusion" in output_lines
    assert "shallow_water2d" in output_lines


def test_compute_normalization_stats_includes_temporal_deltas() -> None:
    split_payload = {
        "data": torch.tensor(
            [
                [
                    [[[1.0, 10.0]]],
                    [[[3.0, 14.0]]],
                    [[[5.0, 18.0]]],
                ]
            ],
            dtype=torch.float32,
        )
    }
    stats_payload = compute_normalization_stats(
        split_payload=split_payload, core_field_names=["U", "V"]
    )
    stats = stats_payload["normalization_stats"]["stats"]

    assert stats["mean"]["U"] == pytest.approx(3.0)
    assert stats["mean"]["V"] == pytest.approx(14.0)
    assert stats["std"]["U"] == pytest.approx((8.0 / 3.0) ** 0.5)
    assert stats["std"]["V"] == pytest.approx((32.0 / 3.0) ** 0.5)
    assert stats["mean_delta"]["U"] == pytest.approx(2.0)
    assert stats["mean_delta"]["V"] == pytest.approx(4.0)
    assert stats["std_delta"]["U"] == pytest.approx(0.0)
    assert stats["std_delta"]["V"] == pytest.approx(0.0)
    assert stats_payload["normalization_stats"]["core_field_names"] == ["U", "V"]
    assert stats_payload["normalization_stats"]["constant_field_names"] == []


def test_generate_normalization_stats_yaml_from_existing_dataset(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    train_dir = dataset_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "data": torch.tensor(
                [
                    [
                        [[[1.0, 2.0]]],
                        [[[2.0, 4.0]]],
                    ],
                    [
                        [[[3.0, 6.0]]],
                        [[[4.0, 8.0]]],
                    ],
                ],
                dtype=torch.float32,
            )
        },
        train_dir / "data.pt",
    )
    output_path = generate_normalization_stats_yaml(
        dataset_dir=dataset_dir, core_field_names=["U", "V"]
    )

    assert output_path.exists()
    stats_cfg = OmegaConf.load(output_path)
    assert isinstance(stats_cfg, DictConfig)
    assert stats_cfg["normalization_stats"]["core_field_names"] == ["U", "V"]
    assert stats_cfg["normalization_stats"]["stats"]["mean"]["U"] == pytest.approx(2.5)
    assert stats_cfg["normalization_stats"]["stats"]["mean_delta"][
        "V"
    ] == pytest.approx(2.0)


def test_cli_stats_subcommand_writes_yaml(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    train_dir = dataset_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "data": torch.tensor(
                [
                    [
                        [[[1.0, 10.0]]],
                        [[[2.0, 12.0]]],
                    ]
                ],
                dtype=torch.float32,
            )
        },
        train_dir / "data.pt",
    )
    repo_root = Path(__file__).resolve().parents[1]

    subprocess.run(
        [
            sys.executable,
            "-m",
            "autosim.cli",
            "stats",
            dataset_dir.as_posix(),
            "--field-names",
            "U,V",
        ],
        check=True,
        cwd=repo_root,
    )

    stats_path = dataset_dir / f"{dataset_dir.name}.yaml"
    assert stats_path.exists()
    stats_cfg = OmegaConf.load(stats_path)
    assert isinstance(stats_cfg, DictConfig)
    assert stats_cfg["normalization_stats"]["stats"]["mean_delta"][
        "U"
    ] == pytest.approx(1.0)
    assert stats_cfg["normalization_stats"]["stats"]["std_delta"]["V"] == pytest.approx(
        0.0
    )


def test_cli_help_outputs_usage() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "autosim.cli", "--help"],
        check=True,
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert "usage:" in result.stdout.lower()
    assert "list" in result.stdout
    assert "stats" in result.stdout


def test_get_per_strata_counts_requires_exact_divisibility() -> None:
    with pytest.raises(ValueError, match="must be divisible"):
        get_per_strata_counts(n_train=10, n_valid=4, n_test=4, n_strata=3)

    train, valid, test = get_per_strata_counts(
        n_train=12, n_valid=6, n_test=3, n_strata=3
    )
    assert (train, valid, test) == (4, 2, 1)


def test_combine_stratified_splits_preserves_strata_order() -> None:
    group_a = {
        "train": {
            "data": torch.full((2, 1, 1, 1, 1), 1.0),
            "constant_scalars": torch.full((2, 1), 1.0),
            "constant_fields": None,
        },
        "valid": {
            "data": torch.full((1, 1, 1, 1, 1), 1.0),
            "constant_scalars": torch.full((1, 1), 1.0),
            "constant_fields": None,
        },
        "test": {
            "data": torch.full((1, 1, 1, 1, 1), 1.0),
            "constant_scalars": torch.full((1, 1), 1.0),
            "constant_fields": None,
        },
    }
    group_b = {
        "train": {
            "data": torch.full((2, 1, 1, 1, 1), 2.0),
            "constant_scalars": torch.full((2, 1), 2.0),
            "constant_fields": None,
        },
        "valid": {
            "data": torch.full((1, 1, 1, 1, 1), 2.0),
            "constant_scalars": torch.full((1, 1), 2.0),
            "constant_fields": None,
        },
        "test": {
            "data": torch.full((1, 1, 1, 1, 1), 2.0),
            "constant_scalars": torch.full((1, 1), 2.0),
            "constant_fields": None,
        },
    }

    combined = combine_stratified_splits([group_a, group_b])

    assert combined["train"]["data"].shape[0] == 4
    assert torch.all(combined["train"]["data"][:2] == 1.0)
    assert torch.all(combined["train"]["data"][2:] == 2.0)


@pytest.fixture
def dummy_splits() -> dict:
    return {
        "train": {
            "data": torch.zeros((3, 2, 4, 4, 2), dtype=torch.float32),
            "constant_scalars": torch.zeros((3, 1), dtype=torch.float32),
            "constant_fields": None,
        },
        "valid": {
            "data": torch.zeros((1, 2, 4, 4, 2), dtype=torch.float32),
            "constant_scalars": torch.zeros((1, 1), dtype=torch.float32),
            "constant_fields": None,
        },
        "test": {
            "data": torch.zeros((1, 2, 4, 4, 2), dtype=torch.float32),
            "constant_scalars": torch.zeros((1, 1), dtype=torch.float32),
            "constant_fields": None,
        },
    }


def test_save_example_videos_disabled_is_noop(tmp_path: Path, dummy_splits) -> None:
    cfg = OmegaConf.create({"enabled": False})
    save_example_videos(splits=dummy_splits, output_dir=tmp_path, visualize_cfg=cfg)
    assert not (tmp_path / "examples").exists()


def test_save_example_videos_empty_batch_indices_is_noop(
    tmp_path: Path, dummy_splits, monkeypatch
) -> None:
    import autosim.cli as cli_module  # noqa: PLC0415

    calls: list = []
    monkeypatch.setattr(
        cli_module, "plot_spatiotemporal_video", lambda **kw: calls.append(kw)
    )

    cfg = OmegaConf.create({"enabled": True, "split": "train", "batch_indices": []})
    save_example_videos(splits=dummy_splits, output_dir=tmp_path, visualize_cfg=cfg)
    assert calls == []


def test_save_example_videos_out_of_range_raises(tmp_path: Path, dummy_splits) -> None:
    cfg = OmegaConf.create(
        {
            "enabled": True,
            "split": "train",
            "batch_indices": [99],
            "fps": 5,
            "file_ext": "gif",
            "overwrite": True,
        }
    )
    with pytest.raises(ValueError, match="out of range"):
        save_example_videos(splits=dummy_splits, output_dir=tmp_path, visualize_cfg=cfg)


def test_save_example_videos_uses_batch_indices_and_split(
    tmp_path: Path, dummy_splits, monkeypatch
) -> None:
    import autosim.cli as cli_module  # noqa: PLC0415

    calls: list[dict] = []

    def _fake(**kwargs):
        save_path = Path(str(kwargs["save_path"]))
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text("stub")
        calls.append(kwargs)

    monkeypatch.setattr(cli_module, "plot_spatiotemporal_video", _fake)

    cfg = OmegaConf.create(
        {
            "enabled": True,
            "split": "train",
            "batch_indices": [0, 2],
            "fps": 7,
            "file_ext": "gif",
            "overwrite": True,
        }
    )

    save_example_videos(
        splits=dummy_splits,
        output_dir=tmp_path,
        visualize_cfg=cfg,
        channel_names=["h", "u"],
    )

    assert len(calls) == 2
    assert all(call["channel_names"] == ["h", "u"] for call in calls)
    saved_paths = sorted(Path(str(call["save_path"])) for call in calls)
    assert saved_paths[0] == tmp_path / "examples" / "train" / "batch_0.gif"
    assert saved_paths[1] == tmp_path / "examples" / "train" / "batch_2.gif"
