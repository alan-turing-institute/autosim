from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import autosim.zarr_export as zarr_export_module
from autosim.cli import compute_normalization_stats
from autosim.simulations.synthetic import GaussianField
from autosim.zarr_export import export_chunked_dataset

zarr = pytest.importorskip("zarr")


def _simulator(**overrides: Any) -> GaussianField:
    options = {
        "height": 3,
        "width": 5,
        "n_channels": 2,
        "n_steps": 4,
        "rho": 0.8,
        "spatial_length_scale": 0.6,
        "channel_correlation": 0.3,
        "output_names": ["a", "b"],
    }
    return GaussianField(**(options | overrides))


def _export_options(output_dir: Path, **overrides: Any) -> dict[str, Any]:
    return {
        "sim": _simulator(),
        "output_dir": output_dir,
        "split_counts": {"train": 2, "valid": 1, "test": 1},
        "split_steps": {"train": 9, "valid": 6, "test": 4},
        "shard_steps": 4,
        "chunk_steps": 1,
        "base_seed": 11,
    } | overrides


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize(("chunk_steps", "shard_steps"), [(1, 2), (2, 4), (1, 16)])
def test_export_preserves_values_windows_provenance_and_statistics(
    tmp_path: Path, dtype: str, chunk_steps: int, shard_steps: int
) -> None:
    options = _export_options(
        tmp_path / "dataset",
        sim=_simulator(dtype=dtype),
        shard_steps=shard_steps,
        chunk_steps=chunk_steps,
        resolved_config="seed: 11\n",
    )
    before = torch.random.get_rng_state().clone()
    stats = export_chunked_dataset(**options)
    assert (
        OmegaConf.to_container(OmegaConf.load(options["output_dir"] / "stats.yml"))
        == stats
    )
    assert torch.equal(before, torch.random.get_rng_state())
    assert (options["output_dir"] / "resolved_config.yaml").read_text() == "seed: 11\n"
    assert options["sim"].n_steps == 4

    for offset, split in enumerate(("train", "valid", "test")):
        sim = copy.copy(options["sim"])
        sim.n_steps = options["split_steps"][split]
        expected = sim.forward_samples_spatiotemporal(
            options["split_counts"][split], random_seed=11 + offset * 101
        )
        group = zarr.open_group(options["output_dir"] / split / "data.zarr", mode="r")
        data = group["data"]
        np.testing.assert_array_equal(data[:], expected["data"].numpy())
        assert data.dtype == np.dtype(dtype)
        assert data.chunks == (1, chunk_steps, 3, 5, 2)
        assert data.shards == (1, shard_steps, 3, 5, 2)
        assert group.attrs["schema"] == "autosim.spatiotemporal"
        assert group.attrs["schema_version"] == 1
        assert group.attrs["field_names"] == ["a", "b"]
        assert group.attrs["complete"] is True
        assert group.attrs["provenance"] == expected["provenance"]

        # A storage boundary must remain readable as part of one temporal window.
        for trajectory in range(options["split_counts"][split]):
            for start in range(max(0, sim.n_steps - 5 + 1)):
                np.testing.assert_array_equal(
                    data[trajectory, start : start + 5],
                    expected["data"][trajectory, start : start + 5].numpy(),
                )
        if split == "train":
            expected_stats = compute_normalization_stats(
                expected, core_field_names=["a", "b"]
            )
            assert stats["core_field_names"] == expected_stats["core_field_names"]
            assert stats["constant_field_names"] == []
            tolerance = 1e-6 if dtype == "float32" else 1e-12
            for metric, values in expected_stats["stats"].items():
                for channel, value in values.items():
                    assert stats["stats"][metric][channel] == pytest.approx(
                        value, rel=tolerance, abs=tolerance
                    )


def test_export_replays_unseeded_provenance(tmp_path: Path) -> None:
    options = _export_options(tmp_path / "dataset", base_seed=None)
    export_chunked_dataset(**options)
    for split in ("train", "valid", "test"):
        group = zarr.open_group(options["output_dir"] / split / "data.zarr", mode="r")
        sim = copy.copy(options["sim"])
        sim.n_steps = options["split_steps"][split]
        replay = sim.forward_samples_spatiotemporal(
            options["split_counts"][split],
            random_seed=group.attrs["provenance"]["random_seed"],
        )
        np.testing.assert_array_equal(group["data"][:], replay["data"].numpy())


@pytest.mark.parametrize(
    "override",
    [
        {"split_counts": {"train": 0, "valid": 1, "test": 1}},
        {"split_counts": {"train": 1, "valid": True, "test": 1}},
        {"split_counts": {"train": 1, "valid": 1}},
        {"split_steps": {"train": 1, "valid": 6, "test": 4}},
        {"split_steps": {"train": 9, "valid": 6, "test": 0}},
        {"split_steps": {"train": 9, "valid": 6}},
        {"shard_steps": 0},
        {"chunk_steps": True},
        {"chunk_steps": 3, "shard_steps": 4},
    ],
)
def test_invalid_config_is_rejected_before_output_creation(
    tmp_path: Path, override: dict[str, Any]
) -> None:
    output_dir = tmp_path / "dataset"
    with pytest.raises((TypeError, ValueError)):
        export_chunked_dataset(**_export_options(output_dir, **override))
    assert not output_dir.exists()


def test_all_splits_are_checked_against_byte_limit_before_generation(
    tmp_path: Path, monkeypatch
) -> None:
    sim = _simulator(max_output_bytes=2 * 3 * 5 * 2 * 4)

    def unexpected_generation(*args, **kwargs):
        pytest.fail("Generation must not start before all split sizes are checked")

    monkeypatch.setattr(sim, "iter_samples_spatiotemporal", unexpected_generation)
    output_dir = tmp_path / "dataset"
    with pytest.raises(ValueError, match="max_output_bytes"):
        export_chunked_dataset(
            **_export_options(
                output_dir,
                sim=sim,
                split_steps={"train": 2, "valid": 2, "test": 4},
            )
        )
    assert not output_dir.exists()


def test_unsupported_simulator_is_rejected_before_output_creation(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "dataset"
    with pytest.raises((TypeError, ValueError)):
        export_chunked_dataset(**_export_options(output_dir, sim=object()))
    assert not output_dir.exists()


@pytest.mark.parametrize(
    "existing",
    [
        "train/data.zarr",
        "valid/data.zarr",
        "test/data.zarr",
        "stats.yml",
        "resolved_config.yaml",
        "test/data.pt",
    ],
)
def test_existing_data_or_metadata_prevents_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: str
) -> None:
    output_dir = tmp_path / "dataset"
    path = output_dir / existing
    if path.suffix == ".zarr":
        path = path / "zarr.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"existing payload")

    def unexpected_generation(*args, **kwargs):
        pytest.fail("Existing outputs must be rejected before generation")

    monkeypatch.setattr(
        GaussianField, "iter_samples_spatiotemporal", unexpected_generation
    )
    with pytest.raises(FileExistsError):
        export_chunked_dataset(**_export_options(output_dir))
    assert path.read_bytes() == b"existing payload"
    assert [entry for entry in output_dir.rglob("*") if entry.is_file()] == [path]


def test_failed_export_does_not_publish_partial_store(
    tmp_path: Path, monkeypatch
) -> None:
    sim = _simulator()
    original = GaussianField.iter_samples_spatiotemporal

    def interrupted(self, *args, **kwargs):
        yield next(original(self, *args, **kwargs))
        msg = "simulated generation failure"
        raise RuntimeError(msg)

    monkeypatch.setattr(GaussianField, "iter_samples_spatiotemporal", interrupted)
    output_dir = tmp_path / "dataset"
    with pytest.raises(RuntimeError, match="simulated generation failure"):
        export_chunked_dataset(**_export_options(output_dir, sim=sim))
    assert not list(output_dir.iterdir())


@pytest.mark.parametrize("failure_path", ["resolved_config.yaml", "valid/data.zarr"])
def test_failed_publication_cleans_only_new_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_path: str
) -> None:
    output_dir = tmp_path / "dataset"
    output_dir.mkdir()
    unrelated = output_dir / "notes.txt"
    unrelated.write_text("keep", encoding="utf-8")
    original_replace = zarr_export_module.os.replace

    def fail_publication(source, destination):
        if Path(destination) == output_dir / failure_path:
            if failure_path == "valid/data.zarr":
                assert (output_dir / "stats.yml").is_file()
                assert (output_dir / "resolved_config.yaml").is_file()
                published = zarr.open_group(output_dir / "train/data.zarr", mode="r")
                assert published.attrs["complete"] is False
                staged = zarr.open_group(source, mode="r")
                assert staged.attrs["complete"] is False
            msg = "simulated publication failure"
            raise OSError(msg)
        return original_replace(source, destination)

    monkeypatch.setattr(zarr_export_module.os, "replace", fail_publication)
    with pytest.raises(OSError, match="simulated publication failure"):
        export_chunked_dataset(
            **_export_options(output_dir, resolved_config="seed: 11\n")
        )
    assert unrelated.read_text() == "keep"
    assert [path for path in output_dir.rglob("*") if path.is_file()] == [unrelated]
    assert not list(output_dir.rglob("data.zarr"))
    assert not list(output_dir.glob(".autosim-zarr-*"))


def test_failed_completion_removes_published_arrays_and_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    options = _export_options(tmp_path / "dataset")
    original_open_group = zarr.open_group

    def fail_completion(store, *args, **kwargs):
        if (
            Path(store) == options["output_dir"] / "test/data.zarr"
            and kwargs.get("mode") == "r+"
        ):
            msg = "simulated completion failure"
            raise OSError(msg)
        return original_open_group(store, *args, **kwargs)

    monkeypatch.setattr(zarr, "open_group", fail_completion)
    with pytest.raises(OSError, match="simulated completion failure"):
        export_chunked_dataset(**options)
    assert not list(options["output_dir"].rglob("zarr.json"))
    assert not (options["output_dir"] / "stats.yml").exists()


def test_failed_metadata_write_does_not_publish_stores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "dataset"
    original_write = Path.write_text

    def fail_config(path, *args, **kwargs):
        result = original_write(path, *args, **kwargs)
        if path.name == "resolved_config.yaml":
            msg = "simulated config write failure"
            raise OSError(msg)
        return result

    monkeypatch.setattr(Path, "write_text", fail_config)
    with pytest.raises(OSError, match="simulated config write failure"):
        export_chunked_dataset(
            **_export_options(output_dir, resolved_config="seed: 11\n")
        )
    assert not list(output_dir.iterdir())
