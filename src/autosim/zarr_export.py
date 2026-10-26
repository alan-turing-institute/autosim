"""Bounded-memory export of synthetic trajectories to sharded Zarr arrays."""

from __future__ import annotations

import copy
import importlib
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from autosim.simulations.synthetic import GaussianField

_SPLITS = ("train", "valid", "test")


class _ChannelMoments:
    """Merge population moments using float64 accumulators."""

    def __init__(self, channels: int) -> None:
        self.count = 0
        self.mean = torch.zeros(channels, dtype=torch.float64, device="cpu")
        self.m2 = torch.zeros_like(self.mean)

    def update(self, values: torch.Tensor) -> None:
        count = math.prod(values.shape[:-1])
        if not count:
            return
        variance, mean = torch.var_mean(
            values.to(torch.float64),
            dim=tuple(range(values.ndim - 1)),
            unbiased=False,
        )
        difference = mean - self.mean
        total = self.count + count
        self.m2 += variance * count + difference.square() * self.count * count / total
        self.mean += difference * count / total
        self.count = total

    def std(self) -> torch.Tensor:
        return (self.m2 / self.count).clamp_min(0).sqrt()


def _validate_export(
    sim: GaussianField,
    output_dir: Path,
    split_counts: dict[str, int],
    split_steps: dict[str, int],
    shard_steps: int,
    chunk_steps: int,
    base_seed: int | None,
) -> int:
    if not isinstance(sim, GaussianField):
        msg = "Chunked synthetic export currently requires a GaussianField simulator."
        raise TypeError(msg)
    for name, values in (("split_counts", split_counts), ("split_steps", split_steps)):
        if set(values) != set(_SPLITS):
            msg = f"{name} must contain exactly train, valid and test."
            raise ValueError(msg)
        for split, value in values.items():
            sim._check_positive_int(f"{name}.{split}", value)
            if name == "split_steps" and value < 2:
                msg = f"split_steps.{split} must contain at least two frames."
                raise ValueError(msg)
    sim._check_positive_int("shard_steps", shard_steps)
    sim._check_positive_int("chunk_steps", chunk_steps)
    if shard_steps % chunk_steps:
        msg = "shard_steps must be divisible by chunk_steps."
        raise ValueError(msg)
    # Zarr codecs may buffer a nominal full shard, including at array edges.
    sim._check_output_size((1, shard_steps, sim.height, sim.width, sim.n_channels))
    seed_stride = sim._retry_budget(max(split_counts.values())) + 1
    if base_seed is not None:
        if isinstance(base_seed, bool) or not isinstance(base_seed, int):
            msg = "base_seed must be an integer or None."
            raise ValueError(msg)
        for offset in range(len(_SPLITS)):
            torch.Generator(device="cpu").manual_seed(base_seed + offset * seed_stride)
    _check_destinations(output_dir)
    return seed_stride


def _check_destinations(output_dir: Path) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        msg = f"Dataset output path is not a directory: {output_dir}."
        raise ValueError(msg)
    for name in ("stats.yml", "resolved_config.yaml"):
        path = output_dir / name
        if path.exists():
            msg = f"Zarr export requires new dataset metadata; '{path}' already exists."
            raise FileExistsError(msg)
    for split in _SPLITS:
        if (output_dir / split / "data.pt").exists():
            msg = f"Refusing to mix Zarr output with legacy {split}/data.pt."
            raise FileExistsError(msg)
        destination = output_dir / split / "data.zarr"
        if destination.exists():
            msg = f"Zarr export requires a new dataset; '{destination}' already exists."
            raise FileExistsError(msg)


def _publish_dataset(staging: Path, output_dir: Path, zarr: Any) -> None:
    """Publish a new dataset; remove only this invocation's outputs on failure."""
    published: list[Path] = []
    stores = [Path(split) / "data.zarr" for split in _SPLITS]
    paths = [Path("stats.yml"), Path("resolved_config.yaml"), *stores]
    try:
        for relative in paths:
            destination = output_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = staging / relative
            if source.exists():
                os.replace(source, destination)
                published.append(destination)
        # No split is complete until every array and metadata file is in place.
        for relative in stores:
            group = zarr.open_group(str(output_dir / relative), mode="r+")
            group.attrs["complete"] = True
    except BaseException:
        for destination in reversed(published):
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        raise


def _stage_metadata(
    staging: Path,
    field_names: list[str],
    moments: _ChannelMoments,
    deltas: _ChannelMoments,
    resolved_config: str | None,
) -> dict[str, Any]:
    def by_channel(values: torch.Tensor) -> dict[str, float]:
        return {field: float(values[i]) for i, field in enumerate(field_names)}

    stats: dict[str, Any] = {
        "stats": {
            "mean": by_channel(moments.mean),
            "std": by_channel(moments.std()),
            "mean_delta": by_channel(deltas.mean),
            "std_delta": by_channel(deltas.std()),
        },
        "core_field_names": field_names,
        "constant_field_names": [],
    }
    OmegaConf.save(OmegaConf.create(stats), staging / "stats.yml")
    if resolved_config is not None:
        (staging / "resolved_config.yaml").write_text(resolved_config, encoding="utf-8")
    return stats


def export_chunked_dataset(
    sim: GaussianField,
    output_dir: Path,
    split_counts: dict[str, int],
    split_steps: dict[str, int],
    shard_steps: int,
    chunk_steps: int = 1,
    base_seed: int | None = None,
    resolved_config: str | None = None,
) -> dict[str, Any]:
    """Write complete synthetic split stores and return training statistics.

    Trajectories remain continuous across storage shards. Each split contains
    ``data.zarr/data`` in ``[trajectory, time, height, width, channel]`` order.
    Zarr handles shard indexing, so training windows need no duplicated halos.
    Generation writes whole shards; memory scales with one shard and codec and
    statistics work buffers, rather than with the full dataset.

    Args:
        sim: Gaussian field generator with generic shape and correlation settings.
        output_dir: New dataset root, used exclusively by this generation process.
        split_counts: Number of independent trajectories in each split.
        split_steps: Number of frames in each trajectory, independently per split.
        shard_steps: Frames per physical storage shard and generation batch.
        chunk_steps: Frames per independently readable chunk; divides shard_steps.
        base_seed: Base seed using the same split offsets as eager CLI generation.
        resolved_config: Optional YAML text saved as resolved_config.yaml alongside
            the arrays and full-precision stats.yml.

    Returns:
        Training-only channel means, population standard deviations and temporal
        difference statistics in the existing AutoSim ``stats.yml`` payload format.

    Raises:
        ImportError: If the optional Zarr dependency is unavailable.
        ValueError: If dimensions, counts, lengths or chunk sizes are invalid.
        FileExistsError: If output contains dataset stores or metadata already,
            including existing legacy data.pt files.
    """
    output_dir = Path(output_dir)
    seed_stride = _validate_export(
        sim,
        output_dir,
        split_counts,
        split_steps,
        shard_steps,
        chunk_steps,
        base_seed,
    )
    try:
        zarr = importlib.import_module("zarr")
    except ModuleNotFoundError as error:
        if error.name != "zarr":
            raise
        msg = "Zarr export requires the autosim[zarr] extra on Python 3.11 or later."
        raise ImportError(msg) from error

    moments = _ChannelMoments(sim.n_channels)
    deltas = _ChannelMoments(sim.n_channels)
    dtype = np.dtype("float32" if sim.dtype == torch.float32 else "float64")
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".autosim-zarr-", dir=output_dir) as name:
        staging = Path(name)
        for offset, split in enumerate(_SPLITS):
            split_sim = copy.copy(sim)
            split_sim.n_steps = split_steps[split]
            shape = (
                split_counts[split],
                split_steps[split],
                sim.height,
                sim.width,
                sim.n_channels,
            )
            group = zarr.open_group(
                str(staging / split / "data.zarr"),
                mode="w",
                zarr_format=3,
            )
            group.attrs.update(
                {
                    "schema": "autosim.spatiotemporal",
                    "schema_version": 1,
                    "field_names": list(sim.output_names),
                    "axes": ["trajectory", "time", "height", "width", "channel"],
                    "complete": False,
                }
            )
            array = group.create_array(
                "data",
                shape=shape,
                dtype=dtype,
                chunks=(1, chunk_steps, sim.height, sim.width, sim.n_channels),
                shards=(1, shard_steps, sim.height, sim.width, sim.n_channels),
                compressors=None,
            )
            seed = None if base_seed is None else base_seed + offset * seed_stride
            previous = None
            for payload in split_sim.iter_samples_spatiotemporal(
                split_counts[split],
                chunk_steps=shard_steps,
                random_seed=seed,
            ):
                data = payload["data"]
                trajectory, start = payload["trajectory_index"], payload["start_step"]
                array[trajectory : trajectory + 1, start : start + data.shape[1]] = (
                    data.numpy()
                )
                if trajectory == 0 and start == 0:
                    group.attrs["provenance"] = payload["provenance"]
                if split == "train":
                    if start == 0:
                        previous = None
                    moments.update(data)
                    if previous is not None:
                        deltas.update(data[:, :1] - previous)
                    deltas.update(data[:, 1:] - data[:, :-1])
                    # Clone one frame so the prior shard can be released.
                    previous = data[:, -1:].clone()

        stats = _stage_metadata(
            staging, list(sim.output_names), moments, deltas, resolved_config
        )
        _publish_dataset(staging, output_dir, zarr)
        return stats
