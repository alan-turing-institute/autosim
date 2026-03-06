from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.utils import get_original_cwd, instantiate
from omegaconf import OmegaConf

from autosim.simulations.base import SpatioTemporalSimulator
from autosim.utils import plot_spatiotemporal_video

if not OmegaConf.has_resolver("shortuuid"):
    OmegaConf.register_new_resolver(
        "shortuuid", lambda n=7: uuid.uuid4().hex[: int(n)], use_cache=True
    )


def build_simulator(simulator_cfg: Any) -> SpatioTemporalSimulator:
    """Instantiate and validate a spatiotemporal simulator from Hydra config."""
    simulator = instantiate(simulator_cfg)
    if not isinstance(simulator, SpatioTemporalSimulator):
        msg = (
            "Configured simulator must inherit from SpatioTemporalSimulator for "
            "dataset generation. For non-spatiotemporal simulators, use their "
            "forward/forward_batch API directly."
        )
        raise TypeError(msg)
    return simulator


def generate_dataset_splits(
    sim: SpatioTemporalSimulator,
    n_train: int,
    n_valid: int,
    n_test: int,
    base_seed: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Generate train/valid/test splits from a simulator."""

    def get_seed(offset: int) -> int | None:
        if base_seed is None:
            return None
        return base_seed + offset

    train = sim.forward_samples_spatiotemporal(n=n_train, random_seed=get_seed(0))
    valid = sim.forward_samples_spatiotemporal(n=n_valid, random_seed=get_seed(1))
    test = sim.forward_samples_spatiotemporal(n=n_test, random_seed=get_seed(2))
    return {"train": train, "valid": valid, "test": test}


def save_dataset_splits(
    splits: dict[str, dict[str, Any]],
    output_dir: str | Path,
    overwrite: bool = False,
) -> None:
    """Persist split dictionaries to `output_dir/{split}/data.pt`."""
    output_path = Path(output_dir)
    expected_files = [
        output_path / split / "data.pt" for split in ("train", "valid", "test")
    ]
    if not overwrite and any(path.exists() for path in expected_files):
        msg = (
            f"Refusing to overwrite existing dataset files in '{output_path}'. "
            "Set overwrite=true to replace them."
        )
        raise FileExistsError(msg)

    for split_name, payload in splits.items():
        split_dir = output_path / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        torch.save(payload, split_dir / "data.pt")


def save_example_videos(
    splits: dict[str, dict[str, Any]],
    output_dir: str | Path,
    visualize_cfg: Any | None,
) -> None:
    """Optionally render example videos for selected batch indices.

    Expected data shape is ``[batch, time, x, y, channels]``.
    """
    if visualize_cfg is None or not bool(visualize_cfg.get("enabled", False)):
        return

    split_name = str(visualize_cfg.get("split", "train"))
    if split_name not in splits:
        msg = f"visualize.split='{split_name}' not found in generated splits."
        raise ValueError(msg)

    split_payload = splits[split_name]
    data = split_payload.get("data")
    if not isinstance(data, torch.Tensor) or data.ndim != 5:
        msg = (
            "visualization expects split payload 'data' as a 5D torch.Tensor "
            "with shape [batch,time,x,y,channels]."
        )
        raise ValueError(msg)

    batch_indices_cfg = visualize_cfg.get("batch_indices", [])
    batch_indices = [int(idx) for idx in batch_indices_cfg]
    if not batch_indices:
        return
    for idx in batch_indices:
        if idx < 0 or idx >= data.shape[0]:
            msg = (
                f"visualize batch index {idx} is out of range for split "
                f"'{split_name}' with batch size {data.shape[0]}."
            )
            raise ValueError(msg)

    fps = int(visualize_cfg.get("fps", 5))
    if fps <= 0:
        msg = "visualize.fps must be positive."
        raise ValueError(msg)

    file_ext = str(visualize_cfg.get("file_ext", "gif")).lstrip(".").lower()
    if file_ext not in {"gif", "mp4"}:
        msg = "visualize.file_ext must be one of ['gif', 'mp4']."
        raise ValueError(msg)

    videos_dir = Path(output_dir) / "examples" / split_name
    videos_dir.mkdir(parents=True, exist_ok=True)

    overwrite = bool(visualize_cfg.get("overwrite", True))

    for idx in batch_indices:
        save_path = videos_dir / f"batch_{idx}.{file_ext}"
        if save_path.exists() and not overwrite:
            continue
        plot_spatiotemporal_video(
            true=data,
            batch_idx=idx,
            fps=fps,
            save_path=str(save_path),
        )


def get_per_strata_counts(
    n_train: int,
    n_valid: int,
    n_test: int,
    n_strata: int,
) -> tuple[int, int, int]:
    """Get per-strata split sizes, requiring exact divisibility."""
    if n_strata <= 0:
        msg = "Number of strata must be positive."
        raise ValueError(msg)

    for split_name, total in (
        ("train", n_train),
        ("valid", n_valid),
        ("test", n_test),
    ):
        if total % n_strata != 0:
            msg = (
                f"dataset.n_{split_name}={total} must be divisible by "
                f"number of strata ({n_strata})."
            )
            raise ValueError(msg)

    return n_train // n_strata, n_valid // n_strata, n_test // n_strata


def combine_stratified_splits(
    ordered_strata_splits: list[dict[str, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Combine per-strata splits preserving strata order in batch dimension."""
    if not ordered_strata_splits:
        msg = "No strata outputs to combine."
        raise ValueError(msg)

    combined: dict[str, dict[str, Any]] = {}
    split_names = ("train", "valid", "test")
    for split in split_names:
        first_payload = ordered_strata_splits[0][split]
        merged_payload: dict[str, Any] = {}

        for key in first_payload:
            values = [group[split][key] for group in ordered_strata_splits]
            if all(isinstance(value, torch.Tensor) for value in values):
                merged_payload[key] = torch.cat(values, dim=0)
            elif all(value is None for value in values):
                merged_payload[key] = None
            else:
                msg = (
                    f"Cannot combine non-tensor field '{key}' across strata. "
                    "Expected all tensors or all None."
                )
                raise ValueError(msg)

        combined[split] = merged_payload

    return combined


@hydra.main(version_base=None, config_path="configs", config_name="generate_data")
def _generate_main(cfg: Any) -> None:
    """Generate simulation datasets from a Hydra-configured simulator."""
    stratify_cfg = cfg.get("stratify")
    if stratify_cfg is not None and bool(stratify_cfg.get("enabled", False)):
        key = stratify_cfg.get("key")
        values = list(stratify_cfg.get("values", []))
        if key is None or str(key).strip() == "":
            msg = "stratify.key must be set when stratify.enabled=true."
            raise ValueError(msg)
        if not values:
            msg = "stratify.values must be a non-empty list when stratify.enabled=true."
            raise ValueError(msg)

        n_train_each, n_valid_each, n_test_each = get_per_strata_counts(
            n_train=cfg.dataset.n_train,
            n_valid=cfg.dataset.n_valid,
            n_test=cfg.dataset.n_test,
            n_strata=len(values),
        )

        key_path = str(key)
        if key_path.startswith("simulator."):
            key_path = key_path[len("simulator.") :]

        per_strata_outputs: list[dict[str, dict[str, Any]]] = []
        for value in values:
            sim_cfg = OmegaConf.create(
                OmegaConf.to_container(cfg.simulator, resolve=True)
            )
            OmegaConf.update(sim_cfg, key_path, value, merge=False)
            sim = build_simulator(sim_cfg)
            splits = generate_dataset_splits(
                sim=sim,
                n_train=n_train_each,
                n_valid=n_valid_each,
                n_test=n_test_each,
                base_seed=cfg.seed,
            )
            per_strata_outputs.append(splits)

        splits = combine_stratified_splits(per_strata_outputs)
    else:
        sim = build_simulator(cfg.simulator)

        splits = generate_dataset_splits(
            sim=sim,
            n_train=cfg.dataset.n_train,
            n_valid=cfg.dataset.n_valid,
            n_test=cfg.dataset.n_test,
            base_seed=cfg.seed,
        )

    output_dir = Path(cfg.dataset.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path(get_original_cwd()) / output_dir

    save_dataset_splits(splits=splits, output_dir=output_dir, overwrite=cfg.overwrite)
    save_example_videos(
        splits=splits,
        output_dir=output_dir,
        visualize_cfg=cfg.get("visualize"),
    )


def list_simulators() -> list[str]:
    """Return available simulator config names from the package config group."""
    simulator_dir = Path(__file__).parent / "configs" / "simulator"
    if not simulator_dir.exists():
        return []
    return sorted(path.stem for path in simulator_dir.glob("*.yaml"))


def main() -> None:
    """Dispatch tiny autosim subcommands.

    - `autosim list` prints simulator config names.
    - `autosim` (or any Hydra overrides) runs data generation.
    """
    argv = sys.argv[1:]

    if argv and argv[0] in {"-h", "--help"}:
        parser = argparse.ArgumentParser(
            prog="autosim",
            description=(
                "Generate simulation datasets using Hydra overrides, or list "
                "available simulator configs."
            ),
        )
        parser.add_argument(
            "command",
            nargs="?",
            help="Subcommand: 'list'. Omit to run data generation with Hydra.",
        )
        parser.print_help()
        return

    if argv and argv[0] == "list":
        list_parser = argparse.ArgumentParser(
            prog="autosim list",
            description="List available simulator config names.",
        )
        list_parser.parse_args(argv[1:])
        for name in list_simulators():
            print(name)
        return

    # Preserve all original arguments for Hydra's own parser.
    sys.argv = [sys.argv[0], *argv]
    _generate_main()


if __name__ == "__main__":
    main()
