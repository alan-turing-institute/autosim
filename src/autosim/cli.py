from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.utils import get_original_cwd, instantiate


def build_simulator(simulator_cfg: Any) -> Any:
    """Instantiate a simulator from Hydra config and validate required interface."""
    simulator = instantiate(simulator_cfg)
    forward_method = getattr(simulator, "forward_samples_spatiotemporal", None)
    if not callable(forward_method):
        msg = "Simulator must implement forward_samples_spatiotemporal(n, random_seed)."
        raise TypeError(msg)
    return simulator


def generate_dataset_splits(
    sim: Any,
    n_train: int,
    n_valid: int,
    n_test: int,
    base_seed: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Generate train/valid/test splits from a simulator."""
    if not hasattr(sim, "forward_samples_spatiotemporal"):
        msg = (
            "Simulator does not implement forward_samples_spatiotemporal, "
            "which is required for training data generation."
        )
        raise TypeError(msg)

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
            "Set run.overwrite=true to replace them."
        )
        raise FileExistsError(msg)

    for split_name, payload in splits.items():
        split_dir = output_path / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        torch.save(payload, split_dir / "data.pt")


@hydra.main(version_base=None, config_path="configs", config_name="generate_data")
def main(cfg: Any) -> None:
    """Generate simulation datasets from a Hydra-configured simulator."""
    sim = build_simulator(cfg.simulator)

    splits = generate_dataset_splits(
        sim=sim,
        n_train=cfg.dataset.n_train,
        n_valid=cfg.dataset.n_valid,
        n_test=cfg.dataset.n_test,
        base_seed=cfg.run.seed,
    )

    output_dir = Path(cfg.dataset.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path(get_original_cwd()) / output_dir

    save_dataset_splits(
        splits=splits, output_dir=output_dir, overwrite=cfg.run.overwrite
    )


if __name__ == "__main__":
    main()
