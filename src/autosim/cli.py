from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf

from autosim.simulations.base import Simulator


def _build_simulator_registry() -> dict[str, type[Simulator]]:
    registry: dict[str, type[Simulator]] = {}

    modules = [
        importlib.import_module("autosim.simulations"),
        importlib.import_module("autosim.experimental.simulations"),
    ]

    for module in modules:
        exported = getattr(module, "__all__", [])
        for name in exported:
            obj = getattr(module, name, None)
            if inspect.isclass(obj) and issubclass(obj, Simulator):
                registry[name] = obj

    return registry


def resolve_simulator_class(simulator_name: str) -> type[Simulator]:
    """Resolve a simulator class by name or fully-qualified class path."""
    registry = _build_simulator_registry()

    if simulator_name in registry:
        return registry[simulator_name]

    if "." in simulator_name:
        module_name, class_name = simulator_name.rsplit(".", 1)
        module = importlib.import_module(module_name)
        simulator_cls = getattr(module, class_name, None)
        if inspect.isclass(simulator_cls) and issubclass(simulator_cls, Simulator):
            return simulator_cls

    available = ", ".join(sorted(registry))
    msg = f"Unknown simulator '{simulator_name}'. Available simulators: {available}."
    raise ValueError(msg)


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


_default_cfg = OmegaConf.create(
    {
        "simulator": {
            "name": "ShallowWater2D",
            "kwargs": {
                "return_timeseries": True,
                "log_level": "warning",
            },
        },
        "dataset": {
            "output_dir": "examples/experimental/generated_datasets/shallow_water",
            "n_train": 200,
            "n_valid": 20,
            "n_test": 20,
        },
        "run": {
            "seed": None,
            "overwrite": False,
        },
    }
)
OmegaConf.set_struct(_default_cfg.simulator.kwargs, False)

_config_store = ConfigStore.instance()
_config_store.store(name="generate_data", node=_default_cfg)


@hydra.main(version_base=None, config_name="generate_data")
def main(cfg: Any) -> None:
    """Generate simulation datasets from a Hydra-configured simulator."""
    simulator_cls = resolve_simulator_class(cfg.simulator.name)
    simulator_kwargs = {
        key: value for key, value in cfg.simulator.kwargs.items() if value is not None
    }
    sim = simulator_cls(**simulator_kwargs)

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
