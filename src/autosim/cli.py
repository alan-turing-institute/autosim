from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path
from typing import Any, cast

import hydra
import torch
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf

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
    ensure_exact_n: bool = False,
) -> dict[str, dict[str, Any]]:
    """Generate train/valid/test splits from a simulator."""
    # Reserve disjoint seed ranges so retries in one split cannot collide with
    # initial or retry seeds in another.
    split_seed_stride = (
        SpatioTemporalSimulator._retry_budget(max(n_train, n_valid, n_test)) + 1
    )

    def get_seed(offset: int) -> int | None:
        if base_seed is None:
            return None
        return base_seed + offset * split_seed_stride

    train = sim.forward_samples_spatiotemporal(
        n=n_train,
        random_seed=get_seed(0),
        ensure_exact_n=ensure_exact_n,
    )
    valid = sim.forward_samples_spatiotemporal(
        n=n_valid,
        random_seed=get_seed(1),
        ensure_exact_n=ensure_exact_n,
    )
    test = sim.forward_samples_spatiotemporal(
        n=n_test,
        random_seed=get_seed(2),
        ensure_exact_n=ensure_exact_n,
    )
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


def save_resolved_config(cfg: Any, output_dir: str | Path) -> None:
    """Persist the fully resolved Hydra config next to generated datasets."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    resolved_cfg_path = output_path / "resolved_config.yaml"
    resolved_yaml = OmegaConf.to_yaml(cfg, resolve=True)
    resolved_cfg_path.write_text(resolved_yaml, encoding="utf-8")


def save_example_videos(
    splits: dict[str, dict[str, Any]],
    output_dir: str | Path,
    visualize_cfg: Any | None,
    channel_names: list[str] | None = None,
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
    preserve_aspect = bool(visualize_cfg.get("preserve_aspect", False))

    configured_channel_names = visualize_cfg.get("channel_names", None)
    resolved_channel_names = channel_names
    if configured_channel_names is not None:
        resolved_channel_names = [str(name) for name in configured_channel_names]

    for idx in batch_indices:
        save_path = videos_dir / f"batch_{idx}.{file_ext}"
        if save_path.exists() and not overwrite:
            continue
        plot_spatiotemporal_video(
            true=data,
            batch_idx=idx,
            fps=fps,
            save_path=str(save_path),
            channel_names=resolved_channel_names,
            preserve_aspect=preserve_aspect,
        )


def _parse_field_names_csv(field_names_csv: str | None) -> list[str] | None:
    """Parse a comma-separated field-name string into a cleaned list."""
    if field_names_csv is None:
        return None
    names = [name.strip() for name in field_names_csv.split(",") if name.strip()]
    return names if names else None


def _infer_core_field_names_from_resolved_config(
    dataset_dir: Path, n_channels: int
) -> list[str] | None:
    """Infer channel names from `resolved_config.yaml` when available."""
    resolved_cfg_path = dataset_dir / "resolved_config.yaml"
    if not resolved_cfg_path.exists():
        return None
    try:
        cfg = OmegaConf.load(resolved_cfg_path)
        assert isinstance(cfg, DictConfig)
        simulator_cfg = cfg.get("simulator")
        if simulator_cfg is None:
            return None
        sim = build_simulator(simulator_cfg)
        inferred_names = [str(name) for name in sim.output_names]
    except Exception:
        return None

    if len(inferred_names) != n_channels:
        return None
    return inferred_names


def compute_normalization_stats(
    split_payload: dict[str, Any],
    core_field_names: list[str] | None = None,
    constant_field_names: list[str] | None = None,
) -> dict[str, Any]:
    """Compute normalization statistics for one split payload."""
    data = split_payload.get("data")
    if not isinstance(data, torch.Tensor) or data.ndim != 5:
        msg = (
            "Normalization stats require split payload 'data' as a 5D torch.Tensor "
            "with shape [batch,time,x,y,channels]."
        )
        raise ValueError(msg)

    _, n_time, _, _, n_channels = data.shape
    if n_time < 2:
        msg = (
            "Normalization delta stats require at least 2 time steps in "
            "split payload 'data'."
        )
        raise ValueError(msg)

    resolved_core_field_names = core_field_names
    if resolved_core_field_names is None:
        resolved_core_field_names = [f"field_{idx}" for idx in range(n_channels)]
    if len(resolved_core_field_names) != n_channels:
        msg = (
            "Number of core field names must match data channel count. "
            f"Received {len(resolved_core_field_names)} names "
            f"for {n_channels} channels."
        )
        raise ValueError(msg)

    deltas = data[:, 1:, ...] - data[:, :-1, ...]

    flattened_data = data.reshape(-1, n_channels)
    flattened_deltas = deltas.reshape(-1, n_channels)
    mean = flattened_data.mean(dim=0)
    std = flattened_data.std(dim=0, unbiased=False)
    mean_delta = flattened_deltas.mean(dim=0)
    std_delta = flattened_deltas.std(dim=0, unbiased=False)

    def _stats_by_channel(values: torch.Tensor) -> dict[str, float]:
        return {
            name: float(values[idx].detach().cpu().item())
            for idx, name in enumerate(resolved_core_field_names or [])
        }

    return {
        "stats": {
            "mean": _stats_by_channel(mean),
            "std": _stats_by_channel(std),
            "mean_delta": _stats_by_channel(mean_delta),
            "std_delta": _stats_by_channel(std_delta),
        },
        "core_field_names": resolved_core_field_names,
        "constant_field_names": constant_field_names or [],
    }


def _robust_zscore(values: torch.Tensor) -> torch.Tensor:
    median = values.median()
    mad = (values - median).abs().median()
    scale = 1.4826 * mad + 1e-12
    return (values - median) / scale


def _iqr_outlier_mask(values: torch.Tensor, iqr_multiplier: float) -> torch.Tensor:
    if iqr_multiplier <= 0:
        msg = "iqr_multiplier must be positive."
        raise ValueError(msg)
    q1 = torch.quantile(values, 0.25)
    q3 = torch.quantile(values, 0.75)
    iqr = q3 - q1
    lower = q1 - iqr_multiplier * iqr
    upper = q3 + iqr_multiplier * iqr
    return (values < lower) | (values > upper)


def compute_split_outlier_report(
    split_payload: dict[str, Any],
    robust_z_threshold: float = 3.5,
    iqr_multiplier: float = 3.0,
) -> dict[str, Any]:
    """Compute per-run outlier report from a split payload.

    Outlier signals are based on trajectory-level mean and variance across all
    dimensions except batch.
    """
    if robust_z_threshold <= 0:
        msg = "robust_z_threshold must be positive."
        raise ValueError(msg)

    data = split_payload.get("data")
    if not isinstance(data, torch.Tensor) or data.ndim != 5:
        msg = (
            "Outlier report requires split payload 'data' as a 5D torch.Tensor "
            "with shape [batch,time,x,y,channels]."
        )
        raise ValueError(msg)

    data_float = data.float()
    per_run_mean = data_float.mean(dim=(1, 2, 3, 4))
    per_run_var = data_float.var(dim=(1, 2, 3, 4), unbiased=False)

    z_mean = _robust_zscore(per_run_mean)
    z_var = _robust_zscore(per_run_var)
    robust_mask = (z_mean.abs() > robust_z_threshold) | (
        z_var.abs() > robust_z_threshold
    )

    iqr_mask = _iqr_outlier_mask(per_run_mean, iqr_multiplier) | _iqr_outlier_mask(
        per_run_var, iqr_multiplier
    )

    robust_indices = torch.where(robust_mask)[0].tolist()
    iqr_indices = torch.where(iqr_mask)[0].tolist()

    return {
        "shape": list(data.shape),
        "n_runs": int(data.shape[0]),
        "mean_range": {
            "min": float(per_run_mean.min().item()),
            "max": float(per_run_mean.max().item()),
        },
        "var_range": {
            "min": float(per_run_var.min().item()),
            "max": float(per_run_var.max().item()),
        },
        "robust_z": {
            "threshold": float(robust_z_threshold),
            "count": len(robust_indices),
            "indices": robust_indices,
        },
        "iqr": {
            "multiplier": float(iqr_multiplier),
            "count": len(iqr_indices),
            "indices": iqr_indices,
        },
    }


def compute_dataset_outlier_report(
    dataset_dir: Path,
    splits: list[str] | None = None,
    robust_z_threshold: float = 3.5,
    iqr_multiplier: float = 3.0,
) -> dict[str, Any]:
    """Compute a per-split outlier report for an existing dataset directory."""
    resolved_splits = splits or ["train", "valid", "test"]
    if not resolved_splits:
        msg = "At least one split must be provided."
        raise ValueError(msg)

    report: dict[str, Any] = {
        "method": "per-run mean/variance outlier detection",
        "splits": {},
    }

    total_robust = 0
    total_iqr = 0
    for split in resolved_splits:
        split_data_path = dataset_dir / split / "data.pt"
        if not split_data_path.exists():
            msg = f"Could not find split file '{split_data_path}'."
            raise FileNotFoundError(msg)
        split_payload = torch.load(split_data_path, map_location="cpu")
        if not isinstance(split_payload, dict):
            msg = f"Expected dict payload in '{split_data_path}'."
            raise ValueError(msg)

        split_report = compute_split_outlier_report(
            split_payload=split_payload,
            robust_z_threshold=robust_z_threshold,
            iqr_multiplier=iqr_multiplier,
        )
        report["splits"][split] = split_report
        total_robust += int(split_report["robust_z"]["count"])
        total_iqr += int(split_report["iqr"]["count"])

    report["totals"] = {
        "robust_z_count": total_robust,
        "iqr_count": total_iqr,
    }
    return report


def _print_outlier_report_summary(report: dict[str, Any]) -> None:
    """Print a concise per-split summary, including zero outlier counts."""
    totals = report.get("totals", {})
    robust_total = int(totals.get("robust_z_count", 0))
    iqr_total = int(totals.get("iqr_count", 0))
    print(f"totals: robust_z={robust_total}, iqr={iqr_total}")

    splits = report.get("splits", {})
    if isinstance(splits, dict):
        for split_name, split_report in splits.items():
            if not isinstance(split_report, dict):
                continue
            robust_info = split_report.get("robust_z", {})
            iqr_info = split_report.get("iqr", {})
            robust_count = int(robust_info.get("count", 0))
            iqr_count = int(iqr_info.get("count", 0))
            n_runs = int(split_report.get("n_runs", 0))
            print(
                f"{split_name}: n_runs={n_runs}, robust_z={robust_count}, "
                f"iqr={iqr_count}"
            )


def generate_outlier_report_yaml(
    dataset_dir: Path,
    splits: list[str] | None = None,
    output_path: Path | None = None,
    robust_z_threshold: float = 3.5,
    iqr_multiplier: float = 3.0,
    sig_figs: int = 4,
) -> Path:
    """Generate a YAML outlier report for one or more dataset splits."""
    report = compute_dataset_outlier_report(
        dataset_dir=dataset_dir,
        splits=splits,
        robust_z_threshold=robust_z_threshold,
        iqr_multiplier=iqr_multiplier,
    )

    resolved_output_path = (
        output_path if output_path is not None else dataset_dir / "outliers.yml"
    )
    save_normalization_stats(
        stats_payload=report,
        output_path=resolved_output_path,
        sig_figs=sig_figs,
    )
    return resolved_output_path


def _round_sigfigs(value: float, sig_figs: int) -> float:
    """Round a float to a fixed number of significant figures."""
    if sig_figs <= 0:
        msg = "sig_figs must be positive."
        raise ValueError(msg)
    if value == 0.0:
        return 0.0
    # General format preserves significant figures; may emit scientific notation.
    return float(f"{value:.{sig_figs}g}")


def _rounded_normalization_stats_payload(
    stats_payload: dict[str, Any], sig_figs: int
) -> dict[str, Any]:
    """Return a copy of stats_payload with rounded float stat values."""
    rounded = cast(
        dict[str, Any],
        OmegaConf.to_container(OmegaConf.create(stats_payload), resolve=True),
    )

    stats = rounded.get("stats")
    if not isinstance(stats, dict):
        return rounded

    for key in ("mean", "std", "mean_delta", "std_delta"):
        bucket = stats.get(key)
        if not isinstance(bucket, dict):
            continue
        for field_name, field_value in list(bucket.items()):
            if isinstance(field_value, int | float):
                bucket[field_name] = _round_sigfigs(float(field_value), sig_figs)

    return rounded


def save_normalization_stats(
    stats_payload: dict[str, Any],
    output_path: Path,
    sig_figs: int = 4,
) -> None:
    """Persist normalization statistics as YAML."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rounded_payload = _rounded_normalization_stats_payload(
        stats_payload=stats_payload, sig_figs=sig_figs
    )
    yaml_payload = OmegaConf.to_yaml(OmegaConf.create(rounded_payload), resolve=True)
    output_path.write_text(yaml_payload, encoding="utf-8")


def generate_normalization_stats_yaml(
    dataset_dir: Path,
    split: str = "train",
    output_path: Path | None = None,
    core_field_names: list[str] | None = None,
    sig_figs: int = 4,
) -> Path:
    """Generate normalization-stats YAML from an existing dataset directory."""
    split_data_path = dataset_dir / split / "data.pt"
    if not split_data_path.exists():
        msg = f"Could not find split file '{split_data_path}'."
        raise FileNotFoundError(msg)
    split_payload = torch.load(split_data_path, map_location="cpu")
    if not isinstance(split_payload, dict):
        msg = f"Expected dict payload in '{split_data_path}'."
        raise ValueError(msg)

    payload_data = split_payload.get("data")
    if not isinstance(payload_data, torch.Tensor) or payload_data.ndim != 5:
        msg = (
            "Expected split payload 'data' as a 5D torch.Tensor with shape "
            "[batch,time,x,y,channels]."
        )
        raise ValueError(msg)

    resolved_field_names = core_field_names
    if resolved_field_names is None:
        resolved_field_names = _infer_core_field_names_from_resolved_config(
            dataset_dir=dataset_dir,
            n_channels=payload_data.shape[-1],
        )
    stats_payload = compute_normalization_stats(
        split_payload=split_payload,
        core_field_names=resolved_field_names,
    )

    resolved_output_path = (
        output_path if output_path is not None else dataset_dir / "stats.yml"
    )
    save_normalization_stats(
        stats_payload=stats_payload,
        output_path=resolved_output_path,
        sig_figs=sig_figs,
    )
    return resolved_output_path


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
    channel_names_for_visualization: list[str] | None = None
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
            if channel_names_for_visualization is None:
                channel_names_for_visualization = list(sim.output_names)
            splits = generate_dataset_splits(
                sim=sim,
                n_train=n_train_each,
                n_valid=n_valid_each,
                n_test=n_test_each,
                base_seed=cfg.seed,
                ensure_exact_n=bool(cfg.dataset.get("ensure_exact_n", False)),
            )
            per_strata_outputs.append(splits)

        splits = combine_stratified_splits(per_strata_outputs)
    else:
        sim = build_simulator(cfg.simulator)
        channel_names_for_visualization = list(sim.output_names)

        splits = generate_dataset_splits(
            sim=sim,
            n_train=cfg.dataset.n_train,
            n_valid=cfg.dataset.n_valid,
            n_test=cfg.dataset.n_test,
            base_seed=cfg.seed,
            ensure_exact_n=bool(cfg.dataset.get("ensure_exact_n", False)),
        )

    output_dir = Path(cfg.dataset.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path(get_original_cwd()) / output_dir

    save_resolved_config(cfg=cfg, output_dir=output_dir)

    save_dataset_splits(splits=splits, output_dir=output_dir, overwrite=cfg.overwrite)
    normalization_stats_payload = compute_normalization_stats(
        split_payload=splits["train"],
        core_field_names=channel_names_for_visualization,
    )
    save_normalization_stats(
        stats_payload=normalization_stats_payload,
        output_path=output_dir / "stats.yml",
    )
    save_example_videos(
        splits=splits,
        output_dir=output_dir,
        visualize_cfg=cfg.get("visualize"),
        channel_names=channel_names_for_visualization,
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
    - `autosim stats` writes normalization stats YAML for an existing dataset.
    - `autosim` (or any Hydra overrides) runs data generation.
    """
    argv = sys.argv[1:]

    if argv and argv[0] in {"-h", "--help"}:
        parser = argparse.ArgumentParser(
            prog="autosim",
            description=(
                "Generate simulation datasets using Hydra overrides, or list "
                "available simulator configs, or compute posthoc dataset reports."
            ),
        )
        parser.add_argument(
            "command",
            nargs="?",
            help=(
                "Subcommand: 'list', 'stats', or 'outliers'. Omit to run data "
                "generation with Hydra."
            ),
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

    if argv and argv[0] == "stats":
        stats_parser = argparse.ArgumentParser(
            prog="autosim stats",
            description=(
                "Generate normalization_stats YAML for an existing dataset directory."
            ),
        )
        stats_parser.add_argument(
            "dataset_dir",
            help="Dataset root containing split folders such as train/data.pt.",
        )
        stats_parser.add_argument(
            "--split",
            default="train",
            help="Split to use for stats (default: train).",
        )
        stats_parser.add_argument(
            "--output",
            default=None,
            help=("Optional output YAML path (default: <dataset_dir>/stats.yml)."),
        )
        stats_parser.add_argument(
            "--field-names",
            default=None,
            help=(
                "Optional comma-separated core field names, e.g. 'U,V'. "
                "If omitted, names are inferred from resolved_config.yaml "
                "when possible."
            ),
        )
        stats_parser.add_argument(
            "--sig-figs",
            type=int,
            default=4,
            help="Significant figures for float stats in YAML (default: 4).",
        )
        args = stats_parser.parse_args(argv[1:])

        output_path = Path(args.output) if args.output is not None else None
        written_path = generate_normalization_stats_yaml(
            dataset_dir=Path(args.dataset_dir),
            split=str(args.split),
            output_path=output_path,
            core_field_names=_parse_field_names_csv(args.field_names),
            sig_figs=int(args.sig_figs),
        )
        print(written_path.as_posix())
        return

    if argv and argv[0] == "outliers":
        outliers_parser = argparse.ArgumentParser(
            prog="autosim outliers",
            description=(
                "Generate per-split outlier report YAML for an existing dataset "
                "directory."
            ),
        )
        outliers_parser.add_argument(
            "dataset_dir",
            help="Dataset root containing split folders such as train/data.pt.",
        )
        outliers_parser.add_argument(
            "--splits",
            default="train,valid,test",
            help="Comma-separated split names (default: train,valid,test).",
        )
        outliers_parser.add_argument(
            "--output",
            default=None,
            help="Optional output YAML path (default: <dataset_dir>/outliers.yml).",
        )
        outliers_parser.add_argument(
            "--robust-z-threshold",
            type=float,
            default=3.5,
            help="Absolute robust-z threshold for outlier flagging (default: 3.5).",
        )
        outliers_parser.add_argument(
            "--iqr-multiplier",
            type=float,
            default=3.0,
            help="IQR fence multiplier for outlier flagging (default: 3.0).",
        )
        outliers_parser.add_argument(
            "--sig-figs",
            type=int,
            default=4,
            help="Significant figures for float stats in YAML (default: 4).",
        )
        outliers_parser.add_argument(
            "--fail-on-outliers",
            action="store_true",
            help="Exit with non-zero status if any outliers are found.",
        )
        outliers_parser.add_argument(
            "--print-only",
            action="store_true",
            help="Print outlier summary to stdout without writing a YAML file.",
        )
        args = outliers_parser.parse_args(argv[1:])

        splits = _parse_field_names_csv(str(args.splits))
        report = compute_dataset_outlier_report(
            dataset_dir=Path(args.dataset_dir),
            splits=splits,
            robust_z_threshold=float(args.robust_z_threshold),
            iqr_multiplier=float(args.iqr_multiplier),
        )

        _print_outlier_report_summary(report)

        totals = report.get("totals")
        robust_total = int(totals.get("robust_z_count", 0)) if totals else 0
        iqr_total = int(totals.get("iqr_count", 0)) if totals else 0

        if not args.print_only:
            output_path = Path(args.output) if args.output is not None else None
            written_path = (
                output_path
                if output_path is not None
                else Path(args.dataset_dir) / "outliers.yml"
            )
            save_normalization_stats(
                stats_payload=report,
                output_path=written_path,
                sig_figs=int(args.sig_figs),
            )
            print(written_path.as_posix())

        if args.fail_on_outliers and (robust_total > 0 or iqr_total > 0):
            raise SystemExit(2)
        return

    # Preserve all original arguments for Hydra's own parser.
    sys.argv = [sys.argv[0], *argv]
    _generate_main()


if __name__ == "__main__":
    main()
