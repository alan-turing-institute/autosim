from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from autosim.cli import build_simulator, generate_dataset_splits
from autosim.simulations import SIMULATOR_REGISTRY, GaussianField


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_native_rectangular_shape_and_provenance(dtype: str) -> None:
    sim = GaussianField(height=5, width=7, n_channels=3, n_steps=2, dtype=dtype)
    payload = sim.forward_samples_spatiotemporal(1, random_seed=17)
    assert payload["data"].shape == (1, 2, 5, 7, 3)
    assert payload["data"].dtype == getattr(torch, dtype)
    assert payload["data"].device.type == "cpu"
    assert torch.isfinite(payload["data"]).all()
    assert payload["provenance"]["kind"] == "synthetic"
    assert payload["provenance"]["random_seed"] == 17
    assert payload["constant_fields"] is None
    assert payload["constant_scalars"] is None
    assert sim.output_names == ["field_0", "field_1", "field_2"]
    assert SIMULATOR_REGISTRY["GaussianField"] is GaussianField


@pytest.mark.parametrize("seed", [None, 42])
def test_seed_is_replayable_without_changing_global_rng(seed: int | None) -> None:
    sim = GaussianField(
        height=8, width=9, rho=0.7, spatial_length_scale=0.5, channel_correlation=0.3
    )
    before = torch.random.get_rng_state().clone()
    result = sim.forward_samples_spatiotemporal(2, random_seed=seed)
    replay = sim.forward_samples_spatiotemporal(
        2, random_seed=result["provenance"]["random_seed"]
    )
    assert torch.equal(result["data"], replay["data"])
    assert torch.equal(before, torch.random.get_rng_state())


def test_splits_are_distinct_and_repeatable() -> None:
    sim = GaussianField(height=3, width=5, n_channels=2)
    first = generate_dataset_splits(sim, 2, 1, 1, base_seed=13)
    second = generate_dataset_splits(sim, 2, 1, 1, base_seed=13)
    for name in ("train", "valid", "test"):
        assert torch.equal(first[name]["data"], second[name]["data"])
    assert not torch.equal(first["train"]["data"][0], first["valid"]["data"][0])
    assert not torch.equal(first["valid"]["data"], first["test"]["data"])


@pytest.mark.parametrize("rho", [0.0, 0.8, -0.6])
def test_temporal_correlation_and_stationary_variance(rho: float) -> None:
    sim = GaussianField(height=32, width=40, n_channels=3, n_steps=24, rho=rho)
    data = sim.forward_samples_spatiotemporal(4, random_seed=7)["data"]
    assert abs(data.mean().item()) < 0.04
    assert abs(data.square().mean().item() - 1) < 0.04
    assert abs((data[:, 1:] * data[:, :-1]).mean().item() - rho) < 0.04
    assert abs(data[:, -1].square().mean().item() - 1) < 0.06


def test_channel_correlation_matches_requested_value() -> None:
    sim = GaussianField(
        height=64, width=65, n_channels=4, n_steps=8, channel_correlation=0.6
    )
    data = sim.forward_samples_spatiotemporal(3, random_seed=11)["data"]
    assert abs(data.square().mean().item() - 1) < 0.04
    for channel in range(1, 4):
        covariance = (data[..., 0] * data[..., channel]).mean().item()
        assert abs(covariance - 0.6) < 0.04
        assert not torch.equal(data[..., 0], data[..., channel])


def test_spatial_smoothing_preserves_variance_and_correlates_neighbours() -> None:
    sim = GaussianField(
        height=32, width=33, n_channels=3, n_steps=12, spatial_length_scale=1.2
    )
    data = sim.forward_samples_spatiotemporal(6, random_seed=5)["data"]
    variance = data.square().mean().item()
    assert abs(variance - 1) < 0.08
    for axis in (2, 3):
        correlation = (data * data.roll(1, dims=axis)).mean().item() / variance
        assert 0.7 < correlation < 0.95
    assert abs((data[..., 0] * data[..., 1]).mean().item()) < 0.08


def test_all_correlations_can_be_enabled_together() -> None:
    sim = GaussianField(
        height=32,
        width=33,
        n_channels=3,
        n_steps=16,
        rho=0.5,
        spatial_length_scale=0.7,
        channel_correlation=0.4,
    )
    data = sim.forward_samples_spatiotemporal(6, random_seed=19)["data"]
    assert abs(data.square().mean().item() - 1) < 0.06
    assert abs((data[:, 1:] * data[:, :-1]).mean().item() - 0.5) < 0.06
    assert abs((data[..., 0] * data[..., 1]).mean().item() - 0.4) < 0.06


def test_output_size_guard_precedes_random_allocation(monkeypatch) -> None:
    sim = GaussianField(max_output_bytes=1)

    def unexpected_allocation(*args, **kwargs):
        pytest.fail("Random data must not be allocated over the byte limit")

    monkeypatch.setattr(torch, "randn", unexpected_allocation)
    with pytest.raises(ValueError, match="max_output_bytes"):
        sim.forward_samples_spatiotemporal(1, random_seed=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"height": 0},
        {"width": -1},
        {"n_channels": True},
        {"n_steps": 1.5},
        {"rho": 1},
        {"rho": -1},
        {"rho": float("nan")},
        {"spatial_length_scale": -1},
        {"spatial_length_scale": float("inf")},
        {"channel_correlation": -0.1},
        {"channel_correlation": 1},
        {"dtype": "int32"},
        {"max_output_bytes": 0},
        {"output_names": ["x"]},
        {"output_names": ["x"] * 4},
    ],
)
def test_invalid_parameters(kwargs: dict) -> None:
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        GaussianField(**kwargs)


def test_cpu_generation_ignores_default_device() -> None:
    with torch.device("meta"):
        sim = GaussianField(height=3, width=5, spatial_length_scale=0.5)
        data = sim.forward_samples_spatiotemporal(1, random_seed=3)["data"]
    assert data.device.type == "cpu"
    assert torch.isfinite(data).all()


def test_base_forward_accepts_empty_parameter_vectors() -> None:
    sim = GaussianField(height=3, width=5, n_channels=2, n_steps=4)
    result = sim.forward(torch.empty(2, 0), allow_failures=False)
    assert result is not None
    assert result.shape == (2, 4 * 3 * 5 * 2)


@pytest.mark.parametrize("workers", [None, 2])
def test_base_batch_seed_is_repeatable_and_worker_invariant(
    workers: int | None,
) -> None:
    sim = GaussianField(height=3, width=5, n_channels=2, n_steps=4, rho=0.7)
    inputs = torch.empty(2, 0)
    expected, _ = sim.forward_batch(inputs, random_seed=21, allow_failures=False)
    actual, valid = sim.forward_batch(
        inputs, num_workers=workers, random_seed=21, allow_failures=False
    )
    other, _ = sim.forward_batch(inputs, random_seed=22, allow_failures=False)
    assert torch.equal(actual, expected)
    assert torch.equal(valid, inputs)
    assert not torch.equal(actual[0], actual[1])
    assert not torch.equal(actual, other)


@pytest.mark.parametrize(
    ("config_name", "shape"),
    [
        ("generate_data_synthetic", (16, 32, 4, 4)),
        ("generate_data_era5_shape", (121, 240, 69, 4)),
    ],
)
def test_synthetic_presets_compose_with_small_defaults(
    config_name: str, shape: tuple[int, int, int, int]
) -> None:
    with initialize_config_module(config_module="autosim.configs", version_base=None):
        cfg = compose(config_name=config_name)
    sim = build_simulator(cfg.simulator)
    assert isinstance(sim, GaussianField)
    assert (sim.height, sim.width, sim.n_channels, sim.n_steps) == shape
    assert (cfg.dataset.n_train, cfg.dataset.n_valid, cfg.dataset.n_test) == (2, 1, 1)
    assert not cfg.visualize.enabled


@pytest.mark.parametrize("stratified", [False, True])
def test_cli_exports_synthetic_splits_config_and_stats(
    tmp_path: Path, stratified: bool
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = tmp_path / "dataset"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "autosim.cli",
            "--config-name=generate_data_synthetic",
            f"dataset.output_dir={output_dir}",
            "simulator.height=5",
            "simulator.width=7",
            "simulator.n_channels=3",
            "simulator.rho=0.5",
            "simulator.spatial_length_scale=0.5",
            "simulator.channel_correlation=0.3",
            f"hydra.run.dir={tmp_path / 'hydra'}",
            "hydra.output_subdir=null",
            *(
                [
                    "stratify.enabled=true",
                    "stratify.key=simulator.rho",
                    "stratify.values=[0.0,0.8]",
                    "dataset.n_valid=2",
                    "dataset.n_test=2",
                ]
                if stratified
                else []
            ),
        ],
        check=True,
        cwd=repo_root,
    )
    train = torch.load(output_dir / "train/data.pt")
    assert train["data"].shape == (2, 4, 5, 7, 3)
    if stratified:
        assert [entry["rho"] for entry in train["provenance"]] == [0.0, 0.8]
        for index, provenance in enumerate(train["provenance"]):
            sim = GaussianField(
                height=5,
                width=7,
                n_channels=3,
                rho=provenance["rho"],
                spatial_length_scale=0.5,
                channel_correlation=0.3,
            )
            expected = sim.forward_samples_spatiotemporal(
                1, random_seed=provenance["random_seed"]
            )
            assert torch.equal(train["data"][index], expected["data"][0])
    else:
        assert train["provenance"]["kind"] == "synthetic"
    for split in ("valid", "test"):
        payload = torch.load(output_dir / split / "data.pt")
        assert payload["data"].shape == (2 if stratified else 1, 4, 5, 7, 3)
    stats = OmegaConf.load(output_dir / "stats.yml")
    assert list(stats.core_field_names) == ["field_0", "field_1", "field_2"]
    expected = train["data"].mean(dim=(0, 1, 2, 3))
    for idx, name in enumerate(stats.core_field_names):
        assert stats.stats.mean[name] == pytest.approx(expected[idx].item(), rel=0.001)
    assert (output_dir / "resolved_config.yaml").exists()
