"""Tests of the dataset configs of the spherical shallow water benchmark."""

from __future__ import annotations

import math
from importlib.resources import files

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

th = pytest.importorskip("torch_harmonics")
if not hasattr(th, "as_grid"):
    pytest.skip("torch-harmonics without grid descriptors", allow_module_level=True)

from autosim.cli import build_simulator  # noqa: E402
from autosim.experimental.simulations.shallow_water_sphere import (  # noqa: E402
    SphericalShallowWater,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIG_DIR = files("autosim") / "configs"
DATASET_CONFIGS = [
    "shallow_water_sphere_d1_galewsky",
    "shallow_water_sphere_d2_rossby_haurwitz",
    "shallow_water_sphere_d3_mountain",
    "shallow_water_sphere_d4_vortical",
    "shallow_water_sphere_d4_momentum",
    "shallow_water_sphere_d4_backscatter",
    "shallow_water_sphere_d5_galewsky_coarse",
    "shallow_water_sphere_d5_forced_coarse",
    "shallow_water_sphere_t1_williamson2",
]


@pytest.mark.parametrize("name", DATASET_CONFIGS)
def test_dataset_config_runs_at_a_small_size(name):
    path = CONFIG_DIR / "simulator" / "experimental" / f"{name}.yaml"
    config = OmegaConf.load(str(path))
    assert isinstance(config, DictConfig)
    assert config.lmax >= 86  # development size, or the fine run of a coarse dataset
    assert 2 * config.nlat >= 3 * config.lmax - 2  # products free of aliasing

    # the same dataset at a small size and for two saved intervals
    config.lmax = 22
    config.nlat = 32
    config.nlon = 64
    config.T = 2 * config.dt_save
    config.skip_nt = 0
    config.batch_size = 2
    config.device = DEVICE
    config.log_level = "warning"
    if config.get("output_lmax") is not None:
        config.output_lmax = 11
        config.output_nlat = 16
        config.output_nlon = 32
    if config.get("forcing_degree_max") is not None:
        config.forcing_degree_max = 10
    if config.get("reference_members", 0) > 0:
        config.reference_members = 2

    sim = build_simulator(config)
    assert isinstance(sim, SphericalShallowWater)
    payload = sim.forward_samples_spatiotemporal(n=2, random_seed=0)
    assert payload["data"].shape == (2, 3, *sim.output_grid.shape, 3)
    assert torch.isfinite(payload["data"]).all()
    assert payload["constant_fields"].shape == (2, *sim.output_grid.shape, 2)
    if config.get("output_lmax") is not None:
        assert payload["baseline"].shape == payload["data"].shape
    if config.get("reference_members", 0) > 0:
        assert payload["reference_ensemble"].shape[:2] == (2, 2)


def test_t1_generation_config_lists_the_four_tilts_of_w92():
    config = OmegaConf.load(str(CONFIG_DIR / "generate_data_swe_sphere_t1.yaml"))
    assert isinstance(config, DictConfig)
    assert config.stratify.enabled
    assert config.stratify.key == "simulator.parameters_range.tilt"
    # "values" is also a method of the config object, so the key is read with []
    tilts = [value[0] for value in config.stratify["values"]]
    expected = [0.0, 0.05, math.pi / 2 - 0.05, math.pi / 2]
    assert tilts == pytest.approx(expected)
    for count in (
        config.dataset.n_train,
        config.dataset.n_valid,
        config.dataset.n_test,
    ):
        assert count % 4 == 0
