"""Tests of the dataset configs of the spherical shallow water benchmark."""

from __future__ import annotations

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
    "shallow_water_sphere_ds1_galewsky",
    "shallow_water_sphere_ds2_galewsky_large",
    "shallow_water_sphere_ds3_galewsky_small",
    "shallow_water_sphere_ds4_mountain",
]
# Configs of the "era5" family, which read the WeatherBench2 store
ERA5_CONFIGS = [
    "shallow_water_sphere_ds5_era5_weather",
    "shallow_water_sphere_ds6_era5_climate",
]


@pytest.mark.parametrize(
    "name",
    DATASET_CONFIGS
    + [pytest.param(name, marks=pytest.mark.network) for name in ERA5_CONFIGS],
)
def test_dataset_config_runs_at_a_small_size(name):
    path = CONFIG_DIR / "simulator" / "experimental" / f"{name}.yaml"
    config = OmegaConf.load(str(path))
    assert isinstance(config, DictConfig)
    assert config.lmax == 256  # benchmark size
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
    skebs = config.get("skebs")
    if skebs is not None and skebs.get("degree_max") is not None:
        # the forced band scaled with the band limit, from 256 to 22
        lowest = max(1, round(skebs.get("degree_min", 1) * 22 / 256))
        highest = min(21, round(skebs.degree_max * 22 / 256))
        skebs.degree_min = lowest
        skebs.degree_max = max(lowest, highest)

    sim = build_simulator(config)
    assert isinstance(sim, SphericalShallowWater)
    payload = sim.forward_samples_spatiotemporal(n=2, random_seed=0)
    assert payload["data"].shape == (2, 3, *sim.output_grid.shape, 3)
    assert torch.isfinite(payload["data"]).all()
    assert payload["constant_fields"].shape == (2, *sim.output_grid.shape, 2)
    if config.get("output_lmax") is not None:
        assert payload["baseline"].shape == payload["data"].shape
    if config.get("return_periodic_forcing", False):
        shape = (2, 3, *sim.output_grid.shape, 1)
        assert payload["periodic_forcing"].shape == shape
