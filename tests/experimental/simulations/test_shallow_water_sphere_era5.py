"""Tests of the ERA5 family of the spherical shallow water simulator (W92 case 7)."""

from __future__ import annotations

import math

import pytest
import torch

th = pytest.importorskip("torch_harmonics")
if not hasattr(th, "as_grid"):
    pytest.skip("torch-harmonics without grid descriptors", allow_module_level=True)

from autosim.experimental.simulations.shallow_water_sphere import (  # noqa: E402
    ERA5_RANGES,
    Era5Fields,
    SphereShallowWaterDynamics,
    SphericalShallowWater,
    era5_hours,
    era5_initial_condition,
    era5_source_dynamics,
    era5_time_index,
    load_era5_fields,
    orography_filter,
    shallow_water_tendencies,
)
from autosim.experimental.simulations.shallow_water_sphere import (  # noqa: E402
    simulator as simulator_module,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RADIUS = 6.37122e6
OMEGA = 7.292e-5
GRAVITY = 9.80616
MEAN_DEPTH = 6000.0
# The source grid stands for the 721 x 1440 grid of the store: equiangular, with the
# poles, latitudes from north to south and longitudes from 0
SOURCE_NLAT = 49
SOURCE_NLON = 96


def _dynamics(lmax=22, nlat=32, nlon=64):
    grid = th.as_grid("legendre-gauss", nlat=nlat, nlon=nlon)
    return SphereShallowWaterDynamics(
        grid, lmax, RADIUS, OMEGA, GRAVITY, MEAN_DEPTH, 300.0, DEVICE
    )


def _synthetic_fields(tilt=0.7, batch=2, mountain=0.0):
    """Band-limited fields on the source grid: a tilted solid-body wind (W92 case 2,
    which has a northward component), a free surface of degree 2 and an orography of
    degree 1 and height ``mountain``."""
    latitude = torch.linspace(
        math.pi / 2, -math.pi / 2, SOURCE_NLAT, dtype=torch.float64
    )
    longitude = torch.arange(SOURCE_NLON, dtype=torch.float64) * 2 * math.pi
    longitude = longitude / SOURCE_NLON
    lat = latitude.reshape(-1, 1)
    lon = longitude.reshape(1, -1)
    u0 = 30.0
    eastward = u0 * (
        torch.cos(lat) * math.cos(tilt)
        + torch.cos(lon) * torch.sin(lat) * math.sin(tilt)
    )
    northward = -u0 * torch.sin(lon) * math.sin(tilt) * torch.ones_like(lat)
    free_surface = GRAVITY * (5600.0 - 300.0 * torch.sin(lat) ** 2 + 0 * lon)
    surface = GRAVITY * mountain * (1 + torch.cos(lat) * torch.cos(lon)) / 2
    return Era5Fields(
        geopotential=free_surface.expand(batch, -1, -1).clone(),
        eastward=eastward.expand(batch, -1, -1).clone(),
        northward=northward.expand(batch, -1, -1).clone(),
        surface_geopotential=surface,
        latitude=latitude,
        longitude=longitude,
    )


# ---------------------------------------------------------------------------
# Times of the store
# ---------------------------------------------------------------------------


def test_era5_hours_and_time_index():
    assert era5_hours("1959-01-01T00") == 0.0
    assert era5_hours("1978-12-21T00") == 175056.0  # W92 case 7
    index = era5_time_index(torch.tensor([-50.0, 2.9, 3.1, 175056.0, 1e9]))
    assert index.tolist() == [0, 0, 1, 29176, 93543]
    assert ERA5_RANGES["era5_time"] == (0.0, 6.0 * 93543)


# ---------------------------------------------------------------------------
# Regridding and balance, on synthetic fields
# ---------------------------------------------------------------------------


def test_regridded_wind_keeps_its_vorticity():
    """The vorticity from the source grid is that of the same wind on the solver grid.

    The tilted solid-body wind has degree 1, so both analyses are exact; a wrong sign
    of the northward wind, or a flipped latitude order, would change the vorticity.
    """
    dynamics = _dynamics()
    fields = _synthetic_fields()
    source = era5_source_dynamics(dynamics, fields)
    state, surface, valid = era5_initial_condition(dynamics, source, fields)

    tilt = 0.7
    latitude = dynamics.latitude.unsqueeze(0)
    longitude = dynamics.longitude.unsqueeze(0)
    eastward = 30.0 * (
        torch.cos(latitude) * math.cos(tilt)
        + torch.cos(longitude) * torch.sin(latitude) * math.sin(tilt)
    )
    northward = (-30.0 * torch.sin(longitude) * math.sin(tilt)).expand_as(eastward)
    vorticity, _ = dynamics.vorticity_divergence(torch.stack([eastward, northward], 1))
    error = (state[:, 1] - vorticity).abs().max()
    assert error < 1e-12 * vorticity.abs().max()
    assert surface.abs().max() == 0.0


def test_balance_removes_the_divergence_and_keeps_the_mean_free_surface():
    dynamics = _dynamics()
    fields = _synthetic_fields(mountain=1500.0)
    source = era5_source_dynamics(dynamics, fields)
    state, surface, valid = era5_initial_condition(dynamics, source, fields)
    assert valid.all()
    assert state[:, 2].abs().max() == 0.0

    # the global mean of the free surface is that of the fields
    free_surface = state[:, 0, 0, 0] + surface[:, 0, 0]
    expected = source.to_coefficients(fields.geopotential.to(DEVICE))[:, 0, 0]
    torch.testing.assert_close(free_surface, expected, rtol=1e-12, atol=0.0)

    # balance: the divergence tendency vanishes, unlike with the supplied height
    tendency = shallow_water_tendencies(dynamics, state, None, surface)
    unbalanced = state.clone()
    supplied = source.to_coefficients(fields.geopotential.to(DEVICE))
    unbalanced[:, 0] = supplied[:, : dynamics.lmax, : dynamics.lmax] - surface
    unbalanced_tendency = shallow_water_tendencies(dynamics, unbalanced, None, surface)
    assert tendency[:, 2].abs().max() < 1e-10 * unbalanced_tendency[:, 2].abs().max()


def test_orography_above_the_free_surface_is_rejected():
    dynamics = _dynamics()
    source = era5_source_dynamics(dynamics, _synthetic_fields())
    _, _, valid = era5_initial_condition(dynamics, source, _synthetic_fields())
    assert valid.all()
    deep = _synthetic_fields(mountain=8000.0)
    _, _, valid = era5_initial_condition(dynamics, source, deep)
    assert not valid.any()
    # a shallower mountain passes the check, and a higher threshold rejects it
    _, _, valid = era5_initial_condition(dynamics, source, _synthetic_fields(1500.0))
    assert valid.all()
    _, _, valid = era5_initial_condition(
        dynamics, source, _synthetic_fields(1500.0), min_depth=6000.0
    )
    assert not valid.any()


def test_orography_filter():
    """exp(-(l/28)^4): 1 at l = 0, 1/e at l = 28, and the same weights at every L."""
    weights = orography_filter(86)[:, 0]
    assert weights[0].item() == 1.0
    assert weights[28].item() == pytest.approx(math.exp(-1.0), rel=1e-12)
    assert weights[56].item() == pytest.approx(math.exp(-16.0), rel=1e-12)
    torch.testing.assert_close(orography_filter(256)[:86, 0], weights)


def test_fields_on_another_grid_raise():
    dynamics = _dynamics()
    fields = _synthetic_fields()
    fields.latitude = fields.latitude.flip(0)  # from south to north
    with pytest.raises(ValueError, match="equiangular grid"):
        era5_source_dynamics(dynamics, fields)


# ---------------------------------------------------------------------------
# Through the simulator, with synthetic fields in place of the store
# ---------------------------------------------------------------------------


def _era5_simulator(monkeypatch, mountain, batch, **options):
    """Simulator of the era5 family reading synthetic fields in place of the store."""

    def fake_fields(hours, _store, _level):
        return _synthetic_fields(mountain=mountain, batch=hours.shape[0])

    monkeypatch.setattr(simulator_module, "load_era5_fields", fake_fields)
    return SphericalShallowWater(
        lmax=22,
        nlat=32,
        nlon=64,
        T=6.0,
        dt_save=3.0,
        skip_nt=0,
        batch_size=batch,
        device=DEVICE,
        log_level="warning",
        initial_condition="era5",
        mean_depth=MEAN_DEPTH,
        parameters_range={"era5_time": (175056.0, 175056.0)},
        **options,
    )


def test_rejected_dates_are_dropped(monkeypatch):
    """A date whose balanced depth is below era5_min_depth gives no simulation."""
    sim = _era5_simulator(monkeypatch, 1500.0, 2, era5_min_depth=0.0)
    y, x = sim.forward_batch(sim.sample_inputs(2, random_seed=0), random_seed=0)
    assert y.shape[0] == 2
    assert torch.isfinite(y).all()
    sim = _era5_simulator(monkeypatch, 1500.0, 2, era5_min_depth=6000.0)
    y, x = sim.forward_batch(sim.sample_inputs(2, random_seed=0), random_seed=0)
    assert y.shape[0] == 0
    assert x.shape[0] == 0


def test_climatology_is_the_relaxation_target(monkeypatch):
    """The equilibrium is the analysed climatology minus the filtered orography."""

    def fake_climatology(hours, _store, _level):
        return _synthetic_fields(batch=hours.shape[0]).geopotential * 1.01

    monkeypatch.setattr(simulator_module, "load_era5_climatology", fake_climatology)
    sim = _era5_simulator(
        monkeypatch,
        1500.0,
        1,
        relaxation_time=24.0,
        relaxation_target="era5_climatology",
        drag_time=2400.0,
    )
    rows = sim.sample_inputs(1, random_seed=0)
    state, coriolis, surface = sim.initial_problem(rows, [])
    _, relaxation_time, equilibrium = sim.damping_settings(rows, coriolis, surface)
    assert relaxation_time is not None
    assert equilibrium is not None
    expected_free = sim.era5_source.to_coefficients(
        fake_climatology(torch.zeros(1), "", 500).to(DEVICE)
    )
    torch.testing.assert_close(
        equilibrium + surface, torch.tril(expected_free), rtol=1e-12, atol=1e-6
    )
    payload = sim.forward_samples_spatiotemporal(n=1, random_seed=0)
    assert torch.isfinite(payload["data"]).all()


def test_climatology_target_needs_the_era5_family():
    with pytest.raises(ValueError, match="era5"):
        SphericalShallowWater(
            lmax=22,
            nlat=32,
            nlon=64,
            T=6.0,
            dt_save=3.0,
            skip_nt=0,
            device=DEVICE,
            log_level="warning",
            relaxation_time=24.0,
            relaxation_target="era5_climatology",
        )


# ---------------------------------------------------------------------------
# The WeatherBench2 store (network)
# ---------------------------------------------------------------------------


@pytest.mark.network
def test_store_gives_the_w92_case_7_state():
    """1978-12-21T00, the first state of W92 case 7, through the simulator.

    Measured: z500 / g has an area mean of 5630 m and a maximum of 5904 m; the
    balanced fluid depth at lmax = 22 is MEASURED_DEPTH.
    """
    pytest.importorskip("xarray")
    hours = era5_hours("1978-12-21T00")
    fields = load_era5_fields(torch.tensor([hours]))
    assert fields.geopotential.shape == (1, 721, 1440)
    assert fields.latitude[0].item() == pytest.approx(math.pi / 2)
    height = fields.geopotential / GRAVITY
    assert 5800.0 < height.max().item() < 6000.0

    sim = SphericalShallowWater(
        lmax=22,
        nlat=32,
        nlon=64,
        T=6.0,
        dt_save=3.0,
        skip_nt=0,
        batch_size=1,
        device=DEVICE,
        log_level="warning",
        initial_condition="era5",
        mean_depth=MEAN_DEPTH,
        return_constant_fields=True,
        parameters_range={"era5_time": (hours + 1.7, hours + 1.7)},
    )
    payload = sim.forward_samples_spatiotemporal(n=1, random_seed=0)
    assert torch.isfinite(payload["data"]).all()
    # the sampled time is rounded to the 6-hourly axis, and the rounded value returned
    assert payload["constant_scalars"][0, 0].item() == hours
    orography = payload["constant_fields"][0, ..., 1]
    assert 2000.0 < orography.max() < 6000.0  # the Himalaya at lmax = 22
