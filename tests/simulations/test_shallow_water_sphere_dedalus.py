import builtins
import math

import numpy as np
import pytest
import torch

import autosim.experimental.simulations.shallow_water_sphere_dedalus as sws
from autosim.experimental.simulations import ShallowWaterSphereDedalus


def test_spherical_swe_timeseries_shape_with_mocked_solver(monkeypatch) -> None:
    def fake_solver(**kwargs) -> torch.Tensor:
        frames = int(kwargs["T"] / kwargs["dt_save"]) + 1 - kwargs["skip_nt"]
        return torch.arange(
            frames * kwargs["nlon"] * kwargs["nlat"] * 3,
            dtype=torch.float32,
        ).reshape(frames, kwargs["nlon"], kwargs["nlat"], 3)

    monkeypatch.setattr(sws, "simulate_swe_sphere_dedalus", fake_solver)
    sim = ShallowWaterSphereDedalus(
        return_timeseries=True,
        log_level="warning",
        nlon=8,
        nlat=4,
        T=3.0,
        dt_save=1.0,
        skip_nt=1,
        parameters_range={"height_perturbation_m": (120.0, 120.0)},
    )

    out = sim.forward_samples_spatiotemporal(n=1, random_seed=0)

    assert out["data"].shape == (1, 3, 8, 4, 3)
    assert out["constant_scalars"].shape == (1, 1)
    assert out["constant_fields"] is None


def test_spherical_swe_final_frame_shape_with_mocked_solver(monkeypatch) -> None:
    def fake_solver(**kwargs) -> torch.Tensor:
        return torch.zeros(kwargs["nlon"], kwargs["nlat"], 3).unsqueeze(0)

    monkeypatch.setattr(sws, "simulate_swe_sphere_dedalus", fake_solver)
    sim = ShallowWaterSphereDedalus(
        return_timeseries=False,
        log_level="warning",
        nlon=8,
        nlat=4,
        parameters_range={"height_perturbation_m": (120.0, 120.0)},
    )

    out = sim.forward_samples_spatiotemporal(n=1, random_seed=0)

    assert out["data"].shape == (1, 1, 8, 4, 3)


def test_spherical_swe_skip_nt_too_large_raises_before_importing_dedalus() -> None:
    with pytest.raises(ValueError, match="skip_nt is too large"):
        sws.simulate_swe_sphere_dedalus(
            height_perturbation_m=120.0,
            return_timeseries=True,
            nlon=8,
            nlat=4,
            T=1.0,
            dt_save=1.0,
            timestep_seconds=600.0,
            dealias=1.5,
            radius_m=sws.EARTH_RADIUS_M,
            omega_per_s=sws.EARTH_OMEGA_PER_S,
            gravity_mps2=sws.EARTH_GRAVITY_MPS2,
            mean_height_m=sws.DEFAULT_MEAN_HEIGHT_M,
            umax_mps=80.0,
            perturbation_lon=0.0,
            perturbation_lat=0.25,
            perturbation_lon_width=1.0 / 3.0,
            perturbation_lat_width=1.0 / 15.0,
            hyperviscosity_m2ps=sws.DEFAULT_HYPERVISCOSITY_M2PS,
            hyperviscosity_match_mode=sws.DEFAULT_HYPERVISCOSITY_MATCH_MODE,
            skip_nt=2,
        )


def test_import_dedalus_error_mentions_optional_extra(monkeypatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "dedalus.public":
            msg = "dedalus missing"
            raise ImportError(msg)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="OpenMP-capable compiler"):
        sws._import_dedalus()


def test_invalid_ic_mode_raises() -> None:
    with pytest.raises(ValueError, match="ic_mode must be one of"):
        ShallowWaterSphereDedalus(ic_mode="not_a_mode", log_level="warning")
    with pytest.raises(ValueError, match="ic_mode must be one of"):
        sws.simulate_swe_sphere_dedalus(
            height_perturbation_m=120.0,
            return_timeseries=False,
            nlon=8,
            nlat=4,
            T=1.0,
            dt_save=1.0,
            timestep_seconds=600.0,
            dealias=1.5,
            radius_m=sws.EARTH_RADIUS_M,
            omega_per_s=sws.EARTH_OMEGA_PER_S,
            gravity_mps2=sws.EARTH_GRAVITY_MPS2,
            mean_height_m=sws.DEFAULT_MEAN_HEIGHT_M,
            umax_mps=80.0,
            perturbation_lon=0.0,
            perturbation_lat=0.25,
            perturbation_lon_width=1.0 / 3.0,
            perturbation_lat_width=1.0 / 15.0,
            hyperviscosity_m2ps=sws.DEFAULT_HYPERVISCOSITY_M2PS,
            hyperviscosity_match_mode=sws.DEFAULT_HYPERVISCOSITY_MATCH_MODE,
            ic_mode="not_a_mode",
        )


def test_random_jet_and_forcing_kwargs_passed_through(monkeypatch) -> None:
    captured: dict = {}

    def fake_solver(**kwargs) -> torch.Tensor:
        captured.update(kwargs)
        return torch.zeros(kwargs["nlon"], kwargs["nlat"], 3).unsqueeze(0)

    monkeypatch.setattr(sws, "simulate_swe_sphere_dedalus", fake_solver)
    sim = ShallowWaterSphereDedalus(
        return_timeseries=False,
        log_level="warning",
        nlon=8,
        nlat=4,
        ic_mode="random_jet",
        n_jet_modes=3,
        perturbation_zonal_mode=6,
        forcing=True,
        forcing_rate_m_per_hour=3.5,
        forcing_day_hours=12.0,
        forcing_year_hours=500.0,
        forcing_declination_rad=0.2,
        parameters_range={"height_perturbation_m": (120.0, 120.0)},
    )

    sim.forward_samples_spatiotemporal(n=1, random_seed=0)

    assert captured["ic_mode"] == "random_jet"
    assert captured["n_jet_modes"] == 3
    assert captured["perturbation_zonal_mode"] == 6
    assert captured["forcing"] is True
    assert captured["forcing_rate_m_per_hour"] == 3.5
    assert captured["forcing_day_hours"] == 12.0
    assert captured["forcing_year_hours"] == 500.0
    assert captured["forcing_declination_rad"] == 0.2


def test_galewsky_jet_helper_is_zonally_confined() -> None:
    lat = np.linspace(-math.pi / 2, math.pi / 2, 200).reshape(1, -1)
    u_phi = sws._galewsky_jet(80.0, lat)
    lat0 = math.pi / 7.0
    lat1 = math.pi / 2.0 - lat0
    outside = (lat < lat0) | (lat > lat1)
    assert np.allclose(u_phi[outside], 0.0)
    assert (u_phi >= -1e-12).all()
    assert u_phi.max() > 0.0


def test_random_zonal_jet_helper_scaling_and_poles() -> None:
    np.random.seed(0)
    lat = np.linspace(-math.pi / 2, math.pi / 2, 50)
    lat_grid = np.broadcast_to(lat, (12, 50)).copy()
    u_phi = sws._random_zonal_jet(40.0, lat_grid, n_jet_modes=4)
    assert u_phi.shape == (12, 50)
    assert np.isclose(np.max(np.abs(u_phi)), 40.0)
    # Tapered toward the poles.
    assert abs(u_phi[:, 0]).max() < abs(u_phi[:, 25]).max()
    # Per-longitude variation (not a single zonal profile).
    assert u_phi.std(axis=0).max() > 0.0
    # Low-order longitude modes avoid grid-scale independent-column noise.
    neighbor_delta = np.abs(np.diff(u_phi, axis=0)).mean()
    shuffled_delta = np.abs(np.diff(np.random.permutation(u_phi), axis=0)).mean()
    assert neighbor_delta < shuffled_delta


def test_save_times_include_terminal_time_for_noninteger_ratio() -> None:
    assert sws._save_times_hours(T=2.5, dt_save=1.0) == [0.0, 1.0, 2.0, 2.5]
    assert sws._save_times_hours(T=2.0, dt_save=1.0) == [0.0, 1.0, 2.0]


@pytest.mark.parametrize(
    ("ic_mode", "forcing", "perturbation_zonal_mode"),
    [("galewsky", False, 0), ("random_jet", False, 6), ("galewsky", True, 0)],
)
def test_real_dedalus_smoke_run(ic_mode, forcing, perturbation_zonal_mode) -> None:
    pytest.importorskip("dedalus")
    np.random.seed(0)
    y = sws.simulate_swe_sphere_dedalus(
        height_perturbation_m=120.0,
        return_timeseries=True,
        nlon=16,
        nlat=8,
        T=2.0,
        dt_save=1.0,
        timestep_seconds=900.0,
        dealias=1.5,
        radius_m=sws.EARTH_RADIUS_M,
        omega_per_s=sws.EARTH_OMEGA_PER_S,
        gravity_mps2=sws.EARTH_GRAVITY_MPS2,
        mean_height_m=sws.DEFAULT_MEAN_HEIGHT_M,
        umax_mps=80.0,
        perturbation_lon=0.0,
        perturbation_lat=math.pi / 4.0,
        perturbation_lon_width=1.0 / 3.0,
        perturbation_lat_width=1.0 / 15.0,
        perturbation_zonal_mode=perturbation_zonal_mode,
        ic_mode=ic_mode,
        n_jet_modes=4,
        hyperviscosity_m2ps=sws.DEFAULT_HYPERVISCOSITY_M2PS,
        hyperviscosity_match_mode=4,
        forcing=forcing,
        forcing_rate_m_per_hour=2.0,
    )
    assert y.shape == (3, 16, 8, 3)
    assert torch.isfinite(y).all()
