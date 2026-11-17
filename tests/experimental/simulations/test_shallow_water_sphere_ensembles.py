"""Tests of reference ensembles and coarse-graining on the sphere."""

from __future__ import annotations

import math

import pytest
import torch

th = pytest.importorskip("torch_harmonics")
if not hasattr(th, "as_grid"):
    pytest.skip("torch-harmonics without grid descriptors", allow_module_level=True)

from autosim.experimental.simulations.shallow_water_sphere import (  # noqa: E402
    SphericalShallowWater,
    fields_to_spectral,
    spectral_to_fields,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LMAX = 22
# The published Galewsky jet: equal bounds fix every parameter
FIXED_GALEWSKY = {
    "jet_speed": (80.0, 80.0),
    "height_perturbation_m": (120.0, 120.0),
    "perturbation_lon": (0.0, 0.0),
    "perturbation_lat": (math.pi / 4, math.pi / 4),
}
FORCING = {
    "forcing_type": "vortical",
    "forcing_energy_rate": 1e-4,
    "forcing_correlation_time": 6.0,
}


def _small_simulator(**overrides) -> SphericalShallowWater:
    options = {
        "lmax": LMAX,
        "grid": "legendre-gauss",
        "nlat": 32,
        "nlon": 64,
        "T": 6.0,
        "dt_save": 3.0,
        "skip_nt": 0,
        "batch_size": 4,
        "device": DEVICE,
        "log_level": "warning",
        "initial_condition": "galewsky",
        "parameters_range": FIXED_GALEWSKY,
    }
    return SphericalShallowWater(**(options | overrides))


# ---------------------------------------------------------------------------
# Reference ensembles of a generated run
# ---------------------------------------------------------------------------


def test_reference_members_agree_with_truth_without_forcing():
    sim = _small_simulator(reference_members=3, skip_nt=1, T=9.0)
    payload = sim.forward_samples_spatiotemporal(n=2, random_seed=0)
    ensemble = payload["reference_ensemble"]
    assert payload["data"].shape == (2, 3, 32, 64, 3)
    assert ensemble.shape == (2, 3, 3, 32, 64, 3)
    for member in range(3):
        torch.testing.assert_close(
            ensemble[:, member], payload["data"], rtol=1e-6, atol=1e-6
        )


def test_reference_members_branch_from_the_first_saved_frame():
    sim = _small_simulator(reference_members=4, skip_nt=1, T=12.0, **FORCING)
    payload = sim.forward_samples_spatiotemporal(n=2, random_seed=0)
    truth = payload["data"]
    ensemble = payload["reference_ensemble"]
    assert ensemble.shape == (2, 4, 4, 32, 64, 3)
    for member in range(4):
        torch.testing.assert_close(
            ensemble[:, member, 0], truth[:, 0], rtol=0.0, atol=0.0
        )
    spread = ensemble.std(dim=1).mean(dim=(0, 2, 3, 4))  # per frame
    assert spread[0] == 0.0
    assert (spread[1:].diff() > 0).all()
    # the truth is one more member: its distance to the ensemble mean is comparable
    # to the spread of the members
    error = (truth - ensemble.mean(dim=1)).abs().mean(dim=(0, 2, 3, 4))
    assert 0.2 * spread[-1] < error[-1] < 5 * spread[-1]


def test_truth_is_the_same_with_and_without_reference_members():
    options = {"skip_nt": 1, "T": 9.0} | FORCING
    alone = _small_simulator(**options).forward_samples_spatiotemporal(
        n=2, random_seed=3
    )
    with_members = _small_simulator(
        reference_members=2, **options
    ).forward_samples_spatiotemporal(n=2, random_seed=3)
    torch.testing.assert_close(alone["data"], with_members["data"], rtol=0.0, atol=0.0)
    assert "reference_ensemble" not in alone


# ---------------------------------------------------------------------------
# Coarse-grained truth and physics baseline
# ---------------------------------------------------------------------------

COARSE = {"output_lmax": 11, "output_nlat": 16, "output_nlon": 32}


def test_coarse_grained_data_is_the_truncation_of_the_run():
    fine = _small_simulator(T=12.0)
    coarse = _small_simulator(T=12.0, **COARSE)
    fine_data = fine.forward_samples_spatiotemporal(n=2, random_seed=0)["data"]
    coarse_data = coarse.forward_samples_spatiotemporal(n=2, random_seed=0)["data"]
    assert coarse_data.shape == (2, 5, 16, 32, 3)
    for frame in (0, 4):
        state = fields_to_spectral(fine.dynamics, fine_data[:, frame])
        truncated = spectral_to_fields(coarse.output_dynamics, state[:, :, :11, :11])
        torch.testing.assert_close(
            coarse_data[:, frame], truncated.float().cpu(), rtol=1e-4, atol=1e-3
        )


def test_baseline_starts_from_the_truncated_state():
    sim = _small_simulator(
        T=48.0, skip_nt=2, return_baseline=True, return_constant_fields=True, **COARSE
    )
    payload = sim.forward_samples_spatiotemporal(n=2, random_seed=0)
    data = payload["data"]
    baseline = payload["baseline"]
    assert baseline.shape == data.shape == (2, 15, 16, 32, 3)
    assert payload["constant_fields"].shape == (2, 16, 32, 2)
    torch.testing.assert_close(baseline[:, 0], data[:, 0], rtol=1e-5, atol=1e-4)
    # the low-resolution solver departs from the coarse-grained truth with time. The
    # Galewsky jet is sharp at lmax = 11, so the baseline departs within the first 3 h.
    # From 3 h to 42 h after the branch the error then grows by a factor 1.73
    # (measured: 1.38 by 18 h, 1.98 by 66 h).
    error = (baseline - data).abs().mean(dim=(0, 2, 3, 4))
    assert error[0] < 1e-3
    assert error[-1] > 1.5 * error[1] > 0
    assert torch.isfinite(baseline).all()


def test_coarse_graining_settings_are_validated():
    with pytest.raises(ValueError, match="output_lmax"):
        _small_simulator(output_lmax=LMAX + 1, output_nlat=64, output_nlon=128)
    with pytest.raises(ValueError, match="return_baseline requires output_lmax"):
        _small_simulator(return_baseline=True)
