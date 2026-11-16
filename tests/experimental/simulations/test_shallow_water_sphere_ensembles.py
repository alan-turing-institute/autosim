"""Tests of restarts, reference ensembles and coarse-graining on the sphere."""

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
    shallow_water_tendencies,
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


def _saved_state(**options) -> torch.Tensor:
    """Last frame of one short run, shape (32, 64, 3)."""
    payload = _small_simulator(**options).forward_samples_spatiotemporal(
        n=1, random_seed=0
    )
    return payload["data"][0, -1].clone()


def _restart(initial_state, **options) -> SphericalShallowWater:
    restart_options = {
        "initial_condition": "restart",
        "initial_state": initial_state,
        "parameters_range": None,
    }
    return _small_simulator(**(restart_options | options))


# ---------------------------------------------------------------------------
# Restart: branching N members from one saved state
# ---------------------------------------------------------------------------


def test_restart_starts_from_the_supplied_state():
    initial_state = _saved_state()
    sim = _restart(initial_state)
    payload = sim.forward_samples_spatiotemporal(n=2, random_seed=0)
    assert payload["constant_scalars"].shape == (2, 0)
    for member in range(2):
        torch.testing.assert_close(
            payload["data"][member, 0], initial_state, rtol=1e-5, atol=1e-4
        )
    # the supplied state can carry a leading dimension of one
    sim = _restart(initial_state.unsqueeze(0))
    again = sim.forward_samples_spatiotemporal(n=2, random_seed=0)
    torch.testing.assert_close(again["data"], payload["data"])


def test_branches_without_forcing_are_identical():
    payload = _restart(_saved_state()).forward_samples_spatiotemporal(
        n=4, random_seed=0
    )
    for member in range(1, 4):
        torch.testing.assert_close(
            payload["data"][member], payload["data"][0], rtol=0.0, atol=0.0
        )


def test_spread_of_forced_branches_starts_at_zero_and_grows():
    sim = _restart(_saved_state(), T=12.0, **FORCING)
    data = sim.forward_samples_spatiotemporal(n=4, random_seed=0)["data"]
    spread = data.std(dim=0).mean(dim=(1, 2, 3))  # per frame
    assert spread[0] == 0.0
    assert (spread[1:].diff() > 0).all()
    assert spread[1] > 0


def test_restart_requires_a_valid_state():
    with pytest.raises(ValueError, match="initial_state is required"):
        _small_simulator(initial_condition="restart", parameters_range=None)
    with pytest.raises(ValueError, match="initial_state is only valid"):
        _small_simulator(initial_state=_saved_state())
    with pytest.raises(ValueError, match="initial_state must have shape"):
        _restart(torch.zeros(16, 32, 3))
    with pytest.raises(ValueError, match="initial_state must be finite"):
        _restart(torch.full((32, 64, 3), float("nan")))


def test_restart_with_orography_needs_the_constant_fields():
    """The saved h is the free surface: the fluid depth needs the orography."""
    options = {
        "initial_condition": "mountain",
        "parameters_range": None,
        "mean_depth": 5960.0,
        "return_constant_fields": True,
    }
    payload = _small_simulator(**options).forward_samples_spatiotemporal(
        n=1, random_seed=0
    )
    sim = _restart(
        payload["data"][0, -1],
        initial_constant_fields=payload["constant_fields"][0],
        mean_depth=5960.0,
        return_constant_fields=True,
    )
    restarted = sim.forward_samples_spatiotemporal(n=1, random_seed=0)
    torch.testing.assert_close(
        restarted["constant_fields"], payload["constant_fields"], rtol=1e-4, atol=1e-2
    )
    torch.testing.assert_close(
        restarted["data"][0, 0], payload["data"][0, -1], rtol=1e-5, atol=1e-3
    )
    # the continuation equals the original run continued for the same time
    longer = _small_simulator(T=12.0, **options).forward_samples_spatiotemporal(
        n=1, random_seed=0
    )
    torch.testing.assert_close(
        restarted["data"][0, -1], longer["data"][0, -1], rtol=1e-3, atol=0.5
    )


# ---------------------------------------------------------------------------
# Latent forcing state: redraw or carry
# ---------------------------------------------------------------------------


def test_carried_forcing_state_continues_the_pattern():
    first = _small_simulator(return_forcing_state=True, **FORCING)
    payload = first.forward_samples_spatiotemporal(n=1, random_seed=0)
    state = payload["data"][0, -1]
    pattern = payload["forcing_state"][0, -1]

    carried = _restart(
        state,
        branch_forcing_state="carry",
        initial_forcing_state=pattern,
        return_forcing_state=True,
        **FORCING,
    ).forward_samples_spatiotemporal(n=3, random_seed=1)
    for member in range(3):
        torch.testing.assert_close(
            carried["forcing_state"][member, 0], pattern, rtol=1e-4, atol=1e-9
        )
    # the members share the pattern at the branch point and then diverge
    later = carried["forcing_state"][:, -1]
    assert (later[0] - later[1]).abs().max() > 0

    redrawn = _restart(
        state, return_forcing_state=True, **FORCING
    ).forward_samples_spatiotemporal(n=3, random_seed=1)
    difference = (redrawn["forcing_state"][0, 0] - pattern).abs().max()
    assert difference > 0.1 * pattern.abs().max()
    assert (
        redrawn["forcing_state"][0, 0] - redrawn["forcing_state"][1, 0]
    ).abs().max() > 0


def test_carry_requires_the_forcing_state():
    with pytest.raises(ValueError, match="initial_forcing_state"):
        _restart(_saved_state(), branch_forcing_state="carry", **FORCING)
    with pytest.raises(ValueError, match="branch_forcing_state must be one of"):
        _small_simulator(branch_forcing_state="unknown")


# ---------------------------------------------------------------------------
# Balance step of from_fields
# ---------------------------------------------------------------------------


def test_from_fields_removes_the_divergence_and_balances_the_height():
    initial_state = _saved_state(**FORCING, forcing_divergent_fraction=0.5)
    sim = _restart(initial_state, initial_condition="from_fields", T=0.0)
    payload = sim.forward_samples_spatiotemporal(n=1, random_seed=0)
    state = fields_to_spectral(sim.dynamics, payload["data"][:, 0])
    vorticity_scale = state[:, 1].abs().max()
    assert state[:, 2].abs().max() < 1e-5 * vorticity_scale  # float32 frames
    # the wind of the supplied fields is kept, without its divergent part
    supplied = fields_to_spectral(sim.dynamics, initial_state.unsqueeze(0))
    assert (state[:, 1] - supplied[:, 1]).abs().max() < 1e-4 * vorticity_scale
    # balance: the divergence tendency vanishes
    balanced = supplied.clone()
    balanced[:, 0] = state[:, 0]
    balanced[:, 2] = 0.0
    tendency = shallow_water_tendencies(sim.dynamics, balanced)
    unbalanced = supplied.clone()
    unbalanced[:, 2] = 0.0
    unbalanced_tendency = shallow_water_tendencies(sim.dynamics, unbalanced)
    assert tendency[:, 2].abs().max() < 1e-3 * unbalanced_tendency[:, 2].abs().max()


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


def test_carried_reference_members_inherit_the_pattern():
    options = {
        "skip_nt": 1,
        "T": 9.0,
        "reference_members": 3,
        "return_forcing_state": True,
    } | FORCING
    carried = _small_simulator(
        branch_forcing_state="carry", **options
    ).forward_samples_spatiotemporal(n=1, random_seed=0)
    redrawn = _small_simulator(**options).forward_samples_spatiotemporal(
        n=1, random_seed=0
    )
    # the spread of the first interval is smaller when the members share the pattern
    spread_carried = carried["reference_ensemble"][:, :, 1].std(dim=1).mean()
    spread_redrawn = redrawn["reference_ensemble"][:, :, 1].std(dim=1).mean()
    assert 0 < spread_carried < spread_redrawn


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
