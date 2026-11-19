"""Tests for the spherical shallow water simulator (torch-harmonics, Galewsky jet)."""

from __future__ import annotations

import math

import pytest
import torch

th = pytest.importorskip("torch_harmonics")
if not hasattr(th, "as_grid"):
    pytest.skip("torch-harmonics without grid descriptors", allow_module_level=True)

from torch_harmonics.examples.shallow_water_equations import (  # noqa: E402
    ShallowWaterSolver,
)

from autosim.experimental.simulations.shallow_water_sphere import (  # noqa: E402
    SphereShallowWaterDynamics,
    SphericalShallowWater,
    adams_bashforth_step,
    galewsky_initial_condition,
    shallow_water_tendencies,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RADIUS = 6.37122e6
OMEGA = 7.292e-5
GRAVITY = 9.80616


def _small_simulator(**overrides) -> SphericalShallowWater:
    options = {
        "lmax": 32,
        "grid": "legendre-gauss",
        "nlat": 48,
        "nlon": 96,
        "T": 6.0,
        "dt_save": 3.0,
        "skip_nt": 0,
        "batch_size": 2,
        "device": DEVICE,
        "log_level": "warning",
    }
    return SphericalShallowWater(**(options | overrides))


def _paper_state(dynamics, height_perturbation=120.0, noise_amplitude=1e-3):
    return galewsky_initial_condition(
        dynamics,
        jet_speed=torch.tensor([80.0]),
        height_perturbation=torch.tensor([height_perturbation]),
        perturbation_lon=torch.tensor([0.0]),
        perturbation_lat=torch.tensor([math.pi / 4]),
        noise_amplitude=noise_amplitude,
        noise_lmax=120,
        device=DEVICE,
    )


def test_output_shape_parameters_and_finiteness():
    sim = _small_simulator()
    payload = sim.forward_samples_spatiotemporal(n=3, random_seed=0)
    assert payload["data"].shape == (3, 3, 48, 96, 3)
    assert payload["constant_scalars"].shape == (3, 4)
    assert payload["constant_fields"] is None
    assert torch.isfinite(payload["data"]).all()


def test_skip_nt_drops_leading_snapshots():
    sim = _small_simulator(skip_nt=1)
    payload = sim.forward_samples_spatiotemporal(n=1, random_seed=0)
    assert payload["data"].shape[1] == 2


def test_seed_reproducibility():
    sim = _small_simulator()
    first = sim.forward_samples_spatiotemporal(n=2, random_seed=3)["data"]
    second = sim.forward_samples_spatiotemporal(n=2, random_seed=3)["data"]
    other = sim.forward_samples_spatiotemporal(n=2, random_seed=4)["data"]
    torch.testing.assert_close(first, second)
    assert not torch.allclose(first, other)


def test_num_workers_is_rejected():
    sim = _small_simulator()
    with pytest.raises(ValueError, match="batch_size"):
        sim.forward_batch(sim.sample_inputs(1, random_seed=0), num_workers=2)


def test_tendencies_match_torch_harmonics():
    sim = _small_simulator()
    torch.manual_seed(0)
    state = _paper_state(sim.dynamics)
    ours = shallow_water_tendencies(sim.dynamics, state)[0]
    solver = ShallowWaterSolver(sim.grid, sim.dt, lmax=32, mmax=32).to(DEVICE)
    reference = solver.dudtspec(state[0])
    for index in range(3):
        scale = max(reference[index].abs().max().item(), 1e-30)
        difference = (ours[index] - reference[index]).abs().max().item()
        assert difference <= 1e-10 * scale + 1e-25


def test_mass_is_conserved():
    sim = _small_simulator()
    torch.manual_seed(0)
    state = _paper_state(sim.dynamics)
    mass = state[0, 0, 0, 0].clone()
    previous = []
    for _ in range(200):
        state, previous = adams_bashforth_step(sim.dynamics, state, previous)
    torch.testing.assert_close(state[0, 0, 0, 0], mass, rtol=1e-13, atol=0.0)


def test_williamson_case_2_is_steady():
    grid = th.as_grid("legendre-gauss", nlat=48, nlon=96)
    dynamics = SphereShallowWaterDynamics(
        grid, 32, RADIUS, OMEGA, GRAVITY, 1.0e4, 300.0, DEVICE
    )
    coords = grid.coords.to(device=DEVICE, dtype=torch.float64)
    latitude = (torch.pi / 2 - coords[:, 0]).reshape(1, *grid.shape)
    u0 = 2 * math.pi * RADIUS / (12 * 86400)
    geopotential_exact = (
        2.94e4 - (RADIUS * OMEGA * u0 + 0.5 * u0**2) * torch.sin(latitude) ** 2
    )
    wind = torch.stack([u0 * torch.cos(latitude), torch.zeros_like(latitude)], dim=1)
    vorticity, divergence = dynamics.vorticity_divergence(wind)
    state = torch.stack(
        [dynamics.to_coefficients(geopotential_exact), vorticity, divergence], dim=1
    )
    state = torch.tril(state)
    previous = []
    for _ in range(288):  # 1 day
        state, previous = adams_bashforth_step(dynamics, state, previous)
    geopotential = dynamics.to_grid(state[:, 0])
    error = (
        geopotential - geopotential_exact
    ).abs().max() / geopotential_exact.abs().max()
    assert error < 1e-10


def test_balanced_jet_is_steady_without_hyperdiffusion():
    sim = _small_simulator()
    dynamics = sim.dynamics
    dynamics.step_damping = torch.ones_like(dynamics.step_damping)
    jet = _paper_state(dynamics, height_perturbation=0.0, noise_amplitude=0.0)
    state = jet.clone()
    previous = []
    for _ in range(100):
        state, previous = adams_bashforth_step(dynamics, state, previous)
    for index in range(2):
        change = (state[0, index] - jet[0, index]).abs().max()
        assert change <= 1e-10 * jet[0, index].abs().max()
