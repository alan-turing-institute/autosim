from __future__ import annotations

import math

import pytest
import torch

from autosim.experimental.simulations import Burgers1D, simulate_burgers_1d


def test_grf_covariance_seed_and_local_rng() -> None:
    sim = Burgers1D(nx=64)
    before = torch.random.get_rng_state().clone()
    fields = sim.sample_initial_conditions(2000, random_seed=7)
    assert torch.equal(before, torch.random.get_rng_state())
    assert torch.equal(fields, sim.sample_initial_conditions(2000, random_seed=7))
    assert not torch.equal(fields, sim.sample_initial_conditions(2000, random_seed=8))
    assert fields[..., 0].mean(dim=-1).abs().max() < 1e-14

    # Check the declared covariance, independently of the sampler implementation.
    coefficients = torch.fft.rfft(fields[..., 0], norm="ortho") / math.sqrt(64)
    for mode in (1, 2, 3):
        expected_variance = 625 / ((2 * math.pi * mode) ** 2 + 25) ** 2
        observed = coefficients[:, mode].abs().square().mean().item()
        assert observed == pytest.approx(expected_variance, rel=0.08)


def test_payload_retains_initial_fields_and_independent_viscosities() -> None:
    sim = Burgers1D(nx=32, T=0.05, dt_save=0.02, parameters_range={"nu": (0.05, 0.2)})
    payload = sim.forward_samples_spatiotemporal(3, random_seed=9, ensure_exact_n=True)
    assert payload["data"].shape == (3, 4, 32, 1, 1)
    assert payload["constant_scalars"].shape == (3, 1)
    assert payload["constant_fields"] is None
    assert torch.equal(
        payload["data"][:, 0, :, 0], sim.sample_initial_conditions(3, 9).float()
    )
    assert torch.allclose(
        payload["times"][0], torch.tensor([0, 0.02, 0.04, 0.05], dtype=torch.float64)
    )
    assert torch.equal(payload["x"][0], sim.x)
    for index in range(3):
        field = sim.sample_initial_conditions(3, 9)[index]
        nu = float(payload["constant_scalars"][index, 0])
        expected = sim.rollout(field, nu=nu).float()
        assert torch.equal(payload["data"][index, :, :, 0], expected)


def test_legacy_forward_is_deterministic_and_zero_sample_payload() -> None:
    sim = Burgers1D(nx=16, T=0.02)
    parameters = torch.tensor([[0.1]])
    first = sim.forward(parameters, allow_failures=False)
    second = sim.forward(parameters, allow_failures=False)
    assert isinstance(first, torch.Tensor)
    assert isinstance(second, torch.Tensor)
    assert torch.equal(first, second)
    assert sim.forward_samples_spatiotemporal(0, 0)["data"].shape == (0, 3, 16, 1, 1)
    final = Burgers1D(nx=16, T=0.02, return_timeseries=False)
    assert final.forward_samples_spatiotemporal(1, 0)["data"].shape == (1, 1, 16, 1, 1)


def test_rollout_requires_viscosity_when_it_varies() -> None:
    sim = Burgers1D(nx=16, T=0, parameters_range={"nu": (0.05, 0.2)})
    with pytest.raises(ValueError, match="explicitly"):
        sim.rollout(sim.sample_initial_conditions(1, 0)[0])


@pytest.mark.parametrize("initial_condition", ["sine", "fourier"])
def test_other_initial_condition_families(initial_condition: str) -> None:
    sim = Burgers1D(nx=32, initial_condition=initial_condition)
    field = sim.sample_initial_conditions(4, 13)
    assert field.shape == (4, 32, 1)
    assert field.mean(dim=-2).abs().max() < 1e-14
    assert torch.all(field.abs().amax(dim=-2) <= 1)


def test_burgers_preserves_constant_states_and_input() -> None:
    state = torch.full((2, 32, 1), 0.7, dtype=torch.float64)
    original = state.clone()
    result = simulate_burgers_1d(state, T=0.1, dt_save=0.05)
    assert result.shape == (2, 3, 32, 1)
    assert torch.equal(state, original)
    assert torch.allclose(result, state[:, None], atol=1e-14, rtol=0)


def test_burgers_matches_cole_hopf_and_refines_in_time() -> None:
    nx, nu, amplitude, T = 64, 0.03, 0.3, 0.1
    x = torch.arange(nx, dtype=torch.float64) / nx
    k = 2 * math.pi
    initial = (
        2 * nu * amplitude * k * torch.sin(k * x) / (1 + amplitude * torch.cos(k * x))
    )[:, None]
    decayed = amplitude * math.exp(-nu * k**2 * T)
    exact = (
        2 * nu * decayed * k * torch.sin(k * x) / (1 + decayed * torch.cos(k * x))
    )[:, None]
    errors = []
    for step in (0.004, 0.002):
        trajectory = simulate_burgers_1d(initial, nu=nu, T=T, dt_save=T, dt_max=step)
        errors.append(float((trajectory[-1] - exact).abs().max()))
        assert trajectory.mean(dim=-2).abs().max() < 1e-14
        assert trajectory[-1].square().mean() < trajectory[0].square().mean()
    assert errors[1] < errors[0] / 3
    assert errors[1] < 1e-5


def test_burgers_accepts_explicit_reference_fields_and_state_gradients() -> None:
    state = torch.sin(2 * math.pi * torch.arange(16, dtype=torch.float64) / 16)[:, None]
    sim = Burgers1D(nx=16, T=0.02, initial_state=state)
    assert torch.equal(sim.sample_initial_conditions(2, 1)[0], state)
    state.requires_grad_()
    sim.rollout(state)[-1].square().sum().backward()
    assert state.grad is not None
    assert torch.isfinite(state.grad).all()


@pytest.mark.parametrize(
    "kwargs", [{"nu": 0}, {"L": -1}, {"T": -1}, {"dt_save": 0}, {"cfl": 0.6}]
)
def test_burgers_rejects_invalid_solver_settings(kwargs: dict) -> None:
    with pytest.raises(ValueError, match="must"):
        simulate_burgers_1d(torch.ones(16, 1), **kwargs)
