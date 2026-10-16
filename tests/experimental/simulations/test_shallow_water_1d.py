from __future__ import annotations

import math

import pytest
import torch

from autosim.experimental.simulations import ShallowWater1D, simulate_shallow_water_1d


def test_swe_sampler_reproducibility_positivity_and_initial_velocity() -> None:
    sim = ShallowWater1D(nx=32, T=0.05, dt_save=0.02)
    fields = sim.sample_initial_conditions(3, 8)
    assert torch.equal(fields, sim.sample_initial_conditions(3, 8))
    assert not torch.equal(fields, sim.sample_initial_conditions(3, 9))
    assert torch.all(fields[..., 0] > 1)
    assert torch.equal(fields[..., 1], torch.zeros(3, 32, dtype=torch.float64))
    payload = sim.forward_samples_spatiotemporal(3, 8)
    assert sim.output_names == ["h", "u"]
    assert payload["data"].shape == (3, 4, 32, 1, 2)
    assert torch.equal(payload["data"][:, 0, :, 0], fields.float())
    assert torch.equal(
        payload["x"][0], (torch.arange(32, dtype=torch.float64) + 0.5) / 32
    )
    assert payload["constant_scalars"].shape == (3, 1)
    assert sim.forward_samples_spatiotemporal(0, 0)["data"].shape == (0, 4, 32, 1, 2)


def test_gaussian_bump_is_periodic_across_domain_seam() -> None:
    sim = ShallowWater1D(
        nx=32,
        initial_condition_kwargs={
            "amplitude_range": (0.1, 0.1),
            "position_fraction_range": (0.0, 0.0),
            "width_fraction_range": (0.1, 0.1),
        },
    )
    h = sim.sample_initial_conditions(1, 0)[0, :, 0]
    assert h[0] == pytest.approx(float(h[-1]), abs=1e-14)
    assert h[0] > h[16]


def test_payload_retains_initial_fields_and_independent_gravities() -> None:
    sim = ShallowWater1D(nx=32, T=0.05, parameters_range={"g": (0.5, 2.0)})
    payload = sim.forward_samples_spatiotemporal(3, random_seed=9)
    states = sim.sample_initial_conditions(3, random_seed=9)
    assert torch.equal(payload["data"][:, 0, :, 0], states.float())
    gravities = payload["constant_scalars"][:, 0]
    assert torch.all((gravities >= 0.5) & (gravities <= 2.0))
    assert len(gravities.unique()) == 3
    for index in range(3):
        expected = sim.rollout(states[index], g=float(gravities[index])).float()
        assert torch.equal(payload["data"][index, :, :, 0], expected)
    fixed_gravity = ShallowWater1D(nx=32, T=0.05, g=1.0)
    assert torch.equal(
        states, fixed_gravity.sample_initial_conditions(3, random_seed=9)
    )
    with pytest.raises(ValueError, match="explicitly"):
        sim.rollout(states[0])


def test_swe_preserves_uniform_moving_water_and_input() -> None:
    state = torch.ones((2, 32, 2), dtype=torch.float64)
    state[..., 1] = 0.2
    original = state.clone()
    result = simulate_shallow_water_1d(state, T=0.1, dt_save=0.05)
    assert result.shape == (2, 3, 32, 2)
    assert torch.equal(state, original)
    assert torch.allclose(result, state[:, None], atol=1e-14, rtol=0)


def test_swe_conserves_water_and_momentum() -> None:
    sim = ShallowWater1D(nx=64, T=0.2, dt_save=0.02)
    state = sim.sample_initial_conditions(1, 17)[0]
    state[:, 1] = 0.1 * torch.sin(2 * math.pi * sim.x)
    result = sim.rollout(state)
    water = result[..., 0].sum(dim=-1)
    momentum = (result[..., 0] * result[..., 1]).sum(dim=-1)
    assert torch.allclose(water, water[0].expand_as(water), atol=1e-12, rtol=0)
    assert torch.allclose(momentum, momentum[0].expand_as(momentum), atol=1e-12, rtol=0)
    assert torch.all(result[..., 0] > 0)


def test_swe_converges_to_linear_gravity_wave() -> None:
    errors = []
    amplitude, T = 1e-5, 0.1
    for nx in (32, 64):
        x = (torch.arange(nx, dtype=torch.float64) + 0.5) / nx
        state = torch.stack(
            (1 + amplitude * torch.cos(2 * math.pi * x), torch.zeros_like(x)), dim=-1
        )
        result = simulate_shallow_water_1d(state, T=T, dt_save=T, dt_max=0.001)[-1]
        exact = torch.stack(
            (
                1 + amplitude * torch.cos(2 * math.pi * x) * math.cos(2 * math.pi * T),
                amplitude * torch.sin(2 * math.pi * x) * math.sin(2 * math.pi * T),
            ),
            dim=-1,
        )
        errors.append(float((result - exact).square().mean().sqrt()) / amplitude)
    assert errors[1] < errors[0] / 1.5
    assert errors[1] < 0.02


def test_gravity_rescales_zero_velocity_dynamics() -> None:
    sim = ShallowWater1D(
        nx=64, initial_condition_kwargs={"amplitude_range": (0.4, 0.4)}
    )
    state = sim.sample_initial_conditions(1, 17)[0]
    base = simulate_shallow_water_1d(state, g=1.0, T=0.2, dt_save=0.05, dt_max=0.001)
    faster = simulate_shallow_water_1d(
        state, g=4.0, T=0.1, dt_save=0.025, dt_max=0.0005
    )
    assert torch.allclose(faster[..., 0], base[..., 0], atol=1e-13, rtol=0)
    assert torch.allclose(faster[..., 1], 2 * base[..., 1], atol=1e-13, rtol=0)


def test_swe_legacy_forward_and_reference_field() -> None:
    sim = ShallowWater1D(nx=16, T=0.02)
    parameters = torch.tensor([[1.0]])
    first = sim.forward(parameters, allow_failures=False)
    second = sim.forward(parameters, allow_failures=False)
    assert isinstance(first, torch.Tensor)
    assert isinstance(second, torch.Tensor)
    assert torch.equal(first, second)
    state = sim.sample_initial_conditions(1, 3)[0]
    fixed = ShallowWater1D(nx=16, T=0, initial_state=state)
    assert torch.equal(fixed.sample_initial_conditions(2, 5)[0], state)
    assert torch.equal(fixed.rollout(state)[0], state)


def test_swe_rejects_dry_states() -> None:
    state = torch.ones((16, 2))
    state[0, 0] = 0
    with pytest.raises(ValueError, match="positive"):
        simulate_shallow_water_1d(state)
    with pytest.raises(ValueError, match="positive"):
        ShallowWater1D(nx=16, initial_state=state)
