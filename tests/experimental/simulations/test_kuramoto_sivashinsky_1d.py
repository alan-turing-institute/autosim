from __future__ import annotations

import math

import pytest
import torch

from autosim.experimental.simulations import (
    KuramotoSivashinsky1D,
    estimate_ks_lyapunov,
    simulate_kuramoto_sivashinsky_1d,
)


def test_ks_sampler_is_reproducible_local_and_independent_of_nu() -> None:
    sim = KuramotoSivashinsky1D(nx=32)
    before = torch.random.get_rng_state().clone()
    fields = sim.sample_initial_conditions(8, 7)
    assert torch.equal(before, torch.random.get_rng_state())
    assert torch.equal(fields, sim.sample_initial_conditions(8, 7))
    assert not torch.equal(fields, sim.sample_initial_conditions(8, 8))
    other = KuramotoSivashinsky1D(nx=32, parameters_range={"nu": (0.8, 1.2)})
    assert torch.equal(fields, other.sample_initial_conditions(8, 7))
    assert fields.mean(dim=-2).abs().max() < 1e-14
    peaks = fields.abs().amax(dim=-2)
    assert torch.all((peaks >= 0.5) & (peaks <= 1 + 1e-14))


def test_ks_payload_records_raw_and_post_warmup_initial_fields() -> None:
    sim = KuramotoSivashinsky1D(
        nx=32,
        T=0.15,
        dt_save=0.1,
        warmup_time=0.2,
        parameters_range={"nu": (0.8, 1.2)},
    )
    payload = sim.forward_samples_spatiotemporal(2, 7, ensure_exact_n=True)
    raw = sim.sample_initial_conditions(2, 7)
    assert payload["data"].shape == (2, 3, 32, 1, 1)
    assert torch.equal(payload["seed_fields"], raw)
    assert torch.equal(
        payload["warmup_times"], torch.full((2,), 0.2, dtype=torch.float64)
    )
    assert torch.equal(payload["x"][0], sim.x)
    assert torch.allclose(
        payload["times"][0], torch.tensor([0, 0.1, 0.15], dtype=torch.float64)
    )
    for index in range(2):
        nu = float(payload["constant_scalars"][index, 0])
        warmed = sim.warmup_state(raw[index], nu=nu)
        assert torch.equal(payload["data"][index, 0, :, 0], warmed.float())
        assert torch.equal(
            payload["data"][index, :, :, 0], sim.rollout(warmed, nu=nu).float()
        )
    assert sim.forward_samples_spatiotemporal(0, 7)["data"].shape == (0, 3, 32, 1, 1)
    final = KuramotoSivashinsky1D(nx=32, T=0.1, return_timeseries=False)
    assert final.forward_samples_spatiotemporal(1, 7)["data"].shape == (1, 1, 32, 1, 1)


def test_ks_explicit_rollout_preserves_input_initial_frame_and_gradients() -> None:
    sim = KuramotoSivashinsky1D(nx=32, T=0.1, warmup_time=10)
    state = sim.sample_initial_conditions(1, 7)[0].requires_grad_()
    original = state.detach().clone()
    trajectory = sim.rollout(state)
    assert torch.equal(trajectory[0], original)
    assert torch.equal(state, original)
    trajectory[-1].square().sum().backward()
    assert state.grad is not None
    assert torch.isfinite(state.grad).all()
    varying = KuramotoSivashinsky1D(nx=32, parameters_range={"nu": (0.8, 1.2)})
    with pytest.raises(ValueError, match="explicitly"):
        varying.rollout(state)
    with pytest.raises(ValueError, match="match nx"):
        sim.rollout(torch.ones(16, 1))


def test_ks_legacy_forward_uses_fixed_reference_field() -> None:
    state = torch.ones(32, 1, dtype=torch.float64)
    sim = KuramotoSivashinsky1D(nx=32, T=0.1, initial_state=state)
    assert torch.equal(sim.sample_initial_conditions(2, 7)[0], state)
    first = sim.forward(torch.tensor([[1.0]]), allow_failures=False)
    second = sim.forward(torch.tensor([[1.0]]), allow_failures=False)
    assert isinstance(first, torch.Tensor)
    assert isinstance(second, torch.Tensor)
    assert torch.equal(first, second)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_ks_preserves_batched_constants_dtype_and_mean(dtype: torch.dtype) -> None:
    state = torch.full((2, 3, 32, 1), 0.7, dtype=dtype)
    result = simulate_kuramoto_sivashinsky_1d(state, T=0.2, dt_save=0.1)
    assert result.shape == (2, 3, 3, 32, 1)
    assert result.dtype == dtype
    assert torch.allclose(result, state[..., None, :, :], atol=1e-7, rtol=0)


@pytest.mark.parametrize("L", [5.0, 22.0])
def test_ks_matches_linear_fourier_growth(L: float) -> None:
    nx, T, amplitude = 64, 0.2, 1e-8
    x = torch.arange(nx, dtype=torch.float64) * L / nx
    k = 2 * math.pi / L
    state = amplitude * torch.cos(k * x)[:, None]
    result = simulate_kuramoto_sivashinsky_1d(state, L=L, T=T, dt_save=T)[-1]
    exact = state * math.exp((k**2 - k**4) * T)
    assert torch.allclose(result, exact, atol=1e-15, rtol=0)


def test_ks_step_is_consistent_with_full_nonlinear_pde() -> None:
    # An independent real-space RHS catches missing ETDRK4 nonlinear weights.
    nx, L, dt = 64, 22.0, 1e-5
    x = torch.arange(nx, dtype=torch.float64) * L / nx
    k = 2 * math.pi / L
    state = 0.8 * torch.cos(k * x) + 0.3 * torch.sin(2 * k * x)
    ux = -0.8 * k * torch.sin(k * x) + 0.6 * k * torch.cos(2 * k * x)
    uxx = -0.8 * k**2 * torch.cos(k * x) - 0.3 * (2 * k) ** 2 * torch.sin(2 * k * x)
    uxxxx = 0.8 * k**4 * torch.cos(k * x) + 0.3 * (2 * k) ** 4 * torch.sin(2 * k * x)
    rhs = -state * ux - uxx - uxxxx
    result = simulate_kuramoto_sivashinsky_1d(state[:, None], T=dt, dt_save=dt)[
        -1, :, 0
    ]
    assert float(((result - state) / dt - rhs).abs().max()) < 1e-5


def test_ks_fourth_order_time_convergence_and_mean_conservation() -> None:
    state = KuramotoSivashinsky1D().sample_initial_conditions(1, 7)[0]
    reference = simulate_kuramoto_sivashinsky_1d(
        state, T=2, dt_save=2, dt_max=0.003125
    )[-1]
    errors = []
    for step in (0.05, 0.025):
        result = simulate_kuramoto_sivashinsky_1d(state, T=2, dt_save=2, dt_max=step)
        assert result.mean(dim=-2).abs().max() < 1e-14
        errors.append(float((result[-1] - reference).square().mean().sqrt()))
    assert errors[1] < errors[0] / 12
    assert errors[1] < 1e-8


def test_ks_refines_in_space() -> None:
    solutions = []
    for nx in (32, 64, 128):
        x = torch.arange(nx, dtype=torch.float64) / nx
        field = torch.cos(2 * math.pi * x) + 0.5 * torch.sin(6 * math.pi * x)
        trajectory = simulate_kuramoto_sivashinsky_1d(
            field[:, None], T=5, dt_save=5, dt_max=0.025
        )
        solutions.append(trajectory[-1, :, 0])
    reference = solutions[-1]
    errors = [
        float((value - reference[:: 128 // len(value)]).square().mean().sqrt())
        for value in solutions[:2]
    ]
    assert errors[1] < errors[0] / 10
    assert errors[1] < 1e-5


def test_ks_chaotic_benchmark_has_positive_renormalized_growth_across_seeds() -> None:
    sim = KuramotoSivashinsky1D()
    states = torch.stack(
        [sim.sample_initial_conditions(1, seed)[0] for seed in (7, 11, 17)]
    )
    estimates = estimate_ks_lyapunov(states, T=200, warmup_time=100, random_seed=7)
    assert estimates.shape == (3,)
    assert torch.isfinite(estimates).all()
    assert torch.all(estimates > 0.01)


def test_ks_stable_control_has_negative_renormalized_growth() -> None:
    state = KuramotoSivashinsky1D(L=5).sample_initial_conditions(1, 7)[0]
    estimate = estimate_ks_lyapunov(state, L=5, T=20, warmup_time=0)
    assert float(estimate) < -0.5


def test_ks_screening_records_metrics_and_rejects_quiescent_states() -> None:
    sim = KuramotoSivashinsky1D(
        T=0.1,
        warmup_time=100,
        chaos_validation_time=200,
    )
    payload = sim.forward_samples_spatiotemporal(1, 7)
    assert payload["lyapunov_estimates"].shape == (1, 1)
    assert float(payload["lyapunov_estimates"][0, 0]) > sim.lyapunov_threshold
    assert float(payload["activity_estimates"][0, 0]) > sim.activity_threshold
    assert sim.forward_samples_spatiotemporal(0, 7)["lyapunov_estimates"].shape == (
        0,
        1,
    )
    equilibrium = KuramotoSivashinsky1D(
        nx=32,
        T=0.1,
        chaos_validation_time=2,
        initial_state=torch.zeros(32, 1),
    )
    with pytest.raises(RuntimeError, match="activity criterion"):
        equilibrium.forward_samples_spatiotemporal(1, 7, ensure_exact_n=True)


def test_ks_screening_raises_instead_of_silently_replacing_failed_samples() -> None:
    sim = KuramotoSivashinsky1D(
        nx=32,
        T=0.1,
        chaos_validation_time=2,
        lyapunov_threshold=100,
    )
    with pytest.raises(RuntimeError, match="No sample was silently replaced"):
        sim.forward_samples_spatiotemporal(1, 7, ensure_exact_n=True)


@pytest.mark.parametrize(
    "kwargs", [{"nu": 0}, {"L": -1}, {"T": -1}, {"dt_save": 0}, {"dt_max": 0}]
)
def test_ks_rejects_invalid_solver_settings(kwargs: dict) -> None:
    with pytest.raises(ValueError, match="must"):
        simulate_kuramoto_sivashinsky_1d(torch.ones(16, 1), **kwargs)


def test_ks_diagnostic_rejects_invalid_precision_and_times() -> None:
    with pytest.raises(TypeError, match="float64"):
        estimate_ks_lyapunov(torch.ones(16, 1))
    with pytest.raises(ValueError, match="positive"):
        estimate_ks_lyapunov(torch.ones(16, 1, dtype=torch.float64), T=0)
