import pytest
import torch

from autosim.experimental.simulations.navier_stokes_conditioned import (
    ConditionedNavierStokes2D,
)


def test_default_parameter_ranges_include_sampled_smoke_controls() -> None:
    sim = ConditionedNavierStokes2D()

    assert sim.parameters_range["buoyancy_y"] == (0.2, 0.5)
    assert sim.parameters_range["smoothness"] == (4.0, 8.0)
    assert sim.parameters_range["noise_scale"] == (8.0, 18.0)
    assert sim.parameters_range["smoke_diffusivity"] == (0.0, 1e-3)


def test_forward_uses_fixed_defaults_when_ranges_not_provided(monkeypatch) -> None:
    captured: dict[str, float | int | torch.Tensor] = {}

    def _fake_simulate_conditioned_navier_stokes_2d(*, params, **kwargs):
        captured["params"] = params.clone()
        captured["smoothness"] = kwargs["smoothness"]
        captured["noise_scale"] = kwargs["noise_scale"]
        captured["smoke_diffusivity"] = kwargs["smoke_diffusivity"]
        return torch.zeros((8, 8, 3), dtype=torch.float32)

    monkeypatch.setattr(
        "autosim.experimental.simulations.navier_stokes_conditioned."
        "simulate_conditioned_navier_stokes_2d",
        _fake_simulate_conditioned_navier_stokes_2d,
    )

    sim = ConditionedNavierStokes2D(
        parameters_range={"buoyancy_y": (0.35, 0.35)},
        n=8,
        T=0.1,
        dt=0.01,
    )

    x = torch.tensor([[0.35]], dtype=torch.float32)
    sim.forward(x, allow_failures=False)

    assert torch.allclose(captured["params"], torch.tensor([0.35], dtype=torch.float32))  # type: ignore  # noqa: PGH003
    assert captured["smoothness"] == 6.0
    assert captured["noise_scale"] == 11.0
    assert captured["smoke_diffusivity"] == 5e-4


def test_forward_uses_sampled_smoke_controls_from_input(monkeypatch) -> None:
    captured: dict[str, float | int | torch.Tensor] = {}

    def _fake_simulate_conditioned_navier_stokes_2d(*, params, **kwargs):
        captured["params"] = params.clone()
        captured["smoothness"] = kwargs["smoothness"]
        captured["noise_scale"] = kwargs["noise_scale"]
        captured["smoke_diffusivity"] = kwargs["smoke_diffusivity"]
        return torch.zeros((8, 8, 3), dtype=torch.float32)

    monkeypatch.setattr(
        "autosim.experimental.simulations.navier_stokes_conditioned."
        "simulate_conditioned_navier_stokes_2d",
        _fake_simulate_conditioned_navier_stokes_2d,
    )

    sim = ConditionedNavierStokes2D(
        parameters_range={
            "buoyancy_y": (0.2, 0.5),
            "smoothness": (4.0, 8.0),
            "noise_scale": (8.0, 18.0),
            "smoke_diffusivity": (0.0, 1e-3),
        },
        n=8,
        T=0.1,
        dt=0.01,
    )

    x = torch.tensor([[0.41, 6.5, 12.5, 7e-4]], dtype=torch.float32)
    sim.forward(x, allow_failures=False)

    assert torch.allclose(captured["params"], torch.tensor([0.41], dtype=torch.float32))  # type: ignore  # noqa: PGH003
    assert captured["smoothness"] == 6.5
    assert captured["noise_scale"] == 12.5
    assert captured["smoke_diffusivity"] == pytest.approx(7e-4)


def test_skip_nt_must_be_non_negative() -> None:
    with pytest.raises(ValueError, match="skip_nt must be non-negative"):
        ConditionedNavierStokes2D(skip_nt=-1)


def test_forward_samples_spatiotemporal_applies_skip_nt(monkeypatch) -> None:
    sim = ConditionedNavierStokes2D(
        return_timeseries=True,
        n=2,
        skip_nt=2,
        parameters_range={"buoyancy_y": (0.3, 0.3)},
    )

    n_time = 6
    channels = 3
    features_per_step = sim.n * sim.n * channels
    fake_y = torch.arange(
        float(n_time * features_per_step), dtype=torch.float32
    ).reshape(1, n_time * features_per_step)
    fake_x = torch.tensor([[0.3]], dtype=torch.float32)

    monkeypatch.setattr(sim, "sample_inputs", lambda n, random_seed=None: fake_x)  # noqa: ARG005
    monkeypatch.setattr(sim, "forward_batch", lambda x: (fake_y, fake_x))  # noqa: ARG005

    out = sim.forward_samples_spatiotemporal(n=1)
    assert out["data"].shape == (1, n_time - (1 + sim.skip_nt), sim.n, sim.n, channels)


def test_forward_samples_spatiotemporal_raises_when_skip_nt_too_large(
    monkeypatch,
) -> None:
    sim = ConditionedNavierStokes2D(
        return_timeseries=True,
        n=2,
        skip_nt=10,
        parameters_range={"buoyancy_y": (0.3, 0.3)},
    )

    n_time = 5
    channels = 3
    features_per_step = sim.n * sim.n * channels
    fake_y = torch.zeros((1, n_time * features_per_step), dtype=torch.float32)
    fake_x = torch.tensor([[0.3]], dtype=torch.float32)

    monkeypatch.setattr(sim, "sample_inputs", lambda n, random_seed=None: fake_x)  # noqa: ARG005
    monkeypatch.setattr(sim, "forward_batch", lambda x: (fake_y, fake_x))  # noqa: ARG005

    with pytest.raises(ValueError, match="skip_nt is too large"):
        sim.forward_samples_spatiotemporal(n=1)
