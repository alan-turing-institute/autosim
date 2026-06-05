from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from autosim.simulations.spatiotemporal import ReactionDiffusion, reaction_diffusion


def test_simulate_reaction_diffusion_forwards_integrator_kwargs(monkeypatch) -> None:
    captured_kwargs = {}

    def fake_solve_ivp(*args, **kwargs):
        del args
        captured_kwargs.update(kwargs)
        t_eval = kwargs["t_eval"]
        n_features = len(kwargs["y0"])
        return SimpleNamespace(y=np.zeros((n_features, len(t_eval)), dtype=complex))

    monkeypatch.setattr(reaction_diffusion, "solve_ivp", fake_solve_ivp)

    reaction_diffusion.simulate_reaction_diffusion(
        np.array([1.3, 0.1]),
        return_timeseries=True,
        n=8,
        L=20,
        T=1.0,
        dt=0.25,
        integrator_kwargs={"rtol": 1e-12, "atol": 1e-12, "max_step": 0.05},
    )

    assert captured_kwargs["method"] == "RK45"
    assert captured_kwargs["rtol"] == 1e-12
    assert captured_kwargs["atol"] == 1e-12
    assert captured_kwargs["max_step"] == 0.05


def test_reaction_diffusion_forwards_integrator_kwargs(monkeypatch) -> None:
    captured_kwargs = {}

    def fake_simulate_reaction_diffusion(
        x,
        return_timeseries,
        n,
        L,
        T,
        dt,
        integrator_kwargs,
    ):
        del x, n, L, T, dt
        captured_kwargs.update(integrator_kwargs)
        if return_timeseries:
            return np.zeros((2, 4, 4)), np.zeros((2, 4, 4))
        return np.zeros((4, 4)), np.zeros((4, 4))

    monkeypatch.setattr(
        reaction_diffusion,
        "simulate_reaction_diffusion",
        fake_simulate_reaction_diffusion,
    )

    sim = ReactionDiffusion(
        return_timeseries=False,
        n=4,
        integrator_kwargs={"rtol": 1e-12, "atol": 1e-12, "max_step": 0.05},
    )
    sim._forward(torch.tensor([[1.3, 0.1]], dtype=torch.float32))

    assert captured_kwargs["method"] == "RK45"
    assert captured_kwargs["rtol"] == 1e-12
    assert captured_kwargs["atol"] == 1e-12
    assert captured_kwargs["max_step"] == 0.05
