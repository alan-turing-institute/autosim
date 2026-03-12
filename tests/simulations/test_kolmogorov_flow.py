import torch


def test_kolmogorov_flow_rollout_smoke() -> None:
    from autosim.simulations import KolmogorovFlow

    sim = KolmogorovFlow(
        return_timeseries=True,
        log_level="warning",
        n=16,
        L=2 * 3.141592653589793,
        T=0.4,
        dt=0.2,
        kf=4,
        parameters_range={
            "nu": (0.005, 0.005),
            "forcing": (1.0, 1.0),
            "alpha": (0.05, 0.05),
        },
    )

    batch = sim.forward_samples_spatiotemporal(n=1, random_seed=0, ensure_exact_n=True)
    data = batch["data"]
    assert data.ndim == 5
    assert data.shape[0] == 1
    assert data.shape[-1] == len(sim.output_names)
    assert torch.isfinite(data).all()


def test_kolmogorov_flow_residual_and_diagnostics_smoke() -> None:
    from autosim.pde_residuals import (
        kolmogorov_flow_diagnostics,
        kolmogorov_flow_residual,
    )
    from autosim.simulations import KolmogorovFlow

    sim = KolmogorovFlow(
        return_timeseries=True,
        log_level="warning",
        n=16,
        L=2 * 3.141592653589793,
        T=0.4,
        dt=0.2,
        kf=4,
        parameters_range={
            "nu": (0.005, 0.005),
            "forcing": (1.0, 1.0),
            "alpha": (0.05, 0.05),
        },
    )
    payload = sim.forward_samples_spatiotemporal(
        n=1, random_seed=0, ensure_exact_n=True
    )

    res = kolmogorov_flow_residual(payload, simulator=sim, simulator_kwargs={})
    diag = kolmogorov_flow_diagnostics(payload, simulator=sim, simulator_kwargs={})
    assert isinstance(res, dict) and res
    assert isinstance(diag, dict) and diag
    assert all(isinstance(v, float) for v in res.values())
    assert all(isinstance(v, float) for v in diag.values())
