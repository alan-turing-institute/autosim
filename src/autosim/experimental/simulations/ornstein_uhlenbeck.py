"""Ornstein-Uhlenbeck stochastic testbed.

The continuous-time AR(1) process dX = -kappa (X - m) dt + c dW, integrated by
Euler-Maruyama. The scheme is *exactly* the discrete AR(1) recursion
``X_{t+1} = a X_t + kappa m dt + c sqrt(dt) xi`` with ``a = 1 - kappa dt``, so
its n-step predictive is exactly Gaussian with variance
``q (1 - a^{2n}) / (1 - a^2)`` where ``q = c^2 dt``. That discrete variance
(:func:`ou_closed_form_var`) is the closed-form calibration oracle -- it matches
the integrator with no discretisation bias. Process noise lives inside the
dynamics; each trajectory is one single-positive realisation.
"""

from __future__ import annotations

import numpy as np
import torch

from ._stochastic_base import ScalarSDESimulator


def ou_closed_form_var(
    leads: torch.Tensor,
    *,
    kappa: float,
    c: float,
    dt: float,
) -> torch.Tensor:
    """Exact Euler-Maruyama OU predictive variance at integer lead times.

    Args:
        leads: Integer lead steps (1-based), e.g. ``torch.arange(1, n_steps + 1)``.
        kappa: Mean-reversion rate.
        c: Diffusion coefficient (noise amplitude).
        dt: Euler-Maruyama step size.

    Returns:
        Variance at each lead step, shape matching ``leads``.

    Notes:
        The Euler-Maruyama scheme is the exact AR(1) recursion with coefficient
        ``a = 1 - kappa dt`` and innovation variance ``q = c^2 dt``, giving
        ``Var[X_n | X_0] = q (1 - a^{2n}) / (1 - a^2)``. This is the discrete
        variance of the *simulated* process (no O(dt) bias), unlike the
        continuous-time limit ``(c^2 / 2 kappa)(1 - exp(-2 kappa n dt))``, which
        it approaches as ``dt -> 0``.
    """
    n = torch.as_tensor(leads, dtype=torch.float32)
    a = 1.0 - kappa * dt  # Euler-Maruyama AR(1) coefficient
    q = c**2 * dt  # per-step innovation variance
    return q * (1.0 - (a**2) ** n) / (1.0 - a**2)


class OrnsteinUhlenbeck(ScalarSDESimulator):
    r"""Euler-Maruyama integrator for the OU process with closed-form variance oracle.

    The continuous-time dynamics are:

        dX = -kappa (X - m) dt + c dW

    where ``dW`` is a Wiener increment. Integration uses Euler-Maruyama, which is
    exactly the discrete AR(1) recursion, so :func:`ou_closed_form_var` (the
    discrete variance) matches the integrator exactly up to Monte-Carlo noise.

    Args:
        parameters_range: Bounds on the sampled initial condition ``x0``.
            Defaults to ``{"x0": (-2.0, 2.0)}``.
        output_names: Human-readable name for the single output channel.
            Defaults to ``["x"]``.
        log_level: Logging verbosity passed to the base ``Simulator``.
        n_steps: Number of Euler-Maruyama integration steps to record per
            trajectory.
        kappa: Mean-reversion rate.
        c: Diffusion coefficient (noise amplitude).
        m: Long-run mean.
        dt: Time step for Euler-Maruyama integration.
    """

    def __init__(
        self,
        parameters_range: dict[str, tuple[float, float]] | None = None,
        output_names: list[str] | None = None,
        log_level: str = "error",
        n_steps: int = 64,
        kappa: float = 1.0,
        c: float = 0.5,
        m: float = 0.0,
        dt: float = 0.05,
    ) -> None:
        """Initialize the OU integrator with its mean-reversion and noise parameters."""
        if parameters_range is None:
            parameters_range = {"x0": (-2.0, 2.0)}
        if output_names is None:
            output_names = ["x"]

        super().__init__(parameters_range, output_names, log_level)

        self.n_steps = n_steps
        self.kappa = kappa
        self.c = c
        self.m = m
        self.dt = dt
        self._validate_ar1_contractive(kappa, dt)

    def _step(self, x: float, rng: np.random.Generator) -> float:
        """Apply one Euler-Maruyama step.

        Args:
            x: Current state.
            rng: NumPy random generator supplying the Wiener increment.

        Returns:
            The updated state after one step.
        """
        return (
            x
            - self.kappa * (x - self.m) * self.dt
            + self.c * np.sqrt(self.dt) * rng.standard_normal()
        )
