"""Cox-Ingersoll-Ross (square-root) stochastic testbed -- OU's nonlinear sibling.

The mean-reverting square-root diffusion

    dX = kappa (theta - X) dt + sigma sqrt(X) dW,

integrated by the full-truncation Euler-Maruyama scheme (Lord et al. 2010): the
state enters the drift and diffusion through ``X^+ = max(X, 0)`` so a negative
excursion cannot produce a NaN square root, while the recorded state is left
un-truncated. Unlike OU, the one-step transition is a (scaled) noncentral
chi-squared -- skewed and heteroscedastic (the noise scales with sqrt(X)) -- so a
Gaussian one-step head accumulates spread error under compounding, which is what
makes CIR the nonlinear *value* test for the continuous-lead ARCI head.

It is the second member of the Pearson-diffusion family alongside OU. Because the
integrator is the full-truncation Euler scheme, the calibration oracle is the
*discrete* conditional mean/variance of that scheme (valid off the zero boundary,
where ``X^+ = X``):

    E[X_n | X_0]   = X_0 a^n + theta (1 - a^n)
    Var[X_n | X_0] = q [ theta (1 - a^{2n}) / (1 - a^2)
                         + (X_0 - theta) a^{n-1} (1 - a^n) / (1 - a) ]

with ``a = 1 - kappa dt`` and ``q = sigma^2 dt``. These match the integrator with
no O(dt) bias and approach the continuous-time CIR moments as ``dt -> 0``. Run
with a high Feller number ``2 kappa theta / sigma^2 >> 1`` so the process stays
off the zero boundary and the spread grows cleanly with lead.
"""

from __future__ import annotations

import numpy as np
import torch

from ._stochastic_base import ScalarSDESimulator


def cir_closed_form_mean(
    leads: torch.Tensor,
    *,
    kappa: float,
    theta: float,
    x0: float | torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Exact full-truncation-Euler CIR conditional mean at integer lead times.

    Args:
        leads: Integer lead steps (1-based), e.g.
            ``torch.arange(1, n_steps + 1)``.
        kappa: Mean-reversion rate.
        theta: Long-run mean (the reversion level).
        x0: Initial state; broadcast against ``leads``.
        dt: Euler step size.

    Returns:
        ``E[X_n | X_0] = X_0 a^n + theta (1 - a^n)`` with ``a = 1 - kappa dt``,
        shape matching the broadcast of ``leads`` and ``x0``.
    """
    n = torch.as_tensor(leads, dtype=torch.float32)
    x0 = torch.as_tensor(x0, dtype=torch.float32)
    a = 1.0 - kappa * dt  # Euler AR(1) coefficient of the conditional mean
    decay = a**n
    return x0 * decay + theta * (1.0 - decay)


def cir_closed_form_var(
    leads: torch.Tensor,
    *,
    kappa: float,
    theta: float,
    sigma: float,
    x0: float | torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Exact full-truncation-Euler CIR conditional variance at integer lead times.

    Args:
        leads: Integer lead steps (1-based).
        kappa: Mean-reversion rate.
        theta: Long-run mean.
        sigma: Diffusion coefficient (noise amplitude).
        x0: Initial state; broadcast against ``leads``. The variance is
            state-dependent -- this is the property OU lacks.
        dt: Euler step size.

    Returns:
        ``Var[X_n | X_0]`` of the Euler scheme (see the module docstring),
        shape matching the broadcast of ``leads`` and ``x0``. Valid off the zero
        boundary, where the full-truncation ``X^+`` equals ``X``.
    """
    n = torch.as_tensor(leads, dtype=torch.float32)
    x0 = torch.as_tensor(x0, dtype=torch.float32)
    a = 1.0 - kappa * dt  # Euler AR(1) coefficient
    q = sigma**2 * dt  # per-step innovation scale (variance is q * E[X])
    stationary = theta * (1.0 - (a**2) ** n) / (1.0 - a**2)
    transient = (x0 - theta) * a ** (n - 1.0) * (1.0 - a**n) / (1.0 - a)
    return q * (stationary + transient)


class CoxIngersollRoss(ScalarSDESimulator):
    r"""Full-truncation Euler integrator for CIR with closed-form moment oracles.

    The continuous-time dynamics are:

        dX = kappa (theta - X) dt + sigma sqrt(X) dW

    where ``dW`` is a Wiener increment. Integration uses the full-truncation
    Euler scheme; the oracles :func:`cir_closed_form_mean` /
    :func:`cir_closed_form_var` are the discrete moments of that scheme (no O(dt)
    bias), valid off the zero boundary that the high Feller default maintains.

    Args:
        parameters_range: Bounds on the sampled initial condition ``x0``.
            Defaults to ``{"x0": (0.5, 2.0)}`` (positive, off the zero
            boundary).
        output_names: Human-readable name for the single output channel.
            Defaults to ``["x"]``.
        log_level: Logging verbosity passed to the base ``Simulator``.
            Defaults to ``"error"``.
        n_steps: Number of Euler steps to record per trajectory. Defaults
            to 64.
        kappa: Mean-reversion rate. Defaults to 1.0.
        theta: Long-run mean. Defaults to 1.0.
        sigma: Diffusion coefficient. With the defaults the Feller number
            ``2 kappa theta / sigma^2`` is ``~22 >> 1`` (well off the zero
            boundary). Defaults to 0.3.
        dt: Time step for Euler integration. Defaults to 0.05.
    """

    def __init__(
        self,
        parameters_range: dict[str, tuple[float, float]] | None = None,
        output_names: list[str] | None = None,
        log_level: str = "error",
        n_steps: int = 64,
        kappa: float = 1.0,
        theta: float = 1.0,
        sigma: float = 0.3,
        dt: float = 0.05,
    ) -> None:
        """Initialize the CIR integrator and validate parameters."""
        if parameters_range is None:
            parameters_range = {"x0": (0.5, 2.0)}
        if output_names is None:
            output_names = ["x"]

        super().__init__(parameters_range, output_names, log_level)

        self.n_steps = n_steps
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self._validate_ar1_contractive(kappa, dt)

    @property
    def feller(self) -> float:
        """The Feller number ``2 kappa theta / sigma^2`` (>> 1 stays off zero)."""
        return 2.0 * self.kappa * self.theta / self.sigma**2

    def _step(self, x: float, rng: np.random.Generator) -> float:
        """Apply one full-truncation Euler step.

        The drift and diffusion use ``x^+ = max(x, 0)`` so a negative excursion
        cannot take the square root of a negative number; the returned state is
        left un-truncated (the full-truncation scheme).
        """
        x_pos = max(x, 0.0)
        return (
            x
            + self.kappa * (self.theta - x_pos) * self.dt
            + self.sigma * np.sqrt(x_pos) * np.sqrt(self.dt) * rng.standard_normal()
        )
