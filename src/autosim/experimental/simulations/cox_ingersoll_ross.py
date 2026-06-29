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

It is the second member of the Pearson-diffusion family alongside OU, and like OU
it has closed-form conditional moments (the affine/noncentral-chi-squared
moments), giving a state-dependent calibration oracle:

    E[X_t | X_0]   = X_0 e^{-kappa t} + theta (1 - e^{-kappa t})
    Var[X_t | X_0] = X_0 (sigma^2/kappa)(e^{-kappa t} - e^{-2 kappa t})
                     + theta (sigma^2 / 2 kappa)(1 - e^{-kappa t})^2

evaluated at ``t = lead * dt``. Run with a high Feller number ``2 kappa theta /
sigma^2 >> 1`` so the process stays off the zero boundary and the spread grows
cleanly with lead (rather than via a boundary blow-up).
"""

from __future__ import annotations

import numpy as np
import torch

from autosim.simulations.base import SpatioTemporalSimulator
from autosim.types import TensorLike


def cir_closed_form_mean(
    leads: torch.Tensor,
    *,
    kappa: float,
    theta: float,
    x0: float | torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Exact CIR conditional mean at integer lead times.

    Parameters
    ----------
    leads: torch.Tensor
        Integer lead steps (1-based), e.g. ``torch.arange(1, n_steps + 1)``.
    kappa: float
        Mean-reversion rate.
    theta: float
        Long-run mean (the reversion level).
    x0: float or torch.Tensor
        Initial state; broadcast against ``leads``.
    dt: float
        Euler-Maruyama step size.

    Returns
    -------
    torch.Tensor
        ``E[X_t | X_0] = X_0 e^{-kappa t} + theta (1 - e^{-kappa t})`` at
        ``t = lead * dt``, shape matching the broadcast of ``leads`` and ``x0``.
    """
    n = torch.as_tensor(leads, dtype=torch.float32)
    x0 = torch.as_tensor(x0, dtype=torch.float32)
    decay = torch.exp(-kappa * n * dt)
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
    """Exact CIR conditional variance at integer lead times.

    Parameters
    ----------
    leads: torch.Tensor
        Integer lead steps (1-based).
    kappa: float
        Mean-reversion rate.
    theta: float
        Long-run mean.
    sigma: float
        Diffusion coefficient (noise amplitude).
    x0: float or torch.Tensor
        Initial state; broadcast against ``leads``. The variance is
        state-dependent -- this is the property OU lacks.
    dt: float
        Euler-Maruyama step size.

    Returns
    -------
    torch.Tensor
        ``Var[X_t | X_0]`` at ``t = lead * dt`` (see the module docstring for the
        closed form), shape matching the broadcast of ``leads`` and ``x0``.
    """
    n = torch.as_tensor(leads, dtype=torch.float32)
    x0 = torch.as_tensor(x0, dtype=torch.float32)
    decay = torch.exp(-kappa * n * dt)
    transient = x0 * (sigma**2 / kappa) * (decay - decay**2)
    stationary = theta * (sigma**2 / (2.0 * kappa)) * (1.0 - decay) ** 2
    return transient + stationary


class CoxIngersollRoss(SpatioTemporalSimulator):
    r"""Full-truncation Euler integrator for CIR with closed-form moment oracles.

    The continuous-time dynamics are:

        dX = kappa (theta - X) dt + sigma sqrt(X) dW

    where ``dW`` is a Wiener increment. Integration uses the full-truncation
    Euler scheme, which introduces O(dt) discretisation bias relative to the
    continuous-time moments :func:`cir_closed_form_mean` / :func:`cir_closed_form_var`.

    Parameters
    ----------
    parameters_range: dict[str, tuple[float, float]], optional
        Bounds on the sampled initial condition ``x0``. Defaults to
        ``{"x0": (0.5, 2.0)}`` (positive, off the zero boundary).
    output_names: list[str], optional
        Human-readable name for the single output channel. Defaults to ``["x"]``.
    log_level: str, default="error"
        Logging verbosity passed to the base ``Simulator``.
    n_steps: int, default=64
        Number of Euler steps to record per trajectory.
    kappa: float, default=1.0
        Mean-reversion rate.
    theta: float, default=1.0
        Long-run mean.
    sigma: float, default=0.3
        Diffusion coefficient. With the defaults the Feller number
        ``2 kappa theta / sigma^2`` is ``~22 >> 1`` (well off the zero boundary).
    dt: float, default=0.05
        Time step for Euler integration.
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

    def _forward(self, x: TensorLike) -> TensorLike:
        """Integrate a single CIR trajectory from initial condition ``x0``.

        Parameters
        ----------
        x: TensorLike
            Input tensor of shape ``(1, 1)`` containing the initial condition.

        Returns
        -------
        TensorLike
            Flattened trajectory tensor of shape ``(1, n_steps)``.
        """
        if x.shape[0] != 1:
            msg = f"CoxIngersollRoss._forward expects a single input, got {x.shape[0]}"
            raise ValueError(msg)

        rng = np.random.default_rng()  # fresh process-noise path per trajectory
        xt = float(x.cpu().numpy()[0, 0])
        traj = np.empty(self.n_steps, dtype=np.float32)
        for t in range(self.n_steps):
            xt = self._step(xt, rng)
            traj[t] = xt

        return torch.from_numpy(traj).reshape(1, -1)

    def forward_samples_spatiotemporal(
        self,
        n: int,
        random_seed: int | None = None,
        ensure_exact_n: bool = False,
    ) -> dict:
        """Produce CIR trajectories along with the sampled initial conditions.

        Returns
        -------
        dict
            ``data``: Float32 tensor ``(batch, n_steps, 1, 1, 1)`` (singleton
            spatial axes -- CIR has no spatial extent). ``constant_scalars``:
            sampled ``x0``, shape ``(batch, 1)``. ``constant_fields``: ``None``
            (API parity with the twin simulators).
        """
        y, x = self._forward_batch_with_optional_retries(
            n=n, random_seed=random_seed, ensure_exact_n=ensure_exact_n
        )

        data = y.reshape(y.shape[0], self.n_steps, 1, 1, 1)
        return {
            "data": data,
            "constant_scalars": x,
            "constant_fields": None,
        }

    def mc_reference(
        self,
        x_state: torch.Tensor,
        n_draws: int,
        n_steps: int,
        random_seed: int | None = None,
    ) -> torch.Tensor:
        """Draw Monte Carlo trajectories from a fixed initial state.

        All ``n_draws`` trajectories start from the same ``x_state`` but receive
        independent Wiener increments, giving an empirical predictive
        distribution that can be checked against :func:`cir_closed_form_mean` /
        :func:`cir_closed_form_var`.

        Returns
        -------
        torch.Tensor
            Float32 tensor of shape ``(n_draws, n_steps, 1, 1, 1)``.
        """
        rng = np.random.default_rng(random_seed)
        x0 = float(torch.as_tensor(x_state).reshape(-1)[0])
        out = np.empty((n_draws, n_steps), dtype=np.float32)
        for d in range(n_draws):
            xt = x0
            for t in range(n_steps):
                xt = self._step(xt, rng)
                out[d, t] = xt

        return torch.from_numpy(out).reshape(n_draws, n_steps, 1, 1, 1)
