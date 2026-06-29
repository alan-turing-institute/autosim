"""Ornstein-Uhlenbeck stochastic testbed.

The continuous-time AR(1) process dX = -kappa (X - m) dt + c dW, integrated by
Euler-Maruyama. The n-step predictive is exactly Gaussian with variance
(c^2 / 2 kappa)(1 - exp(-2 kappa n dt)), giving a closed-form calibration
oracle. Process noise lives inside the dynamics; each trajectory is one
single-positive realisation.
"""

from __future__ import annotations

import numpy as np
import torch

from autosim.simulations.base import SpatioTemporalSimulator
from autosim.types import TensorLike


def ou_closed_form_var(
    leads: torch.Tensor,
    *,
    kappa: float,
    c: float,
    dt: float,
) -> torch.Tensor:
    """Exact OU predictive variance at integer lead times.

    Parameters
    ----------
    leads: torch.Tensor
        Integer lead steps (1-based), e.g. ``torch.arange(1, n_steps + 1)``.
    kappa: float
        Mean-reversion rate.
    c: float
        Diffusion coefficient (noise amplitude).
    dt: float
        Euler-Maruyama step size.

    Returns
    -------
    torch.Tensor
        Variance at each lead step, shape matching ``leads``.

    Notes
    -----
    Derived from the continuous OU variance
    ``Var[X_t | X_0] = (c^2 / 2k)(1 - exp(-2k t))``, evaluated at t = lead * dt.
    """
    n = torch.as_tensor(leads, dtype=torch.float32)
    return (c**2 / (2 * kappa)) * (1.0 - torch.exp(-2.0 * kappa * n * dt))


class OrnsteinUhlenbeck(SpatioTemporalSimulator):
    r"""Euler-Maruyama integrator for the OU process with closed-form variance oracle.

    The continuous-time dynamics are:

        dX = -kappa (X - m) dt + c dW

    where ``dW`` is a Wiener increment. Integration uses Euler-Maruyama, which
    introduces O(dt) discretisation bias relative to the continuous-time variance
    ``ou_closed_form_var``.

    Parameters
    ----------
    parameters_range: dict[str, tuple[float, float]], optional
        Bounds on the sampled initial condition ``x0``.  Defaults to
        ``{"x0": (-2.0, 2.0)}``.
    output_names: list[str], optional
        Human-readable name for the single output channel.  Defaults to ``["x"]``.
    log_level: str, default="error"
        Logging verbosity passed to the base ``Simulator``.
    n_steps: int, default=64
        Number of Euler-Maruyama integration steps to record per trajectory.
    kappa: float, default=1.0
        Mean-reversion rate.
    c: float, default=0.5
        Diffusion coefficient (noise amplitude).
    m: float, default=0.0
        Long-run mean.
    dt: float, default=0.05
        Time step for Euler-Maruyama integration.
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

    def _step(self, x: float, rng: np.random.Generator) -> float:
        """Apply one Euler-Maruyama step.

        Parameters
        ----------
        x: float
            Current state.
        rng: np.random.Generator
            NumPy random generator supplying the Wiener increment.

        Returns
        -------
        float
            Updated state after one step.
        """
        return (
            x
            - self.kappa * (x - self.m) * self.dt
            + self.c * np.sqrt(self.dt) * rng.standard_normal()
        )

    def _forward(self, x: TensorLike) -> TensorLike:
        """Integrate a single OU trajectory.

        Parameters
        ----------
        x: TensorLike
            Input tensor of shape ``(1, 1)`` containing the initial condition ``x0``.

        Returns
        -------
        TensorLike
            Flattened trajectory tensor of shape ``(1, n_steps)``.
        """
        if x.shape[0] != 1:
            msg = f"OrnsteinUhlenbeck._forward expects a single input, got {x.shape[0]}"
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
        """Produce OU trajectories along with the sampled initial conditions.

        Parameters
        ----------
        n: int
            Number of trajectories to sample.
        random_seed: int, optional
            Seed for reproducible initial-condition draws.

        Returns
        -------
        dict
            Dictionary with keys:

            ``data``
                Float32 tensor of shape ``(batch, n_steps, 1, 1, 1)``.  The
                spatial axes (last two ``1`` dimensions) are singleton — the OU
                process has no spatial extent.
            ``constant_scalars``
                Sampled ``x0`` initial conditions, shape ``(batch, 1)``.
            ``constant_fields``
                Always ``None``; placeholder kept for API consistency with the
                twin ``AdvectionDiffusion`` simulator.
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
        independent Wiener increments, giving an empirical predictive distribution
        that can be compared against ``ou_closed_form_var``.

        Parameters
        ----------
        x_state: torch.Tensor
            Initial state; any shape — the first scalar value is used as ``x0``.
        n_draws: int
            Number of independent Monte Carlo trajectories.
        n_steps: int
            Number of Euler-Maruyama steps per trajectory.
        random_seed: int, optional
            Seed for reproducible draws.

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
