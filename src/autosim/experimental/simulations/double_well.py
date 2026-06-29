"""Double-well Langevin stochastic testbed.

Overdamped Langevin dynamics in the double-well potential U(x) = (x^2 - 1)^2,
integrated by Euler-Maruyama:

    x_{t+1} = x_t - 4 x_t (x_t^2 - 1) dt + c sqrt(dt) xi,   xi ~ N(0,1)

The potential has minima at x = ±1 separated by a barrier at x = 0 of height
U(0) = 1.  Starting near x0 ≈ 0 (the barrier top), trajectories escape into
either well with roughly equal probability, producing a bimodal predictive
distribution at intermediate lead times.  There is NO closed-form predictive
for this system; the Monte Carlo reference is the oracle.

This distinguishes it from the Ornstein-Uhlenbeck testbed where the
predictive is exactly Gaussian and a closed-form calibration oracle exists.
"""

from __future__ import annotations

import numpy as np
import torch

from autosim.simulations.base import SpatioTemporalSimulator
from autosim.types import TensorLike


class DoubleWell(SpatioTemporalSimulator):
    r"""Euler-Maruyama integrator for overdamped Langevin dynamics in a double well.

    The potential is U(x) = (x^2 - 1)^2, giving drift force
    -U'(x) = -4x(x^2 - 1).  The EM update is:

        x_{t+1} = x_t - 4 x_t (x_t^2 - 1) dt + c sqrt(dt) xi

    where xi ~ N(0, 1).  Wells sit at x = ±1; the barrier top at x = 0 has
    height U(0) = 1.  Starting near x0 ≈ 0 the predictive is bimodal at
    intermediate leads (the defining feature of this testbed
    hierarchy — emergent vs enforced uncertainty demo).

    Parameters
    ----------
    parameters_range: dict[str, tuple[float, float]], optional
        Bounds on the sampled initial condition ``x0``.  Defaults to
        ``{"x0": (-0.2, 0.2)}``, placing starts near the barrier top so
        trajectories escape into both wells.
    output_names: list[str], optional
        Human-readable name for the single output channel.  Defaults to ``["x"]``.
    log_level: str, default="error"
        Logging verbosity passed to the base ``Simulator``.
    n_steps: int, default=64
        Number of Euler-Maruyama integration steps to record per trajectory.
    c: float, default=0.5
        Diffusion coefficient (noise amplitude).
    dt: float, default=0.05
        Time step for Euler-Maruyama integration.
    """

    def __init__(
        self,
        parameters_range: dict[str, tuple[float, float]] | None = None,
        output_names: list[str] | None = None,
        log_level: str = "error",
        n_steps: int = 64,
        c: float = 0.5,
        dt: float = 0.05,
    ) -> None:
        if parameters_range is None:
            parameters_range = {"x0": (-0.2, 0.2)}
        if output_names is None:
            output_names = ["x"]

        super().__init__(parameters_range, output_names, log_level)

        self.n_steps = n_steps
        self.c = c
        self.dt = dt

    def _step(self, x: float, rng: np.random.Generator) -> float:
        """Apply one Euler-Maruyama step in the double-well potential.

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
            - 4.0 * x * (x**2 - 1.0) * self.dt
            + self.c * np.sqrt(self.dt) * rng.standard_normal()
        )

    def _forward(self, x: TensorLike) -> TensorLike:
        """Integrate a single double-well trajectory.

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
            msg = f"DoubleWell._forward expects a single input, got {x.shape[0]}"
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
        """Produce double-well trajectories along with the sampled initial conditions.

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
                spatial axes (last two ``1`` dimensions) are singleton — the
                double-well process has no spatial extent.
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
        independent Wiener increments, giving an empirical predictive distribution.
        As there is no closed-form predictive for the double-well potential, this
        Monte Carlo ensemble is the calibration oracle.

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
