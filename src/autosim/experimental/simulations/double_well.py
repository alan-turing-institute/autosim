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

from ._stochastic_base import ScalarSDESimulator


class DoubleWell(ScalarSDESimulator):
    r"""Euler-Maruyama integrator for overdamped Langevin dynamics in a double well.

    The potential is U(x) = (x^2 - 1)^2, giving drift force
    -U'(x) = -4x(x^2 - 1).  The EM update is:

        x_{t+1} = x_t - 4 x_t (x_t^2 - 1) dt + c sqrt(dt) xi

    where xi ~ N(0, 1).  Wells sit at x = ±1; the barrier top at x = 0 has
    height U(0) = 1.  Starting near x0 ≈ 0 the predictive is bimodal at
    intermediate leads (the defining feature of this testbed
    hierarchy — emergent vs enforced uncertainty demo).

    Args:
        parameters_range: Bounds on the sampled initial condition ``x0``.
            Defaults to ``{"x0": (-0.2, 0.2)}``, placing starts near the
            barrier top so trajectories escape into both wells.
        output_names: Human-readable name for the single output channel.
            Defaults to ``["x"]``.
        log_level: Logging verbosity passed to the base ``Simulator``.
        n_steps: Number of Euler-Maruyama integration steps to record per
            trajectory.
        c: Diffusion coefficient (noise amplitude).
        dt: Time step for Euler-Maruyama integration.
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
        """Initialize the double-well integrator and validate parameters."""
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

        Args:
            x: Current state.
            rng: NumPy random generator supplying the Wiener increment.

        Returns:
            The updated state after one step.
        """
        return (
            x
            - 4.0 * x * (x**2 - 1.0) * self.dt
            + self.c * np.sqrt(self.dt) * rng.standard_normal()
        )
