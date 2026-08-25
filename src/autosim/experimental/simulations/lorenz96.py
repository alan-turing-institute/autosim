"""Stochastic Lorenz-96 ring testbed.

The Lorenz-96 model is a chaotic 1-D ring of N coupled sites:

    dX_i/dt = (X_{i+1} - X_{i-2}) * X_{i-1} - X_i + F

Add independent Gaussian process noise at each site (stochastic forcing), integrated
by Euler-Maruyama:

    X <- X + ((roll(X,-1) - roll(X,2)) * roll(X,1) - X + F) * dt + c*sqrt(dt)*xi

where xi ~ N(0, I_N) and ``roll`` is ``np.roll`` (ring/periodic boundary).

The standard chaotic regime uses F = 8.0.  Unlike the OU and double-well
scalar testbeds, the state is a 1-D field of N sites, so the predictive
spread is spatial and grows with lead time due to chaotic divergence amplified by
process noise.  There is no closed-form predictive; Monte Carlo is the oracle.

IC sampling: a single scalar parameter ``ic_scale`` (range 0-1) scales a fixed
random perturbation off the F-uniform fixed point (X_i = F for all i), giving a
natural spread of starting conditions while remaining in the chaotic attractor basin.
"""

from __future__ import annotations

import numpy as np
import torch

from autosim.simulations.base import SpatioTemporalSimulator
from autosim.types import TensorLike


class Lorenz96(SpatioTemporalSimulator):
    r"""Euler-Maruyama integrator for the stochastic Lorenz-96 ring.

    Deterministic RHS for site i (periodic/ring boundaries, indices mod N):

        dX_i/dt = (X_{i+1} - X_{i-2}) * X_{i-1} - X_i + F

    Stochastic Euler-Maruyama update (vectorised over all N sites):

        X <- X + rhs(X) * dt + c * sqrt(dt) * xi,   xi ~ N(0, I_N)

    The forcing F = 8.0 places the system in the standard chaotic regime where the
    Lyapunov time is roughly 0.2 model time units.  Process noise amplifies
    chaotic divergence so the cross-trajectory variance grows rapidly with lead.

    The state is a 1-D ring field of ``n_sites`` sites, stored in the
    spatiotemporal tensor as spatial shape (n_sites, 1) with 1 channel — matching
    the spatiotemporal convention ``(batch, time, space_0, space_1, channels)``.

    IC sampling uses a single scalar parameter ``ic_scale`` (range 0-1) that
    scales a deterministic perturbation pattern off the F-uniform fixed point:
    ``X_i = F + ic_scale * perturbation_i``, where ``perturbation_i`` is a
    fixed site-dependent offset drawn once at construction.  This keeps all
    initial conditions within the attractor basin while providing trajectory
    diversity across the training set.

    Args:
        parameters_range: Bounds on the sampled IC scale parameter. Defaults to
            ``{"ic_scale": (0.0, 1.0)}``.
        output_names: Names for the flattened outputs (length n_steps * n_sites).
            Defaults to ``["x"]`` (single flat vector; the reshape to spatial
            form is done in ``forward_samples_spatiotemporal``).
        log_level: Logging verbosity passed to the base ``Simulator``.
        n_steps: Number of Euler-Maruyama steps to record per trajectory.
        n_sites: Number of ring sites N.
        forcing: Lorenz-96 forcing constant F. F = 8.0 is the standard chaotic
            regime.
        c: Diffusion coefficient (noise amplitude).
        dt: Euler-Maruyama step size. The deterministic L96 is stable with
            4th-order RK for dt<=0.05; EM requires dt<=0.01 to stay bounded.
    """

    # Fixed perturbation pattern used to build initial conditions from ic_scale.
    # Constructed once at class level so all instances share the same IC geometry.
    _IC_PATTERN: np.ndarray = np.array(
        [
            0.5,
            -0.3,
            0.8,
            -0.6,
            0.2,
            0.9,
            -0.4,
            0.7,
            -0.1,
            0.6,
            -0.8,
            0.3,
            0.5,
            -0.7,
            0.4,
            -0.2,
            0.9,
            -0.5,
            0.1,
            0.8,
            -0.3,
            0.6,
            -0.9,
            0.2,
            0.7,
            -0.4,
            0.5,
            0.3,
            -0.6,
            0.8,
            -0.1,
            0.4,
            -0.7,
            0.9,
            -0.5,
            0.2,
            0.6,
            -0.8,
            0.3,
            -0.4,
        ],
        dtype=np.float64,
    )

    def __init__(
        self,
        parameters_range: dict[str, tuple[float, float]] | None = None,
        output_names: list[str] | None = None,
        log_level: str = "error",
        n_steps: int = 128,
        n_sites: int = 40,
        forcing: float = 8.0,
        c: float = 0.5,
        dt: float = 0.01,
    ) -> None:
        """Initialize the Lorenz-96 integrator and validate parameters."""
        if parameters_range is None:
            parameters_range = {"ic_scale": (0.0, 1.0)}
        if output_names is None:
            # Single flat output vector; the base Simulator sizes out_features from
            # this list (length = n_steps * n_sites when using forward_batch).
            # forward_samples_spatiotemporal reshapes it to (n, n_steps, n_sites, 1, 1).
            output_names = ["x"]

        super().__init__(parameters_range, output_names, log_level)

        self.n_steps = n_steps
        self.n_sites = n_sites
        self.forcing = forcing
        self.c = c
        self.dt = dt

        # Trim or tile the IC pattern to match n_sites.
        if n_sites <= len(self._IC_PATTERN):
            self._ic_pattern: np.ndarray = self._IC_PATTERN[:n_sites]
        else:
            reps = (n_sites // len(self._IC_PATTERN)) + 1
            self._ic_pattern = np.tile(self._IC_PATTERN, reps)[:n_sites]

    # ------------------------------------------------------------------
    # Dynamics helpers
    # ------------------------------------------------------------------

    def _l96_rhs(self, x: np.ndarray) -> np.ndarray:
        """Vectorised Lorenz-96 RHS for a state vector of length n_sites.

        Args:
            x: Current state, shape (n_sites,).

        Returns:
            Time derivative dX/dt, shape (n_sites,).

        Notes:
            Indices use periodic (ring) boundary conditions via ``np.roll``:
                roll(x, -1)[i] = x[(i+1) % N]   (X_{i+1})
                roll(x,  1)[i] = x[(i-1) % N]   (X_{i-1})
                roll(x,  2)[i] = x[(i-2) % N]   (X_{i-2})
        """
        return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + self.forcing

    def _step(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Apply one Euler-Maruyama step to the full L96 state vector.

        Args:
            x: Current state, shape (n_sites,).
            rng: NumPy random generator supplying the i.i.d. Wiener increments.

        Returns:
            Updated state after one step, shape (n_sites,).
        """
        xi = rng.standard_normal(self.n_sites)
        return x + self._l96_rhs(x) * self.dt + self.c * np.sqrt(self.dt) * xi

    # ------------------------------------------------------------------
    # Simulator interface
    # ------------------------------------------------------------------

    def _integrate(self, ic_scale: float, rng: np.random.Generator) -> np.ndarray:
        """Integrate one L96 trajectory from an IC-scale parameter.

        Args:
            ic_scale: Scales the fixed perturbation off the F-uniform fixed
                point to build the initial state.
            rng: Random generator supplying the process noise.

        Returns:
            Float32 trajectory of shape ``(n_steps, n_sites)`` (time outer, sites
            inner).
        """
        x = self.forcing + ic_scale * self._ic_pattern.astype(np.float64)
        traj = np.empty((self.n_steps, self.n_sites), dtype=np.float32)
        for t in range(self.n_steps):
            x = self._step(x, rng)
            traj[t] = x.astype(np.float32)
        return traj

    def _forward(self, x: TensorLike) -> TensorLike:
        """Integrate a single stochastic L96 trajectory.

        Args:
            x: Input tensor of shape ``(1, 1)`` containing the sampled IC scale
                parameter ``ic_scale``.

        Returns:
            Flattened trajectory tensor of shape ``(1, n_steps * n_sites)`` (time
            axis outer, site axis inner, row-major).
        """
        if x.shape[0] != 1:
            msg = (
                f"{type(self).__name__}._forward expects a single input, "
                f"got {x.shape[0]}"
            )
            raise ValueError(msg)

        ic_scale = float(x.cpu().numpy()[0, 0])
        rng = np.random.default_rng()  # fresh process-noise path per call
        traj = self._integrate(ic_scale, rng)
        return torch.from_numpy(traj.reshape(1, self.n_steps * self.n_sites))

    def forward_samples_spatiotemporal(
        self,
        n: int,
        random_seed: int | None = None,
        ensure_exact_n: bool = False,  # noqa: ARG002 -- generation is always exact
    ) -> dict:
        """Produce stochastic L96 trajectories with sampled initial conditions.

        Both the sampled IC scales and the process noise are seeded from
        ``random_seed``, so a given seed reproduces the full batch. L96 never
        fails, so ``ensure_exact_n`` is always satisfied without retries.

        Args:
            n: Number of trajectories to sample.
            random_seed: Seed for reproducible IC scales and process noise.
            ensure_exact_n: Accepted for API parity; the batch already contains
                exactly ``n`` trajectories.

        Returns:
            A dict with ``data`` (float32 ``(batch, n_steps, n_sites, 1, 1)``),
            ``constant_scalars`` (sampled ``ic_scale``, ``(batch, 1)``) and
            ``constant_fields`` (``None``, API parity).
        """
        x = self.sample_inputs(n, random_seed)
        rng = np.random.default_rng(random_seed)
        traj = np.stack([self._integrate(float(x[i, 0]), rng) for i in range(n)])
        data = torch.from_numpy(traj).reshape(n, self.n_steps, self.n_sites, 1, 1)
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
        """Draw Monte Carlo trajectories from a fixed L96 initial state.

        All ``n_draws`` trajectories start from the same ``x_state`` but receive
        independent Wiener increments, producing an empirical predictive
        distribution.  Because L96 is chaotic, the ensemble spread grows rapidly
        with lead time — this is the defining property of this chaotic toy
        testbed hierarchy.

        Args:
            x_state: Initial state; any shape that can be flattened to
                ``(n_sites,)``. The first ``n_sites`` elements are used as
                ``X_0``.
            n_draws: Number of independent Monte Carlo trajectories.
            n_steps: Number of Euler-Maruyama steps per trajectory.
            random_seed: Seed for reproducible draws.

        Returns:
            Float32 tensor of shape ``(n_draws, n_steps, n_sites, 1, 1)``.
        """
        rng = np.random.default_rng(random_seed)
        flat = (
            torch.as_tensor(x_state).detach().cpu().reshape(-1)[: self.n_sites].numpy()
        )
        x0 = flat.astype(np.float64)

        out = np.empty((n_draws, n_steps, self.n_sites), dtype=np.float32)
        for d in range(n_draws):
            X = x0.copy()
            for t in range(n_steps):
                X = self._step(X, rng)
                out[d, t] = X.astype(np.float32)

        return torch.from_numpy(out).reshape(n_draws, n_steps, self.n_sites, 1, 1)
