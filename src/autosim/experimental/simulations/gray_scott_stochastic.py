"""Stochastic Gray-Scott reaction-diffusion testbed.

The Gray-Scott model is a two-species reaction-diffusion system on a periodic
grid:

    dU/dt = Du * lap(U) - U V^2 + f (1 - U)
    dV/dt = Dv * lap(V) + U V^2 - (f + k) V

with ``lap`` the periodic Laplacian (second-order central differences, the
finite-difference style of ``advection_diffusion._laplacian_periodic``). Add
independent Gaussian process noise to each species at each site, integrated by
Euler-Maruyama:

    U <- U + rhs_U(U, V) * dt + c * sqrt(dt) * xi_U
    V <- V + rhs_V(U, V) * dt + c * sqrt(dt) * xi_V

with ``xi_U, xi_V ~ N(0, I)`` drawn fresh each step.

This is the spatial bridge of the toy hierarchy: unlike the contractive
advection-diffusion control (where perturbations decay), the Gray-Scott reaction
terms sustain structure, so the injected process noise yields genuine
conditional spread that does not collapse with lead. The state is a 2-channel
``grid_size x grid_size`` field stored as ``(batch, time, grid, grid, 2)``,
matching the spatiotemporal convention ``(batch, time, space_0, space_1, channels)``.
There is no closed-form predictive; Monte Carlo is the oracle.

IC sampling uses a single scalar parameter ``ic_scale`` (range 0-1): a central
active seed (``U`` depressed, ``V`` raised) whose amplitude and a smooth
symmetry-breaking ripple are modulated by ``ic_scale``, giving trajectory
diversity off the trivial ``U=1, V=0`` homogeneous state while keeping every
initial condition in the active (pattern-sustaining) regime.
"""

from __future__ import annotations

import numpy as np
import torch

from autosim.simulations.base import SpatioTemporalSimulator
from autosim.types import TensorLike


def _laplacian_periodic(field: np.ndarray) -> np.ndarray:
    """Second-order central-difference Laplacian with periodic (wrap) boundaries.

    Mirrors ``advection_diffusion._laplacian_periodic`` with unit grid spacing
    (``dx = 1``): a 5-point stencil applied via ``np.roll`` on both spatial axes.

    Args:
        field: Scalar field on a 2-D grid, shape ``(grid, grid)``.

    Returns:
        Discrete Laplacian, same shape as ``field``.
    """
    return (
        np.roll(field, -1, axis=0)
        + np.roll(field, 1, axis=0)
        + np.roll(field, -1, axis=1)
        + np.roll(field, 1, axis=1)
        - 4.0 * field
    )


class GrayScottStochastic(SpatioTemporalSimulator):
    r"""Euler-Maruyama integrator for the stochastic Gray-Scott system.

    Deterministic right-hand sides on a periodic ``grid_size x grid_size`` grid:

        rhs_U = Du * lap(U) - U V^2 + f (1 - U)
        rhs_V = Dv * lap(V) + U V^2 - (f + k) V

    Stochastic Euler-Maruyama update (vectorised over the grid):

        U <- U + rhs_U * dt + c * sqrt(dt) * xi_U
        V <- V + rhs_V * dt + c * sqrt(dt) * xi_V,   xi ~ N(0, I)

    The default ``(f, k) = (0.06, 0.062)`` sit in the active spot regime where
    the reaction terms sustain ``V`` structure, so process noise produces
    conditional spread that grows with lead instead of decaying — the defining
    contrast with the contractive advection-diffusion control.

    Args:
        parameters_range: Bounds on the sampled IC scale parameter. Defaults
            to ``{"ic_scale": (0.0, 1.0)}``.
        output_names: Names for the flattened outputs. Defaults to ``["x"]``
            (a single flat vector; ``forward_samples_spatiotemporal`` reshapes
            to spatial form).
        log_level: Logging verbosity passed to the base ``Simulator``.
        n_steps: Number of Euler-Maruyama steps recorded per trajectory.
        grid_size: Side length of the square periodic grid.
        diffusion_u: Diffusion coefficient ``Du`` for species U.
        diffusion_v: Diffusion coefficient ``Dv`` for species V.
        feed: Feed rate ``f``.
        kill: Kill rate ``k``.
        c: Process-noise amplitude (per species, per site). At this amplitude
            the cross-draw spread grows monotonically over a ~96-step horizon
            (the reaction-sustained, non-contractive signature) without the
            field blowing up.
        dt: Euler-Maruyama step size. With unit grid spacing the explicit
            diffusion limit is ``dt <= 0.25 / max(Du, Dv)`` (~1.56 here), so
            ``dt = 1.0`` is stable.
    """

    def __init__(
        self,
        parameters_range: dict[str, tuple[float, float]] | None = None,
        output_names: list[str] | None = None,
        log_level: str = "error",
        n_steps: int = 96,
        grid_size: int = 8,
        diffusion_u: float = 0.16,
        diffusion_v: float = 0.08,
        feed: float = 0.06,
        kill: float = 0.062,
        c: float = 0.03,
        dt: float = 1.0,
    ) -> None:
        """Initialize the stochastic Gray-Scott integrator and validate parameters."""
        if parameters_range is None:
            parameters_range = {"ic_scale": (0.0, 1.0)}
        if output_names is None:
            # Single flat output vector; forward_samples_spatiotemporal reshapes
            # it to (n, n_steps, grid_size, grid_size, 2).
            output_names = ["x"]

        super().__init__(parameters_range, output_names, log_level)

        self.n_steps = n_steps
        self.grid_size = grid_size
        self.diffusion_u = diffusion_u
        self.diffusion_v = diffusion_v
        self.feed = feed
        self.kill = kill
        self.c = c
        self.dt = dt

        # Fixed smooth, mean-zero ripple used to break the seed's symmetry and
        # diversify initial conditions across ``ic_scale`` (deterministic; no RNG).
        ii, jj = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing="ij")
        self._ripple: np.ndarray = (
            np.sin(2.0 * np.pi * ii / grid_size) * np.cos(2.0 * np.pi * jj / grid_size)
        ).astype(np.float64)

    # ------------------------------------------------------------------
    # Dynamics helpers
    # ------------------------------------------------------------------

    def _initial_condition(self, ic_scale: float) -> tuple[np.ndarray, np.ndarray]:
        """Build the (U, V) initial fields from the scalar ``ic_scale``.

        A central square seed depresses ``U`` and raises ``V`` into the active
        regime; ``ic_scale`` modulates both the seed amplitude and a smooth
        symmetry-breaking ripple so different trajectories start from distinct
        (but all active) states.

        Args:
            ic_scale: Sampled IC parameter in ``[0, 1]``.

        Returns:
            The ``(U, V)`` fields, each shape ``(grid_size, grid_size)``.
        """
        g = self.grid_size
        u = np.ones((g, g), dtype=np.float64)
        v = np.zeros((g, g), dtype=np.float64)

        amp = 0.2 + 0.1 * ic_scale  # active seed amplitude in [0.2, 0.3]
        lo, hi = g // 2 - 1, g // 2 + 2  # central 3x3 block
        u[lo:hi, lo:hi] = 1.0 - 2.0 * amp
        v[lo:hi, lo:hi] = amp

        # Small ic_scale-scaled ripple on V; clip to keep concentrations valid.
        v = np.clip(v + 0.05 * ic_scale * self._ripple, 0.0, None)
        return u, v

    def _rhs(self, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Deterministic Gray-Scott right-hand sides for both species.

        Args:
            u: Current ``U`` field, shape ``(grid_size, grid_size)``.
            v: Current ``V`` field, shape ``(grid_size, grid_size)``.

        Returns:
            ``(dU/dt, dV/dt)``, same shapes.
        """
        reaction = u * v * v
        du = (
            self.diffusion_u * _laplacian_periodic(u) - reaction + self.feed * (1.0 - u)
        )
        dv = (
            self.diffusion_v * _laplacian_periodic(v)
            + reaction
            - (self.feed + self.kill) * v
        )
        return du, dv

    def _step(
        self, u: np.ndarray, v: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply one stochastic Euler-Maruyama step to the (U, V) fields.

        Args:
            u: Current ``U`` field, shape ``(grid_size, grid_size)``.
            v: Current ``V`` field, shape ``(grid_size, grid_size)``.
            rng: Generator supplying the i.i.d. Wiener increments for each
                species.

        Returns:
            Updated ``(U, V)`` after one step.
        """
        du, dv = self._rhs(u, v)
        noise = self.c * np.sqrt(self.dt)
        g = self.grid_size
        u_next = u + du * self.dt + noise * rng.standard_normal((g, g))
        v_next = v + dv * self.dt + noise * rng.standard_normal((g, g))
        return u_next, v_next

    # ------------------------------------------------------------------
    # Simulator interface
    # ------------------------------------------------------------------

    def _stack_channels(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Stack (U, V) fields into a channel-last frame ``(grid, grid, 2)``."""
        return np.stack([u, v], axis=-1)

    def _integrate(self, ic_scale: float, rng: np.random.Generator) -> np.ndarray:
        """Integrate one stochastic Gray-Scott trajectory from an IC-scale parameter.

        Args:
            ic_scale: Sampled IC parameter modulating the active seed and ripple.
            rng: Random generator supplying the process noise.

        Returns:
            Float32 trajectory of shape ``(n_steps, grid, grid, 2)`` (time outer,
            the ``(grid, grid, channel)`` frame inner).
        """
        u, v = self._initial_condition(ic_scale)
        g = self.grid_size
        traj = np.empty((self.n_steps, g, g, 2), dtype=np.float32)
        for t in range(self.n_steps):
            u, v = self._step(u, v, rng)
            traj[t] = self._stack_channels(u, v).astype(np.float32)
        return traj

    def _forward(self, x: TensorLike) -> TensorLike:
        """Integrate a single stochastic Gray-Scott trajectory.

        Args:
            x: Input tensor of shape ``(1, 1)`` containing the sampled
                ``ic_scale``.

        Returns:
            Flattened trajectory of shape
            ``(1, n_steps * grid_size * grid_size * 2)``, row-major with time
            outer and the ``(grid, grid, channel)`` frame inner.
        """
        if x.shape[0] != 1:
            msg = (
                f"GrayScottStochastic._forward expects a single input, got {x.shape[0]}"
            )
            raise ValueError(msg)

        ic_scale = float(x.cpu().numpy()[0, 0])
        rng = np.random.default_rng()  # fresh process-noise path per trajectory
        traj = self._integrate(ic_scale, rng)
        return torch.from_numpy(traj.reshape(1, -1))

    def forward_samples_spatiotemporal(
        self,
        n: int,
        random_seed: int | None = None,
        ensure_exact_n: bool = False,  # noqa: ARG002 -- generation is always exact
    ) -> dict:
        """Produce stochastic Gray-Scott trajectories with sampled ICs.

        Both the sampled IC scales and the process noise are seeded from
        ``random_seed``, so a given seed reproduces the full batch. This
        integrator never fails, so ``ensure_exact_n`` is always satisfied
        without retries.

        Args:
            n: Number of trajectories to sample.
            random_seed: Seed for reproducible initial-condition draws and
                process noise.
            ensure_exact_n: Accepted for API parity; the batch already
                contains exactly ``n`` trajectories.

        Returns:
            Dictionary with keys:

            ``data``
                Float32 tensor of shape ``(batch, n_steps, grid, grid, 2)``.
            ``constant_scalars``
                Sampled ``ic_scale`` parameters, shape ``(batch, 1)``.
            ``constant_fields``
                Always ``None``; placeholder for API consistency with the
                ``AdvectionDiffusion`` twin simulator.
        """
        x = self.sample_inputs(n, random_seed)
        rng = np.random.default_rng(random_seed)
        g = self.grid_size
        traj = np.stack([self._integrate(float(x[i, 0]), rng) for i in range(n)])
        data = torch.from_numpy(traj).reshape(n, self.n_steps, g, g, 2)
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
        """Draw Monte Carlo trajectories from a fixed Gray-Scott initial state.

        All ``n_draws`` trajectories start from the same ``x_state`` but receive
        independent process-noise paths, producing the empirical predictive
        distribution. The reaction terms sustain ``V`` structure, so the
        cross-draw spread is non-trivial and grows with lead — the property that
        distinguishes this rung from the contractive advection-diffusion control.

        Args:
            x_state: Initial state; any shape whose last axis is the 2
                channels and whose leading ``grid_size * grid_size`` spatial
                entries define ``(U, V)``. Accepts ``(grid, grid, 2)``,
                ``(grid*grid, 2)``, or any shape that reshapes to
                ``(grid, grid, 2)``.
            n_draws: Number of independent Monte Carlo trajectories.
            n_steps: Number of Euler-Maruyama steps per trajectory.
            random_seed: Seed for reproducible draws.

        Returns:
            Float32 tensor of shape ``(n_draws, n_steps, grid, grid, 2)``.
        """
        g = self.grid_size
        frame = torch.as_tensor(x_state).reshape(g, g, 2).detach().cpu().numpy()
        u0 = frame[..., 0].astype(np.float64)
        v0 = frame[..., 1].astype(np.float64)

        rng = np.random.default_rng(random_seed)
        out = np.empty((n_draws, n_steps, g, g, 2), dtype=np.float32)
        for d in range(n_draws):
            u, v = u0.copy(), v0.copy()
            for t in range(n_steps):
                u, v = self._step(u, v, rng)
                out[d, t] = self._stack_channels(u, v).astype(np.float32)

        return torch.from_numpy(out)
