"""Chaotic Lorenz-96 with KNOWN spatially-correlated forcing (l96c).

Rung 3 of the toy testbed hierarchy, but with smooth ring-correlated process
noise instead of the white forcing of :class:`Lorenz96`.

Same dynamics and chaotic regime as the white-forcing :class:`Lorenz96` ring
(N = 40 sites, F = 8.0, c = 0.5, dt = 0.01):

    dX_i/dt = (X_{i+1} - X_{i-2}) * X_{i-1} - X_i + F

but the process forcing is correlated across sites:

    X <- X + rhs(X) * dt + xi ,    xi ~ N(0, c^2 * dt * C)

where ``C`` is a smooth, circulant ring-correlation with UNIT diagonal -- so the
per-site forcing VARIANCE is identical to white l96 (``c^2 * dt``) and only the
cross-site CORRELATION is added. ``C_ij = exp(-d_ij^2 / (2 ell^2))`` with ``d_ij``
the ring distance; ``ell = 2`` gives a nearest-neighbour forcing correlation of
``exp(-1/(2 ell^2)) ~ 0.88``.

For stability over long integrations the step is sub-stepped Euler-Maruyama
(``n_substeps`` increments of ``dt / n_substeps``); the per-lead (recorded-step)
forcing covariance is unchanged because the ``n_substeps`` independent increments
accumulate to ``n_substeps * c^2 * (dt/n_substeps) * C = c^2 * dt * C``.

Because l96c is chaotic and nonlinear there is **no closed-form one-step
predictive covariance**; the oracle is direct Monte-Carlo forward simulation
(:meth:`Lorenz96Correlated.mc_reference`). The one-step *forcing* covariance is
however known exactly (:func:`l96c_forcing_cov`).

The data-generating process uses a fixed reference discretisation: a ring
correlation, a Cholesky forcing factor, sub-stepping, burn-in and per-split
seeds.

Twins of :class:`Lorenz96` (dynamics, state layout, chaos) and
:class:`CorrelatedDiffusion1D` (generation path plus a module-level
forcing-covariance helper).
"""

from __future__ import annotations

import numpy as np
import torch

from autosim.simulations.base import SpatioTemporalSimulator
from autosim.types import TensorLike


def _ring_correlation(n_sites: int, ell: float) -> np.ndarray:
    """Smooth circulant ring correlation ``C`` with UNIT diagonal.

    Uses a fixed reference discretisation: ``C_ij = exp(-d_ij^2 / (2 ell^2))`` with
    ``d_ij`` the periodic (ring) distance between sites ``i`` and ``j``. The
    diagonal is ``1`` so ``C`` is a correlation matrix, not a covariance.

    Parameters
    ----------
    n_sites: int
        Number of ring sites ``N``.
    ell: float
        Correlation length in sites (sets the off-diagonal decay).

    Returns
    -------
    np.ndarray
        The ``(n_sites, n_sites)`` correlation matrix (float64).
    """
    ii = np.arange(n_sites)
    dist = np.minimum(
        np.abs(ii[:, None] - ii[None, :]),
        n_sites - np.abs(ii[:, None] - ii[None, :]),
    )
    return np.exp(-(dist.astype(np.float64) ** 2) / (2.0 * ell**2))


def l96c_forcing_cov(
    *,
    n_sites: int = 40,
    c: float = 0.5,
    dt: float = 0.01,
    ell: float = 2.0,
) -> torch.Tensor:
    """Exact one-step (recorded-step) forcing covariance ``c^2 * dt * C``.

    The forcing accumulated over one recorded step has covariance
    ``c^2 * dt * C``: the diagonal is the white-l96 per-site variance
    ``c^2 * dt`` (so the per-site marginals tie with white l96) and the
    off-diagonal carries the added ring correlation ``C``. This is the forcing
    covariance only -- l96c has no closed-form *predictive* covariance because
    the dynamics are chaotic (use :meth:`Lorenz96Correlated.mc_reference`).

    Parameters
    ----------
    n_sites, c, dt, ell:
        Ring DGP parameters; defaults match :class:`Lorenz96Correlated`.

    Returns
    -------
    torch.Tensor
        The ``(n_sites, n_sites)`` forcing covariance, float32.
    """
    c_mat = _ring_correlation(n_sites, ell)
    return torch.from_numpy((c**2) * dt * c_mat).float()


class Lorenz96Correlated(SpatioTemporalSimulator):
    r"""Chaotic Lorenz-96 ring with smooth ring-correlated process forcing.

    Sub-stepped Euler-Maruyama integrator. Deterministic RHS for site ``i``
    (periodic/ring boundaries, indices mod N):

        dX_i/dt = (X_{i+1} - X_{i-2}) * X_{i-1} - X_i + F

    Sub-stepped stochastic update (``n_substeps`` increments of
    ``dt_sub = dt / n_substeps``):

        X <- X + rhs(X) * dt_sub + c * sqrt(dt_sub) * (eta @ L^T),
        eta ~ N(0, I_N),   L L^T = C

    where ``C`` is the smooth ring correlation with unit diagonal and ``L`` its
    Cholesky factor. The forcing F = 8.0 places the system in the standard
    chaotic regime; correlated process noise amplifies chaotic divergence so the
    cross-trajectory spread grows rapidly with lead, but -- unlike white-forcing
    :class:`Lorenz96` -- the spread carries genuine cross-site correlation.

    The per-site forcing variance equals white l96's (``c^2 * dt``): only the
    cross-site correlation is added (see :func:`l96c_forcing_cov`).

    The state is a 1-D ring field of ``n_sites`` sites, stored in the
    spatiotemporal tensor as spatial shape (n_sites, 1) with 1 channel -- matching
    the spatiotemporal convention ``(batch, time, space_0, space_1, channels)``.

    Trajectories are produced directly by :meth:`forward_samples_spatiotemporal`
    from a seeded burn-in off the F-uniform fixed point with a small Gaussian
    perturbation (``X = F + ic_scale * eta``); the
    ``_forward`` single-trajectory hook is retained only for ``Simulator`` API
    parity.

    Parameters
    ----------
    parameters_range: dict[str, tuple[float, float]], optional
        Placeholder bound for the (unused) sampled input; defaults to
        ``{"x0": (-1.0, 1.0)}`` for API parity with the diff1d twin.
    output_names: list[str], optional
        Channel name; defaults to ``["x"]``.
    log_level: str, default="error"
        Logging verbosity passed to the base ``Simulator``.
    n_steps: int, default=128
        Number of recorded Euler-Maruyama steps per trajectory (``T``).
    burn: int, default=150
        Burn-in steps discarded before recording.
    n_sites: int, default=40
        Number of ring sites ``N``.
    forcing: float, default=8.0
        Lorenz-96 forcing constant ``F`` (8.0 is the standard chaotic regime).
    c: float, default=0.5
        Diffusion coefficient (noise amplitude).
    dt: float, default=0.01
        Recorded-step size. EM stability over long integrations is provided by
        the sub-stepping.
    ell: float, default=2.0
        Ring correlation length in sites (sets the cross-site forcing
        correlation; ``ell = 2`` gives nearest-neighbour corr ~0.88).
    n_substeps: int, default=10
        Number of Euler-Maruyama sub-steps per recorded step (``SUB``); each
        advances ``dt / n_substeps``.
    ic_scale: float, default=0.1
        Standard deviation of the Gaussian IC perturbation off the F-uniform
        fixed point (using ``F + 0.1 * randn``).
    """

    def __init__(
        self,
        parameters_range: dict[str, tuple[float, float]] | None = None,
        output_names: list[str] | None = None,
        log_level: str = "error",
        n_steps: int = 128,
        burn: int = 150,
        n_sites: int = 40,
        forcing: float = 8.0,
        c: float = 0.5,
        dt: float = 0.01,
        ell: float = 2.0,
        n_substeps: int = 10,
        ic_scale: float = 0.1,
    ) -> None:
        if parameters_range is None:
            parameters_range = {"x0": (-1.0, 1.0)}
        if output_names is None:
            output_names = ["x"]
        if n_substeps < 1:
            msg = f"n_substeps must be >= 1, got {n_substeps}"
            raise ValueError(msg)

        super().__init__(parameters_range, output_names, log_level)

        self.n_steps = n_steps
        self.burn = burn
        self.n_sites = n_sites
        self.forcing = forcing
        self.c = c
        self.dt = dt
        self.ell = ell
        self.n_substeps = n_substeps
        self.ic_scale = ic_scale
        self.dt_sub = dt / n_substeps

        # Cholesky factor of the ring correlation, with the same 1e-9 jitter as
        # the reference generator so the forcing draws (and thus trajectories) are byte
        # identical. C has unit diagonal -> the per-site forcing variance is
        # exactly the white-l96 value c^2 * dt after sub-step accumulation.
        self._C: np.ndarray = _ring_correlation(n_sites, ell)
        self._chol: np.ndarray = np.linalg.cholesky(self._C + 1e-9 * np.eye(n_sites))

    # ------------------------------------------------------------------
    # Dynamics helpers
    # ------------------------------------------------------------------

    def correlation_matrix(self) -> torch.Tensor:
        """Return the smooth ring correlation ``C`` (unit diagonal), ``(N, N)``."""
        return torch.from_numpy(self._C).float()

    def forcing_covariance(self) -> torch.Tensor:
        """Exact one-step forcing covariance ``c^2 * dt * C`` (see module helper)."""
        return l96c_forcing_cov(
            n_sites=self.n_sites, c=self.c, dt=self.dt, ell=self.ell
        )

    def _l96_rhs(self, x: np.ndarray) -> np.ndarray:
        """Vectorised Lorenz-96 RHS (periodic/ring boundaries via ``np.roll``).

        Acts on the last axis, so ``x`` may be ``(n_sites,)`` or
        ``(..., n_sites)`` (batched). Mirrors ``Lorenz96._l96_rhs``:
            roll(x, -1)[i] = x[(i+1) % N]   (X_{i+1})
            roll(x,  1)[i] = x[(i-1) % N]   (X_{i-1})
            roll(x,  2)[i] = x[(i-2) % N]   (X_{i-2})
        """
        return (
            (np.roll(x, -1, -1) - np.roll(x, 2, -1)) * np.roll(x, 1, -1)
            - x
            + self.forcing
        )

    def _step(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """One recorded step = ``n_substeps`` sub-stepped Euler-Maruyama updates.

        The reference sub-step update: each sub-step draws fresh
        i.i.d. ``eta`` and applies the correlated forcing
        ``c * sqrt(dt_sub) * (eta @ L^T)`` (so ``Cov = c^2 * dt_sub * C`` per
        sub-step, accumulating to ``c^2 * dt * C`` per recorded step). Acts on
        the last axis so ``x`` may be ``(n_sites,)`` or ``(batch, n_sites)``.
        """
        for _ in range(self.n_substeps):
            eta = rng.standard_normal(x.shape)
            x = (
                x
                + self._l96_rhs(x) * self.dt_sub
                + self.c * np.sqrt(self.dt_sub) * (eta @ self._chol.T)
            )
        return x

    def _generate(self, n: int, seed: int | None) -> np.ndarray:
        """Generate ``n`` trajectories."""
        rng = np.random.default_rng(seed)
        x = self.forcing + self.ic_scale * rng.standard_normal((n, self.n_sites))
        for _ in range(self.burn):
            x = self._step(x, rng)
        traj = np.empty((n, self.n_steps, self.n_sites), dtype=np.float32)
        for t in range(self.n_steps):
            x = self._step(x, rng)
            traj[:, t] = x
        if not np.isfinite(traj).all():
            msg = "Lorenz96Correlated generation diverged"
            raise RuntimeError(msg)
        return traj

    # ------------------------------------------------------------------
    # Sampling helpers (forcing draws + plausible states)
    # ------------------------------------------------------------------

    def sample_state(self, seed: int | None = None) -> torch.Tensor:
        """Draw one plausible ring state ``(n_sites,)`` from the burned-in process."""
        traj = self._generate(1, seed)
        return torch.from_numpy(traj[0, -1]).float()

    def sample_noise(self, n: int, seed: int | None = None) -> torch.Tensor:
        """Draw ``n`` one-step forcing samples ``xi ~ N(0, c^2 * dt * C)``.

        Accumulates ``n_substeps`` independent correlated sub-step increments via
        the same ``c * sqrt(dt_sub) * (eta @ L^T)`` construction as the dynamics,
        so the empirical covariance matches :func:`l96c_forcing_cov`. The forcing
        is state-independent (additive), so no state argument is needed. Returns
        ``(n, n_sites)``.
        """
        rng = np.random.default_rng(seed)
        eps = np.zeros((n, self.n_sites), dtype=np.float64)
        for _ in range(self.n_substeps):
            eta = rng.standard_normal((n, self.n_sites))
            eps = eps + self.c * np.sqrt(self.dt_sub) * (eta @ self._chol.T)
        return torch.from_numpy(eps).float()

    # ------------------------------------------------------------------
    # Simulator interface
    # ------------------------------------------------------------------

    def _forward(self, x: TensorLike) -> TensorLike:
        """Integrate a single trajectory (one fresh process-noise path)."""
        if x.shape[0] != 1:
            msg = (
                f"Lorenz96Correlated._forward expects a single input, got {x.shape[0]}"
            )
            raise ValueError(msg)
        traj = self._generate(1, None)  # (1, n_steps, n_sites)
        return torch.from_numpy(traj.reshape(1, -1))

    def forward_samples_spatiotemporal(
        self,
        n: int,
        random_seed: int | None = None,
        ensure_exact_n: bool = False,  # noqa: ARG002
    ) -> dict:
        """Produce ``n`` correlated-L96 trajectories in the spatiotemporal layout.

        Parameters
        ----------
        n: int
            Number of trajectories to sample.
        random_seed: int, optional
            Seed for reproducible draws (initial perturbation + process noise).

        Returns
        -------
        dict
            ``data``: float32 ``(n, n_steps, n_sites, 1, 1)`` ring fields.
            ``constant_scalars``: ``zeros(n, 1)`` (the ring has no exposed
            scalar; kept for API parity with the diff1d twin).
            ``constant_fields``: ``None``.
        """
        traj = self._generate(n, random_seed)  # (n, n_steps, n_sites)
        data = torch.from_numpy(traj).reshape(n, self.n_steps, self.n_sites, 1, 1)
        return {
            "data": data.float(),
            "constant_scalars": torch.zeros(n, 1),
            "constant_fields": None,
        }

    def mc_reference(
        self,
        x_state: torch.Tensor,
        n_draws: int,
        n_steps: int,
        random_seed: int | None = None,
    ) -> torch.Tensor:
        """Sample-feed Monte-Carlo rollout from a fixed state -- the DGP oracle.

        Because l96c is chaotic/nonlinear there is no closed-form one-step
        predictive; the true predictive ensemble is direct forward simulation.
        All ``n_draws`` members start from the same ``x_state`` but advance with
        INDEPENDENT correlated-forcing draws (the identical sub-stepped
        integrator + Cholesky forcing as :meth:`forward_samples_spatiotemporal`),
        so the across-member spread is a faithful oracle. The spread grows
        rapidly with lead (chaotic divergence) and carries the cross-site forcing
        correlation.

        Parameters
        ----------
        x_state: torch.Tensor
            Initial state; any shape that flattens to ``(n_sites,)`` (e.g. the
            ``(n_sites,)`` vector or the ``(B=1, n_sites, 1)`` spatial form).
        n_draws: int
            Number of independent ensemble members.
        n_steps: int
            Number of recorded steps (leads) per member.
        random_seed: int, optional
            Seed for reproducible draws.

        Returns
        -------
        torch.Tensor
            Float32 tensor of shape ``(n_draws, n_steps, n_sites, 1, 1)``
            (member-first), matching the :class:`Lorenz96` /
            :class:`~autosim.experimental.simulations.correlated_diffusion_1d.CorrelatedDiffusion1D`
            oracle convention.
        """
        rng = np.random.default_rng(random_seed)
        x0 = (
            torch.as_tensor(x_state)
            .detach()
            .cpu()
            .numpy()
            .reshape(self.n_sites)
            .astype(np.float64)
        )
        xm = np.repeat(x0[None, :], n_draws, axis=0)  # (n_draws, n_sites)
        out = np.empty((n_draws, n_steps, self.n_sites), dtype=np.float32)
        for t in range(n_steps):
            xm = self._step(xm, rng)
            out[:, t] = xm
        return torch.from_numpy(out).reshape(n_draws, n_steps, self.n_sites, 1, 1)

    def mc_oracle_rollout(
        self,
        x0: torch.Tensor,
        n_leads: int,
        n_members: int,
        *,
        seed: int | None = None,
    ) -> torch.Tensor:
        """Member-last view of the MC oracle (:meth:`mc_reference`).

        Identical ensemble to :meth:`mc_reference` (same integrator, same
        per-member independent correlated forcing) but returned in the
        member-last layout some downstream rollout code expects.

        Parameters
        ----------
        x0: torch.Tensor
            Initial state; any shape flattening to ``(n_sites,)``.
        n_leads: int
            Number of recorded steps (leads).
        n_members: int
            Number of independent ensemble members ``M``.
        seed: int, optional
            Seed for reproducible draws (shares draws with
            :meth:`mc_reference` at the same ``seed``).

        Returns
        -------
        torch.Tensor
            Float32 tensor of shape ``(B=1, n_leads, n_sites, 1, n_members)``.
        """
        ref = self.mc_reference(
            x0, n_draws=n_members, n_steps=n_leads, random_seed=seed
        )
        # (M, L, N, 1, 1) -> (1, L, N, 1, M)
        return ref.squeeze(-1).permute(1, 2, 3, 0).unsqueeze(0)
