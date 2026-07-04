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
however known exactly (:meth:`Lorenz96Correlated.forcing_covariance`).

Twin of :class:`Lorenz96` (dynamics, state layout, chaos); shares its generation
scaffolding with the correlated-diffusion toys via
:class:`~autosim.experimental.simulations._stochastic_base.CorrelatedFieldSimulator`.
"""

from __future__ import annotations

import numpy as np
import torch

from ._stochastic_base import CorrelatedFieldSimulator


def _ring_correlation(n_sites: int, ell: float) -> np.ndarray:
    """Smooth circulant ring correlation ``C`` with UNIT diagonal.

    Uses a fixed reference discretisation: ``C_ij = exp(-d_ij^2 / (2 ell^2))`` with
    ``d_ij`` the periodic (ring) distance between sites ``i`` and ``j``. The
    diagonal is ``1`` so ``C`` is a correlation matrix, not a covariance.

    Args:
        n_sites: Number of ring sites ``N``.
        ell: Correlation length in sites (sets the off-diagonal decay).

    Returns:
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
    n_sites: int,
    c: float,
    dt: float,
    ell: float,
) -> torch.Tensor:
    """Exact one-step (recorded-step) forcing covariance ``c^2 * dt * C``.

    The forcing accumulated over one recorded step has covariance
    ``c^2 * dt * C``: the diagonal is the white-l96 per-site variance
    ``c^2 * dt`` (so the per-site marginals tie with white l96) and the
    off-diagonal carries the added ring correlation ``C``. This is the forcing
    covariance only -- l96c has no closed-form *predictive* covariance because
    the dynamics are chaotic (use :meth:`Lorenz96Correlated.mc_reference`).

    All parameters are required (no defaults) so a non-default instance and its
    oracle cannot silently desync; :meth:`Lorenz96Correlated.forcing_covariance`
    forwards the instance's own parameters.

    Args:
        n_sites: Number of ring sites ``N``.
        c: Diffusion coefficient (noise amplitude).
        dt: Recorded-step size.
        ell: Ring correlation length in sites.

    Returns:
        The ``(n_sites, n_sites)`` forcing covariance, float32.
    """
    c_mat = _ring_correlation(n_sites, ell)
    return torch.from_numpy((c**2) * dt * c_mat).float()


class Lorenz96Correlated(CorrelatedFieldSimulator):
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
    cross-site correlation is added (see :meth:`forcing_covariance`).

    Args:
        parameters_range: Placeholder bound for the (unused) sampled input;
            defaults to ``{"x0": (-1.0, 1.0)}`` for API parity with the diff1d
            twin.
        output_names: Channel name; defaults to ``["x"]``.
        log_level: Logging verbosity passed to the base ``Simulator``.
        n_steps: Number of recorded Euler-Maruyama steps per trajectory (``T``).
        burn: Burn-in steps discarded before recording.
        n_sites: Number of ring sites ``N``.
        forcing: Lorenz-96 forcing constant ``F`` (8.0 is the chaotic regime).
        c: Diffusion coefficient (noise amplitude).
        dt: Recorded-step size; EM stability is provided by the sub-stepping.
        ell: Ring correlation length in sites (``ell = 2`` gives nearest-neighbour
            correlation ~0.88).
        n_substeps: Number of Euler-Maruyama sub-steps per recorded step, each
            advancing ``dt / n_substeps``.
        ic_scale: Standard deviation of the Gaussian IC perturbation off the
            F-uniform fixed point (``F + ic_scale * randn``).
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
        """Initialize the correlated Lorenz-96 integrator and validate parameters."""
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
        self._field_shape = (n_sites,)
        self.forcing = forcing
        self.c = c
        self.dt = dt
        self.ell = ell
        self.n_substeps = n_substeps
        self.ic_scale = ic_scale
        self.dt_sub = dt / n_substeps

        # Cholesky factor of the ring correlation, with a 1e-9 jitter for
        # positive-definiteness. C has unit diagonal -> the per-site forcing
        # variance is exactly the white-l96 value c^2 * dt after sub-step
        # accumulation.
        self._C: np.ndarray = _ring_correlation(n_sites, ell)
        self._chol: np.ndarray = np.linalg.cholesky(self._C + 1e-9 * np.eye(n_sites))

    def correlation_matrix(self) -> torch.Tensor:
        """Return the smooth ring correlation ``C`` (unit diagonal), ``(N, N)``."""
        return torch.from_numpy(self._C).float()

    def forcing_covariance(self) -> torch.Tensor:
        """Exact one-step forcing covariance ``c^2 * dt * C`` for this instance."""
        return l96c_forcing_cov(
            n_sites=self.n_sites, c=self.c, dt=self.dt, ell=self.ell
        )

    def _l96_rhs(self, x: np.ndarray) -> np.ndarray:
        """Vectorised Lorenz-96 RHS (periodic/ring boundaries via ``np.roll``).

        Acts on the last axis, so ``x`` may be ``(n_sites,)`` or
        ``(..., n_sites)`` (batched)::

            roll(x, -1)[i] = x[(i+1) % N]   (X_{i+1})
            roll(x,  1)[i] = x[(i-1) % N]   (X_{i-1})
            roll(x,  2)[i] = x[(i-2) % N]   (X_{i-2})

        Args:
            x: State array with sites on the last axis.

        Returns:
            The RHS, same shape as ``x``.
        """
        return (
            (np.roll(x, -1, -1) - np.roll(x, 2, -1)) * np.roll(x, 1, -1)
            - x
            + self.forcing
        )

    def _step(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """One recorded step = ``n_substeps`` sub-stepped Euler-Maruyama updates.

        Each sub-step draws fresh i.i.d. ``eta`` and applies the correlated
        forcing ``c * sqrt(dt_sub) * (eta @ L^T)`` (so ``Cov = c^2 * dt_sub * C``
        per sub-step, accumulating to ``c^2 * dt * C`` per recorded step). Acts on
        the last axis so ``x`` may be ``(n_sites,)`` or ``(batch, n_sites)``.

        Args:
            x: State batch with sites on the last axis.
            rng: Random generator supplying the correlated forcing draws.

        Returns:
            The updated state batch, same shape as ``x``.
        """
        for _ in range(self.n_substeps):
            eta = rng.standard_normal(x.shape)
            x = (
                x
                + self._l96_rhs(x) * self.dt_sub
                + self.c * np.sqrt(self.dt_sub) * (eta @ self._chol.T)
            )
        return x

    def _initial_state(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Gaussian perturbation off the F-uniform fixed point, ``(n, n_sites)``."""
        return self.forcing + self.ic_scale * rng.standard_normal((n, self.n_sites))

    def sample_noise(self, n: int, seed: int | None = None) -> torch.Tensor:
        """Draw ``n`` one-step forcing samples ``xi ~ N(0, c^2 * dt * C)``.

        Accumulates ``n_substeps`` independent correlated sub-step increments via
        the same ``c * sqrt(dt_sub) * (eta @ L^T)`` construction as the dynamics,
        so the empirical covariance matches :meth:`forcing_covariance`. The forcing
        is state-independent (additive), so no state argument is needed.

        Args:
            n: Number of forcing samples.
            seed: Seed for reproducible draws.

        Returns:
            Float32 tensor of shape ``(n, n_sites)``.
        """
        rng = np.random.default_rng(seed)
        eps = np.zeros((n, self.n_sites), dtype=np.float64)
        for _ in range(self.n_substeps):
            eta = rng.standard_normal((n, self.n_sites))
            eps = eps + self.c * np.sqrt(self.dt_sub) * (eta @ self._chol.T)
        return torch.from_numpy(eps).float()

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

        Args:
            x0: Initial state; any shape flattening to ``(n_sites,)``.
            n_leads: Number of recorded steps (leads).
            n_members: Number of independent ensemble members ``M``.
            seed: Seed for reproducible draws (shares draws with
                :meth:`mc_reference` at the same ``seed``).

        Returns:
            Float32 tensor of shape ``(1, n_leads, n_sites, 1, n_members)``.
        """
        ref = self.mc_reference(
            x0, n_draws=n_members, n_steps=n_leads, random_seed=seed
        )
        # (M, L, N, 1, 1) -> (1, L, N, 1, M)
        return ref.squeeze(-1).permute(1, 2, 3, 0).unsqueeze(0)
