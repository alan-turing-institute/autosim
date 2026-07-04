"""Latent multivariate-OU process viewed through a fixed nonlinear lens.

A stationary vector AR(1) latent ``z_{t+1} = A z_t + eps``,
``eps ~ N(0, Sigma_z)`` with a known low-rank-plus-diagonal noise covariance
``Sigma_z = diag(D_z) + U_z U_zᵀ``, observed through a fixed analytic lens
``x_t = g(z_t) in R^{d_x}`` with ``d_x > d_z``. The lens gives the ambient field
a genuinely low-dimensional nonlinear manifold (intrinsic dim ``d_z``) with
heavy, tunable tails — the one property the four scalar/ambient toys lack — while
keeping the predictive law tractable: the ``k``-step latent covariance is the
closed-form Lyapunov recursion :func:`latent_var`, and the ambient oracle is the
Monte-Carlo push-forward of latent OU draws through ``g`` (:meth:`mc_reference`).

Design + verified math:
``notes/research/uncertainty/latent_lens_toy_design_2026-06-08/`` (the Lyapunov
recursion, the ``sinh(0.7 z)`` -> kurtosis ~7.7 tail knob, and oracle calibration
are checked numerically in ``verify_claims.py``).
"""

from __future__ import annotations

import numpy as np
import torch

from autosim.simulations.base import SpatioTemporalSimulator
from autosim.types import TensorLike


def latent_var(
    leads: torch.Tensor,
    A: TensorLike,
    Sigma_z: TensorLike,
) -> torch.Tensor:
    """Closed-form ``k``-step latent predictive covariance (Lyapunov recursion).

    For the latent AR(1) ``z_{t+1} = A z_t + eps``, ``eps ~ N(0, Sigma_z)``, the
    covariance of ``z_{t+k}`` given ``z_t`` is

        ``V_k = sum_{j=0}^{k-1} A^j Sigma_z (A^j)ᵀ``,

    which satisfies the recursion ``V_k = A V_{k-1} Aᵀ + Sigma_z`` with
    ``V_0 = 0``. The mean path is deterministic (``A^k z_t``), so this covariance
    fully specifies the latent Gaussian predictive. Analogous to OU's
    ``ou_closed_form_var`` but matrix-valued.

    Args:
        leads: Integer lead steps (1-based), e.g.
            ``torch.arange(1, n_steps + 1)``.
        A: Transition matrix, shape ``(d_z, d_z)``.
        Sigma_z: One-step noise covariance, shape ``(d_z, d_z)``.

    Returns:
        Covariance at each requested lead, shape ``(len(leads), d_z, d_z)``.
    """
    a = torch.as_tensor(A, dtype=torch.float64)
    sigma = torch.as_tensor(Sigma_z, dtype=torch.float64)
    lead_list = [int(k) for k in torch.as_tensor(leads).reshape(-1).tolist()]
    if min(lead_list) < 1:
        msg = f"latent_var expects 1-based positive lead steps; got {lead_list}."
        raise ValueError(msg)

    d_z = a.shape[0]
    wanted = set(lead_list)
    collected: dict[int, torch.Tensor] = {}
    v = torch.zeros((d_z, d_z), dtype=torch.float64)
    for k in range(1, max(lead_list) + 1):
        v = a @ v @ a.T + sigma
        if k in wanted:
            collected[k] = v.clone()
    return torch.stack([collected[k] for k in lead_list])


def _build_cross_mixing(d_z: int, d_x: int, seed: int) -> np.ndarray:
    """Build the fixed mixing for the bilinear cross-term ambient coordinates.

    The first ``d_z`` ambient coordinates are the pure ``sinh`` marginals; the
    remaining ``d_x - d_z`` are fixed linear combinations of the ``d_z(d_z+1)/2``
    distinct second-order monomials ``z_a z_b`` (``a <= b``). The mixing is a
    fixed seeded matrix scaled to unit row norm, so the cross coordinates are
    smooth, deterministic, and comparable in scale — a "fixed smooth mixing"
    (not a trainable net). Returns shape ``(d_x - d_z, d_z(d_z+1)/2)``.
    """
    n_pairs = d_z * (d_z + 1) // 2
    n_cross = d_x - d_z
    if n_cross <= 0:
        return np.zeros((0, n_pairs), dtype=np.float64)
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((n_cross, n_pairs))
    # Unit row norm keeps each cross coordinate O(1) regardless of d_z.
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    return m


class MultivariateOULens(SpatioTemporalSimulator):
    r"""Latent multivariate-OU process observed through a fixed nonlinear lens.

    Dynamics (latent): ``z_{t+1} = A z_t + eps``, ``eps ~ N(0, Sigma_z)`` with
    ``Sigma_z = diag(D_z) + U_z U_zᵀ`` (a known rank-``r_star`` correlation).
    ``A`` is contractive (``rho(A) < 1``), so the process is stationary and the
    spread saturates at the discrete Lyapunov fixed point.

    Observation (lens ``g: R^{d_z} -> R^{d_x}``, ``d_x > d_z``), fixed and known:

    - the first ``d_z`` coordinates are ``sinh(tail_alpha * z_i)`` — symmetric
      heavy tails with a kurtosis knob (``tail_alpha ~ 0.7`` -> kurtosis ~7.7,
      matching Lorenz-96);
    - the remaining ``d_x - d_z`` coordinates are a fixed smooth mixing of the
      second-order monomials ``z_a z_b`` (genuine nonlinear cross terms).

    The ambient field is returned reshaped to ``obs_shape`` (default a square
    single-channel field, e.g. ``d_x = 16`` -> ``4 x 4 x 1``).

    Heavy-tail negative control: set ``latent_noise="student_t"`` to draw ``eps``
    from a Student-t (scaled to ``Sigma_z``) so the heavy tails live in the
    latent process rather than the lens — the falsifiable contrast to the
    Gaussian-latent main construction.

    Args:
        A: Latent transition ``(d_z, d_z)``. Defaults to a ``d_z = 4``
            contractive instance.
        D_z: Noise diagonal ``(d_z,)``. Defaults to a ``d_z = 4`` contractive
            instance.
        U_z: Low-rank noise factor ``(d_z, r_star)``. Defaults to a
            ``d_z = 4``, ``r_star = 2`` contractive instance.
        obs_dim: Ambient dimension ``d_x`` (default 16).
        obs_shape: Ambient ``(H, W, C)`` reshape of the ``d_x`` vector (default
            a square single-channel field; must multiply to ``obs_dim``).
        tail_alpha: ``sinh`` tail-weight knob (default 0.7).
        mixing_seed: Seed for the fixed cross-term mixing (default 0).
        latent_noise: ``"gaussian"`` (default) or ``"student_t"`` (the
            heavy-tailed-latent negative control).
        student_t_dof: Degrees of freedom when ``latent_noise="student_t"``
            (default 4).
        ic_bound: Half-width of the uniform box the initial latent ``z_0`` is
            drawn from.
        n_steps: Number of latent steps recorded per trajectory (default 64).
    """

    def __init__(
        self,
        A: TensorLike | None = None,
        D_z: TensorLike | None = None,
        U_z: TensorLike | None = None,
        obs_dim: int = 16,
        obs_shape: tuple[int, int, int] | None = None,
        tail_alpha: float = 0.7,
        mixing_seed: int = 0,
        latent_noise: str = "gaussian",
        student_t_dof: float = 4.0,
        ic_bound: float = 1.5,
        parameters_range: dict[str, tuple[float, float]] | None = None,
        output_names: list[str] | None = None,
        log_level: str = "error",
        n_steps: int = 64,
    ) -> None:
        """Initialize the multivariate-OU lens and validate its matrices."""
        a = _default_transition() if A is None else np.asarray(A, dtype=np.float64)
        d_z = a.shape[0]
        d_diag = (
            _default_diag(d_z) if D_z is None else np.asarray(D_z, dtype=np.float64)
        )
        u = _default_lowrank(d_z) if U_z is None else np.asarray(U_z, dtype=np.float64)

        self._validate_dynamics(a, d_diag, u, d_z)
        sigma_z = np.diag(d_diag) + u @ u.T
        # Sigma_z must be strictly PD for the Cholesky sampler below; a zero D_z
        # entry with a rank-deficient U_z column span is only PSD (its zero
        # eigenvalue shows up as ~1e-17 in floating point, so compare against a
        # scale-relative tolerance). Fail early with a clear message rather than
        # an opaque LinAlgError.
        eigs = np.linalg.eigvalsh(sigma_z)
        min_eig = float(eigs.min())
        if min_eig <= 1e-10 * float(eigs.max()):
            msg = (
                "MultivariateOULens requires a positive-definite noise covariance "
                "Sigma_z = diag(D_z) + U_z U_zᵀ; got smallest eigenvalue "
                f"{min_eig:.3e} (a zero D_z entry not covered by a U_z "
                "direction). Use strictly positive D_z entries."
            )
            raise ValueError(msg)

        if obs_dim <= d_z:
            msg = f"MultivariateOULens requires obs_dim > d_z; got {obs_dim} <= {d_z}."
            raise ValueError(msg)
        if obs_shape is None:
            obs_shape = _square_obs_shape(obs_dim)
        if int(np.prod(obs_shape)) != obs_dim:
            msg = (
                f"MultivariateOULens obs_shape {obs_shape} must multiply to "
                f"obs_dim={obs_dim}."
            )
            raise ValueError(msg)
        if latent_noise not in ("gaussian", "student_t"):
            msg = (
                "MultivariateOULens latent_noise must be 'gaussian' or "
                f"'student_t'; got {latent_noise!r}."
            )
            raise ValueError(msg)
        # A Student-t has finite (and rescalable-to-Sigma_z) covariance only for
        # dof > 2; at dof <= 2 the scaling factor zeros or NaNs the noise.
        if latent_noise == "student_t" and student_t_dof <= 2.0:
            msg = (
                "MultivariateOULens latent_noise='student_t' requires "
                f"student_t_dof > 2 for a finite covariance; got {student_t_dof}."
            )
            raise ValueError(msg)

        if parameters_range is None:
            parameters_range = {f"z0_{i}": (-ic_bound, ic_bound) for i in range(d_z)}
        if output_names is None:
            output_names = [f"x{i}" for i in range(obs_dim)]
        super().__init__(parameters_range, output_names, log_level)

        self.d_z = d_z
        self.obs_dim = obs_dim
        self.obs_shape = tuple(obs_shape)
        self.A = a
        self.D_z = d_diag
        self.U_z = u
        self.Sigma_z = sigma_z
        self._chol_sigma = np.linalg.cholesky(sigma_z)
        self.tail_alpha = tail_alpha
        self.mixing_seed = mixing_seed
        self.latent_noise = latent_noise
        self.student_t_dof = student_t_dof
        self.n_steps = n_steps
        self._cross_mixing = _build_cross_mixing(d_z, obs_dim, mixing_seed)

    @staticmethod
    def _validate_dynamics(
        a: np.ndarray, d_diag: np.ndarray, u: np.ndarray, d_z: int
    ) -> None:
        """Fail-early on a non-stationary or ill-formed latent process."""
        if a.shape != (d_z, d_z):
            msg = f"MultivariateOULens A must be ({d_z}, {d_z}); got {a.shape}."
            raise ValueError(msg)
        rho = float(np.max(np.abs(np.linalg.eigvals(a))))
        if rho >= 1.0:
            msg = (
                "MultivariateOULens requires a contractive transition "
                f"(spectral radius rho(A) < 1) for a stationary process; got "
                f"rho(A) = {rho:.4f}."
            )
            raise ValueError(msg)
        if d_diag.shape != (d_z,) or np.any(d_diag < 0):
            msg = (
                f"MultivariateOULens D_z must be a non-negative ({d_z},) "
                f"diagonal; got shape {d_diag.shape}."
            )
            raise ValueError(msg)
        if u.ndim != 2 or u.shape[0] != d_z:
            msg = f"MultivariateOULens U_z must be ({d_z}, r_star); got {u.shape}."
            raise ValueError(msg)

    def lens(self, z: np.ndarray) -> np.ndarray:
        """Apply the fixed nonlinear lens ``g: R^{d_z} -> R^{d_x}``.

        ``z`` has shape ``(..., d_z)``; returns ``(..., d_x)``. The first
        ``d_z`` outputs are ``sinh(tail_alpha * z_i)`` (heavy-tailed marginals);
        the rest are the fixed smooth mixing of the second-order monomials.
        """
        z = np.asarray(z, dtype=np.float64)
        sinh_part = np.sinh(self.tail_alpha * z)  # (..., d_z)
        if self._cross_mixing.shape[0] == 0:
            return sinh_part
        # Distinct second-order monomials z_a z_b (a <= b), in a fixed order.
        idx_a, idx_b = np.triu_indices(self.d_z)
        monomials = z[..., idx_a] * z[..., idx_b]  # (..., n_pairs)
        cross = monomials @ self._cross_mixing.T  # (..., d_x - d_z)
        return np.concatenate([sinh_part, cross], axis=-1)

    def _draw_noise(self, rng: np.random.Generator) -> np.ndarray:
        """One latent noise draw ``eps`` with covariance ``Sigma_z``."""
        std = rng.standard_normal(self.d_z)
        if self.latent_noise == "student_t":
            # Scale a standard normal by sqrt(dof / chi2_dof) -> multivariate t
            # with covariance dof/(dof-2) * Sigma_z; renormalise so the second
            # moment matches Sigma_z exactly (heavy tails, same scale).
            dof = self.student_t_dof
            g = rng.chisquare(dof) / dof
            std = std / np.sqrt(g)
            std = std * np.sqrt((dof - 2.0) / dof)
        return self._chol_sigma @ std

    def _roll_latent(
        self, z0: np.ndarray, rng: np.random.Generator, n_steps: int | None = None
    ) -> np.ndarray:
        """Integrate one latent OU trajectory from ``z0``.

        Args:
            z0: Initial latent state, shape ``(d_z,)``.
            rng: Random generator supplying the process noise.
            n_steps: Number of steps to record; defaults to ``self.n_steps``.

        Returns:
            Latent trajectory of shape ``(n_steps, d_z)``.
        """
        steps = self.n_steps if n_steps is None else n_steps
        z = np.asarray(z0, dtype=np.float64).reshape(self.d_z)
        traj = np.empty((steps, self.d_z), dtype=np.float64)
        for t in range(steps):
            z = self.A @ z + self._draw_noise(rng)
            traj[t] = z
        return traj

    def _forward(self, x: TensorLike) -> TensorLike:
        """Integrate one latent trajectory and observe it through the lens.

        ``x`` is the sampled initial latent ``z_0`` of shape ``(1, d_z)``;
        returns the flattened ambient trajectory ``(1, n_steps * d_x)``.
        """
        if x.shape[0] != 1:
            msg = (
                f"MultivariateOULens._forward expects a single input, got {x.shape[0]}"
            )
            raise ValueError(msg)
        rng = np.random.default_rng()  # fresh process-noise path per trajectory
        z0 = x.cpu().numpy().reshape(self.d_z)
        latent_traj = self._roll_latent(z0, rng)  # (n_steps, d_z)
        ambient = self.lens(latent_traj).astype(np.float32)  # (n_steps, d_x)
        return torch.from_numpy(ambient.reshape(1, -1))

    def forward_samples_spatiotemporal(
        self,
        n: int,
        random_seed: int | None = None,
        ensure_exact_n: bool = False,  # noqa: ARG002 -- generation is always exact
    ) -> dict:
        """Sample ``n`` lens trajectories and the true initial latents.

        Both the sampled initial latents and the process noise are seeded from
        ``random_seed``, so a given seed reproduces the full batch. This
        integrator never fails, so ``ensure_exact_n`` is always satisfied
        without retries.

        Returns:
            A dict with ``data`` (float32 ``(n, n_steps, H, W, C)`` ambient
            fields), ``constant_scalars`` (the sampled initial latents
            ``z_0``, ``(n, d_z)``), ``constant_fields`` (``None``, API parity
            with the OU twin) and ``latent_states`` (the same ``z_0`` exposed
            for the oracle arm, ``(n, d_z)``).
        """
        x = self.sample_inputs(n, random_seed)
        rng = np.random.default_rng(random_seed)  # one shared process-noise stream
        h, w, c = self.obs_shape
        ambient = np.stack(
            [
                self.lens(self._roll_latent(x[i].cpu().numpy().reshape(self.d_z), rng))
                for i in range(n)
            ]
        ).astype(np.float32)
        data = torch.from_numpy(ambient).reshape(n, self.n_steps, h, w, c)
        return {
            "data": data,
            "constant_scalars": x,
            "constant_fields": None,
            "latent_states": x,
        }

    def mc_reference(
        self,
        x_state: torch.Tensor,
        n_draws: int,
        n_steps: int,
        random_seed: int | None = None,
    ) -> torch.Tensor:
        """Oracle ambient ensemble: latent OU draws pushed through the lens.

        From a fixed initial **latent** state ``x_state`` (``z_0``; the oracle
        knows the true latent exposed by
        :meth:`forward_samples_spatiotemporal`), draw ``n_draws`` independent
        latent OU trajectories and observe each through ``g``. The empirical
        distribution of these ambient draws is the calibrated oracle for the
        nonlinear lens (a scalar closed-form variance is exact only for a linear
        ``g``; here the MC-through-``g`` ensemble is the correct oracle).

        Returns float32 ``(n_draws, n_steps, H, W, C)``.
        """
        rng = np.random.default_rng(random_seed)
        z0 = torch.as_tensor(x_state).detach().cpu().numpy().reshape(-1)[: self.d_z]
        h, w, c = self.obs_shape
        out = np.empty((n_draws, n_steps, self.obs_dim), dtype=np.float32)
        for d in range(n_draws):
            latent_traj = self._roll_latent(z0, rng, n_steps)
            out[d] = self.lens(latent_traj).astype(np.float32)
        return torch.from_numpy(out).reshape(n_draws, n_steps, h, w, c)


def _default_transition() -> np.ndarray:
    """Contractive 4x4 upper-bidiagonal transition (rho = 0.90)."""
    a = np.diag([0.90, 0.88, 0.86, 0.84])
    a[0, 1] = a[1, 2] = a[2, 3] = 0.05
    return a


def _default_diag(d_z: int) -> np.ndarray:
    """Return the default noise diagonal D_z (length d_z)."""
    base = np.array([0.10, 0.15, 0.20, 0.12])
    if d_z <= base.size:
        return base[:d_z].copy()
    return np.full(d_z, 0.12)


def _default_lowrank(d_z: int) -> np.ndarray:
    """Return the default rank-2 noise factor U_z (shape (d_z, 2))."""
    u = np.zeros((d_z, 2))
    half = d_z // 2
    u[:half, 0] = 0.35
    u[half:, 1] = 0.30
    return u


def _square_obs_shape(obs_dim: int) -> tuple[int, int, int]:
    """Reshape d_x to a square single-channel (H, W, 1) field when possible."""
    side = round(obs_dim**0.5)
    if side * side == obs_dim:
        return (side, side, 1)
    return (obs_dim, 1, 1)
