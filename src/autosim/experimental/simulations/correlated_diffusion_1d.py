"""Correlated-diffusion ring (diff1d): the correlation-structure preservation toy.

A line of ``N`` points on a ring evolves by a stable, linear, damped diffusion
(the conditional mean is linear and trivially learnable -- no chaos), driven by
Gaussian forcing whose covariance is genuinely low-rank-plus-diagonal and
state-dependent:

    x_{t+1} = A x_t + eps_t ,    A = (1 - dt*gamma) I + dt*kappa * Lap1D   (stable)
    eps_t   ~ N(0, Sigma(x_t)) , Sigma(x_t) = G diag(sigma2(x_t)) Gᵀ + delta^2 I
    sigma2_i(x_t) = sigma0^2 (1 + beta * |grad x_t|_i)   (forcing at fronts)

``G`` is a fixed Gaussian-ring low-pass smoother: white forcing is scaled by the
local std then smoothed, so the off-diagonal correlation emerges from the
smoothing length ``ell`` and is approximately low-rank; ``delta^2 I`` is a
diagonal nugget the diagonal head can fit. Low-rank wins only via the smoothed
off-diagonal, and only on collection (regional-sum) coverage where the per-site
marginals tie.

Because ``Sigma(x)`` is known exactly, the oracle predictive ``N(A x, Sigma(x))``
is closed form (:func:`diff1d_closed_form_cov`). The dynamics follow a fixed
reference discretisation, giving a process with a known correlation structure.

Twin of :class:`OrnsteinUhlenbeck`: a :class:`SpatioTemporalSimulator` subclass
paired with a module-level closed-form covariance helper.
"""

from __future__ import annotations

import numpy as np
import torch

from autosim.simulations.base import SpatioTemporalSimulator
from autosim.types import TensorLike


def _ring_operators(
    n_sites: int, dt: float, gamma: float, kappa: float, ell: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return the Fourier mean operator ``A_hat`` and smoother ``G_hat``.

    Uses a fixed reference discretisation: a 3-point Laplacian on the ring and a
    Gaussian low-pass (DC = 1).
    """
    q = np.fft.fftfreq(n_sites) * 2.0 * np.pi  # angular wavenumbers
    lap = -4.0 * np.sin(q / 2.0) ** 2  # 3-point Laplacian eigenvalues, [-4, 0]
    a_hat = (1.0 - dt * gamma) + dt * kappa * lap  # mean operator A (per mode)
    g_hat = np.exp(-0.5 * ell**2 * q**2)  # Gaussian low-pass (DC = 1)
    return a_hat, g_hat


def _smoother_matrix(g_hat: np.ndarray) -> np.ndarray:
    """Build the explicit (symmetric) smoother matrix ``G`` from ``G_hat``.

    Applies ``apply_G`` to each unit impulse, then symmetrises (the kernel is
    symmetric up to floating-point).
    """
    n = g_hat.shape[0]
    g = np.zeros((n, n))
    for p in range(n):
        e = np.zeros(n)
        e[p] = 1.0
        g[:, p] = np.real(np.fft.ifft(g_hat * np.fft.fft(e)))
    return 0.5 * (g + g.T)


def diff1d_closed_form_cov(
    x: TensorLike,
    *,
    n_sites: int = 40,
    dt: float = 1.0,
    gamma: float = 0.2,
    kappa: float = 0.1,
    ell: float = 3.0,
    sigma0: float = 0.35,
    beta: float = 1.5,
    delta: float = 0.12,
) -> torch.Tensor:
    """Exact one-step covariance ``Sigma(x) = G diag(sigma2(x)) Gᵀ + delta^2 I``.

    Parameters
    ----------
    x: TensorLike
        A single ring state, any shape flattening to ``(n_sites,)``.
    n_sites, dt, gamma, kappa, ell, sigma0, beta, delta:
        Ring DGP parameters; defaults match
        :class:`CorrelatedDiffusion1D`.

    Returns
    -------
    torch.Tensor
        The ``(n_sites, n_sites)`` covariance, float32.
    """
    x_np = torch.as_tensor(x).detach().cpu().numpy().reshape(n_sites).astype(np.float64)
    _, g_hat = _ring_operators(n_sites, dt, gamma, kappa, ell)
    g_mat = _smoother_matrix(g_hat)
    grad_mag = np.abs((np.roll(x_np, -1) - np.roll(x_np, 1)) * 0.5)
    s2 = sigma0**2 * (1.0 + beta * grad_mag)
    cov = g_mat @ (s2[:, None] * g_mat.T) + delta**2 * np.eye(n_sites)
    return torch.from_numpy(cov).float()


class CorrelatedDiffusion1D(SpatioTemporalSimulator):
    r"""Stable 1-D ring diffusion with a known low-rank-plus-diagonal forcing.

    The conditional mean is the linear, contractive operator
    ``A = (1 - dt*gamma) I + dt*kappa * Lap1D`` (no chaos); the forcing is
    ``eps_t ~ N(0, Sigma(x_t))`` with ``Sigma(x_t) = G diag(sigma2(x_t)) Gᵀ +
    delta^2 I`` and ``sigma2_i = sigma0^2 (1 + beta |grad x|_i)``. The mean
    operator's spectral radius is checked ``< 1`` at construction (fail-early).

    Parameters
    ----------
    parameters_range: dict[str, tuple[float, float]], optional
        Placeholder bound for the (unused) sampled input; defaults to
        ``{"x0": (-1.0, 1.0)}`` for API parity with the OU twin. Trajectories
        are produced directly by :meth:`forward_samples_spatiotemporal` from a
        seeded burn-in, not from this input.
    output_names: list[str], optional
        Channel name; defaults to ``["x"]``.
    log_level: str, default="error"
        Logging verbosity passed to the base ``Simulator``.
    n_steps: int, default=128
        Number of recorded steps per trajectory (``T``).
    burn: int, default=200
        Burn-in steps discarded before recording.
    n_sites: int, default=40
        Ring sites (``N``).
    dt, gamma, kappa: float
        Damped-diffusion mean operator parameters (``1.0, 0.2, 0.1``).
    ell: float, default=3.0
        Gaussian smoothing length in sites (sets the effective rank).
    sigma0: float, default=0.35
        Base forcing standard deviation.
    beta: float, default=1.5
        Gradient/front sensitivity of the forcing amplitude.
    delta: float, default=0.12
        Diagonal nugget standard deviation.
    """

    def __init__(
        self,
        parameters_range: dict[str, tuple[float, float]] | None = None,
        output_names: list[str] | None = None,
        log_level: str = "error",
        n_steps: int = 128,
        burn: int = 200,
        n_sites: int = 40,
        dt: float = 1.0,
        gamma: float = 0.2,
        kappa: float = 0.1,
        ell: float = 3.0,
        sigma0: float = 0.35,
        beta: float = 1.5,
        delta: float = 0.12,
    ) -> None:
        if parameters_range is None:
            parameters_range = {"x0": (-1.0, 1.0)}
        if output_names is None:
            output_names = ["x"]

        self.a_hat, self.g_hat = _ring_operators(n_sites, dt, gamma, kappa, ell)
        self._validate_dynamics(self.a_hat)

        super().__init__(parameters_range, output_names, log_level)

        self.n_steps = n_steps
        self.burn = burn
        self.n_sites = n_sites
        self.dt = dt
        self.gamma = gamma
        self.kappa = kappa
        self.ell = ell
        self.sigma0 = sigma0
        self.beta = beta
        self.delta = delta

    @staticmethod
    def _validate_dynamics(a_hat: np.ndarray) -> None:
        """Fail-early on a non-contractive (unstable) mean operator."""
        sr = float(np.max(np.abs(a_hat)))
        if sr >= 1.0:
            msg = (
                "CorrelatedDiffusion1D requires a contractive mean operator "
                f"(spectral radius |A| < 1) for a stable process; got |A| = {sr:.4f}."
            )
            raise ValueError(msg)

    def A_spectral_radius(self) -> float:
        """Spectral radius of the linear mean operator ``A`` (``< 1`` when stable)."""
        return float(np.max(np.abs(self.a_hat)))

    def _apply_A(self, x: np.ndarray) -> np.ndarray:
        """Apply the damped-diffusion mean operator ``A`` (Fourier)."""
        return np.real(np.fft.ifft(self.a_hat * np.fft.fft(x, axis=-1), axis=-1))

    def _apply_G(self, v: np.ndarray) -> np.ndarray:
        """Apply the Gaussian low-pass smoother ``G`` (Fourier)."""
        return np.real(np.fft.ifft(self.g_hat * np.fft.fft(v, axis=-1), axis=-1))

    def _grad_mag(self, x: np.ndarray) -> np.ndarray:
        """Magnitude of the central-difference gradient on the ring."""
        return np.abs((np.roll(x, -1, -1) - np.roll(x, 1, -1)) * 0.5)

    def _sigma2(self, x: np.ndarray) -> np.ndarray:
        """State-dependent per-site forcing variance (front-concentrated)."""
        return self.sigma0**2 * (1.0 + self.beta * self._grad_mag(x))

    def _step(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """One DGP step ``A x + G(sqrt(sigma2) eta1) + delta eta2`` (raw space).

        The reference update step -- the two standard-normal
        draws are taken in the same order so trajectories are byte-identical.
        """
        mean = self._apply_A(x)
        std = np.sqrt(self._sigma2(x))
        eta = rng.standard_normal(x.shape)
        eta2 = rng.standard_normal(x.shape)
        eps = self._apply_G(std * eta) + self.delta * eta2
        return mean + eps

    def _generate(self, n: int, seed: int | None) -> np.ndarray:
        """Generate ``n`` trajectories."""
        rng = np.random.default_rng(seed)
        x = 0.1 * rng.standard_normal((n, self.n_sites))
        for _ in range(self.burn):
            x = self._step(x, rng)
        traj = np.empty((n, self.n_steps, self.n_sites), dtype=np.float32)
        for t in range(self.n_steps):
            x = self._step(x, rng)
            traj[:, t] = x
        if not np.isfinite(traj).all():
            msg = "CorrelatedDiffusion1D generation diverged"
            raise RuntimeError(msg)
        return traj

    def sample_state(self, seed: int | None = None) -> torch.Tensor:
        """Draw one plausible ring state ``(n_sites,)`` from the burned-in process."""
        traj = self._generate(1, seed)
        return torch.from_numpy(traj[0, -1]).float()

    def sample_noise(
        self, x: TensorLike, n: int, seed: int | None = None
    ) -> torch.Tensor:
        """Draw ``n`` forcing samples ``eps ~ N(0, Sigma(x))`` via the DGP construction.

        Uses the same ``G(sqrt(sigma2(x)) eta1) + delta eta2`` forcing as the
        dynamics (no Cholesky), so the empirical covariance matches
        :func:`diff1d_closed_form_cov`. Returns ``(n, n_sites)``.
        """
        x_np = torch.as_tensor(x).detach().cpu().numpy().reshape(self.n_sites)
        rng = np.random.default_rng(seed)
        std = np.sqrt(self._sigma2(x_np))
        eta1 = rng.standard_normal((n, self.n_sites))
        eta2 = rng.standard_normal((n, self.n_sites))
        eps = self._apply_G(std[None, :] * eta1) + self.delta * eta2
        return torch.from_numpy(eps).float()

    def _forward(self, x: TensorLike) -> TensorLike:
        """Integrate a single ring trajectory (one fresh process-noise path)."""
        if x.shape[0] != 1:
            msg = (
                "CorrelatedDiffusion1D._forward expects a single input, got "
                f"{x.shape[0]}"
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
        """Produce ``n`` ring trajectories in the spatiotemporal data layout.

        Returns
        -------
        dict
            ``data``: float32 ``(n, n_steps, n_sites, 1, 1)`` ring fields.
            ``constant_scalars``: ``zeros(n, 1)`` (the ring has no exposed
            scalar; kept for API parity with the OU twin).
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
        """Sample-feed Monte-Carlo rollout from a fixed state (the DGP oracle).

        Each of ``n_draws`` members advances its own raw state with fresh
        per-step forcing whose variance is evaluated at that member's own
        current state -- the exact predictive ceiling. Returns float32
        ``(n_draws, n_steps, n_sites, 1, 1)``.
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
