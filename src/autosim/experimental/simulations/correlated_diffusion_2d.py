"""Correlated-diffusion torus (diff2d): the 2-D correlation-structure toy.

The 2-D version of :mod:`~autosim.experimental.simulations.correlated_diffusion_1d`. An
``n_side x n_side`` field on a torus evolves by a stable, linear, damped
diffusion (linear, learnable conditional mean -- no chaos), driven by Gaussian
forcing whose covariance is genuinely low-rank-plus-diagonal and state-dependent:

    x_{t+1} = A x_t + eps_t ,    A = (1 - dt*gamma) I + dt*kappa * Lap2D   (stable)
    eps_t   ~ N(0, Sigma(x_t)) , Sigma(x_t) = G diag(sigma2(x_t)) Gᵀ + delta^2 I
    sigma2_p(x_t) = sigma0^2 (1 + beta * |grad x_t|_p)   (forcing at fronts)

``A`` uses the 5-point torus Laplacian and ``G`` a 2-D Gaussian low-pass; the
emergent off-diagonal correlation is sharper than the ring's. ``Sigma(x)`` is
known exactly, so the oracle predictive ``N(A x, Sigma(x))`` is closed form
(:func:`diff2d_closed_form_cov`). It follows a fixed reference discretisation,
giving a process with a known correlation structure.

Twin of :class:`CorrelatedDiffusion1D` on a 2-D torus.
"""

from __future__ import annotations

import numpy as np
import torch

from autosim.simulations.base import SpatioTemporalSimulator
from autosim.types import TensorLike


def _torus_operators(
    n_side: int, dt: float, gamma: float, kappa: float, ell: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return the Fourier mean operator ``A_hat`` and smoother ``G_hat``.

    Uses a fixed reference discretisation: a 5-point torus Laplacian and a 2-D
    Gaussian low-pass (DC = 1). Both are ``(n_side, n_side)`` per-mode arrays.
    """
    kx = np.fft.fftfreq(n_side) * 2.0 * np.pi  # angular wavenumbers
    qx, qy = np.meshgrid(kx, kx, indexing="ij")
    lap = -4.0 * (np.sin(qx / 2.0) ** 2 + np.sin(qy / 2.0) ** 2)  # 5-point Laplacian
    a_hat = (1.0 - dt * gamma) + dt * kappa * lap
    g_hat = np.exp(-0.5 * ell**2 * (qx**2 + qy**2))
    return a_hat, g_hat


def _smoother_matrix(g_hat: np.ndarray) -> np.ndarray:
    """Build the explicit (symmetric) ``S x S`` smoother matrix from ``G_hat``.

    Applies ``apply_G`` to each unit impulse on the flattened field, then
    symmetrises to a symmetric smoother matrix.
    """
    n_side = g_hat.shape[0]
    s = n_side * n_side
    g = np.zeros((s, s))
    for p in range(s):
        e = np.zeros((n_side, n_side))
        e.flat[p] = 1.0
        smoothed = np.real(np.fft.ifft2(g_hat * np.fft.fft2(e), axes=(-2, -1)))
        g[:, p] = smoothed.reshape(-1)
    return 0.5 * (g + g.T)


def diff2d_closed_form_cov(
    x: TensorLike,
    *,
    n_side: int = 8,
    dt: float = 1.0,
    gamma: float = 0.2,
    kappa: float = 0.05,
    ell: float = 1.5,
    sigma0: float = 0.35,
    beta: float = 1.5,
    delta: float = 0.12,
) -> torch.Tensor:
    """Exact one-step covariance ``Sigma(x) = G diag(sigma2(x)) Gᵀ + delta^2 I``.

    Parameters
    ----------
    x: TensorLike
        A single torus field, any shape flattening to ``(n_side*n_side,)`` in
        row-major (C) order.
    n_side, dt, gamma, kappa, ell, sigma0, beta, delta:
        Torus DGP parameters; defaults match :class:`CorrelatedDiffusion2D`.

    Returns
    -------
    torch.Tensor
        The ``(n_side*n_side, n_side*n_side)`` covariance, float32.
    """
    x_np = (
        torch.as_tensor(x)
        .detach()
        .cpu()
        .numpy()
        .reshape(n_side, n_side)
        .astype(np.float64)
    )
    _, g_hat = _torus_operators(n_side, dt, gamma, kappa, ell)
    g_mat = _smoother_matrix(g_hat)
    gx = (np.roll(x_np, -1, -2) - np.roll(x_np, 1, -2)) * 0.5
    gy = (np.roll(x_np, -1, -1) - np.roll(x_np, 1, -1)) * 0.5
    grad_mag = np.sqrt(gx**2 + gy**2)
    s2 = (sigma0**2 * (1.0 + beta * grad_mag)).reshape(-1)
    cov = g_mat @ (s2[:, None] * g_mat.T) + delta**2 * np.eye(n_side * n_side)
    return torch.from_numpy(cov).float()


class CorrelatedDiffusion2D(SpatioTemporalSimulator):
    r"""Stable 2-D torus diffusion with a known low-rank-plus-diagonal forcing.

    The 2-D analogue of :class:`CorrelatedDiffusion1D`: the conditional mean is
    the contractive ``A = (1 - dt*gamma) I + dt*kappa * Lap2D`` (no chaos), the
    forcing is ``eps_t ~ N(0, Sigma(x_t))`` with the smoothed state-dependent
    ``Sigma`` above. The mean operator's spectral radius is checked ``< 1`` at
    construction (fail-early).

    Parameters
    ----------
    parameters_range, output_names, log_level:
        As in :class:`CorrelatedDiffusion1D` (API parity with the OU twin).
    n_steps: int, default=128
        Number of recorded steps per trajectory.
    burn: int, default=200
        Burn-in steps discarded before recording.
    n_side: int, default=8
        Torus side (``NS``); the field has ``n_side*n_side`` sites.
    dt, gamma, kappa: float
        Damped-diffusion mean operator parameters (``1.0, 0.2, 0.05``).
    ell: float, default=1.5
        Gaussian smoothing length in pixels.
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
        n_side: int = 8,
        dt: float = 1.0,
        gamma: float = 0.2,
        kappa: float = 0.05,
        ell: float = 1.5,
        sigma0: float = 0.35,
        beta: float = 1.5,
        delta: float = 0.12,
    ) -> None:
        if parameters_range is None:
            parameters_range = {"x0": (-1.0, 1.0)}
        if output_names is None:
            output_names = ["x"]

        self.a_hat, self.g_hat = _torus_operators(n_side, dt, gamma, kappa, ell)
        self._validate_dynamics(self.a_hat)

        super().__init__(parameters_range, output_names, log_level)

        self.n_steps = n_steps
        self.burn = burn
        self.n_side = n_side
        self.n_sites = n_side * n_side
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
                "CorrelatedDiffusion2D requires a contractive mean operator "
                f"(spectral radius |A| < 1) for a stable process; got |A| = {sr:.4f}."
            )
            raise ValueError(msg)

    def A_spectral_radius(self) -> float:
        """Spectral radius of the linear mean operator ``A`` (``< 1`` when stable)."""
        return float(np.max(np.abs(self.a_hat)))

    def _apply_A(self, x: np.ndarray) -> np.ndarray:
        """Apply the damped-diffusion mean operator ``A`` (2-D Fourier)."""
        return np.real(
            np.fft.ifft2(self.a_hat * np.fft.fft2(x, axes=(-2, -1)), axes=(-2, -1))
        )

    def _apply_G(self, v: np.ndarray) -> np.ndarray:
        """Apply the Gaussian low-pass smoother ``G`` (2-D Fourier)."""
        return np.real(
            np.fft.ifft2(self.g_hat * np.fft.fft2(v, axes=(-2, -1)), axes=(-2, -1))
        )

    def _grad_mag(self, x: np.ndarray) -> np.ndarray:
        """Magnitude of the central-difference gradient on the torus."""
        gx = (np.roll(x, -1, -2) - np.roll(x, 1, -2)) * 0.5
        gy = (np.roll(x, -1, -1) - np.roll(x, 1, -1)) * 0.5
        return np.sqrt(gx**2 + gy**2)

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
        x = 0.1 * rng.standard_normal((n, self.n_side, self.n_side))
        for _ in range(self.burn):
            x = self._step(x, rng)
        traj = np.empty((n, self.n_steps, self.n_side, self.n_side), dtype=np.float32)
        for t in range(self.n_steps):
            x = self._step(x, rng)
            traj[:, t] = x
        if not np.isfinite(traj).all():
            msg = "CorrelatedDiffusion2D generation diverged"
            raise RuntimeError(msg)
        return traj

    def sample_state(self, seed: int | None = None) -> torch.Tensor:
        """Draw one plausible torus field (flattened to ``(n_sites,)``)."""
        traj = self._generate(1, seed)
        return torch.from_numpy(traj[0, -1].reshape(-1)).float()

    def sample_noise(
        self, x: TensorLike, n: int, seed: int | None = None
    ) -> torch.Tensor:
        """Draw ``n`` forcing samples ``eps ~ N(0, Sigma(x))`` via the DGP construction.

        Uses the same ``G(sqrt(sigma2(x)) eta1) + delta eta2`` forcing as the
        dynamics, so the empirical covariance matches
        :func:`diff2d_closed_form_cov`. Returns ``(n, n_sites)`` row-major.
        """
        x_np = (
            torch.as_tensor(x).detach().cpu().numpy().reshape(self.n_side, self.n_side)
        )
        rng = np.random.default_rng(seed)
        std = np.sqrt(self._sigma2(x_np))
        eta1 = rng.standard_normal((n, self.n_side, self.n_side))
        eta2 = rng.standard_normal((n, self.n_side, self.n_side))
        eps = self._apply_G(std[None] * eta1) + self.delta * eta2
        return torch.from_numpy(eps.reshape(n, self.n_sites)).float()

    def _forward(self, x: TensorLike) -> TensorLike:
        """Integrate a single torus trajectory (one fresh process-noise path)."""
        if x.shape[0] != 1:
            msg = (
                "CorrelatedDiffusion2D._forward expects a single input, got "
                f"{x.shape[0]}"
            )
            raise ValueError(msg)
        traj = self._generate(1, None)  # (1, n_steps, n_side, n_side)
        return torch.from_numpy(traj.reshape(1, -1))

    def forward_samples_spatiotemporal(
        self,
        n: int,
        random_seed: int | None = None,
        ensure_exact_n: bool = False,  # noqa: ARG002
    ) -> dict:
        """Produce ``n`` torus trajectories in the spatiotemporal data layout.

        Returns
        -------
        dict
            ``data``: float32 ``(n, n_steps, n_side, n_side, 1)`` torus fields.
            ``constant_scalars``: ``zeros(n, 1)`` (API parity).
            ``constant_fields``: ``None``.
        """
        traj = self._generate(n, random_seed)  # (n, n_steps, n_side, n_side)
        data = torch.from_numpy(traj).reshape(
            n, self.n_steps, self.n_side, self.n_side, 1
        )
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
        """Sample-feed Monte-Carlo rollout from a fixed field (the DGP oracle).

        Each of ``n_draws`` members advances its own raw field with fresh
        per-step forcing evaluated at that member's own current state. Returns
        float32 ``(n_draws, n_steps, n_side, n_side, 1)``.
        """
        rng = np.random.default_rng(random_seed)
        x0 = (
            torch.as_tensor(x_state)
            .detach()
            .cpu()
            .numpy()
            .reshape(self.n_side, self.n_side)
            .astype(np.float64)
        )
        xm = np.repeat(x0[None], n_draws, axis=0)  # (n_draws, n_side, n_side)
        out = np.empty((n_draws, n_steps, self.n_side, self.n_side), dtype=np.float32)
        for t in range(n_steps):
            xm = self._step(xm, rng)
            out[:, t] = xm
        return torch.from_numpy(out).reshape(
            n_draws, n_steps, self.n_side, self.n_side, 1
        )
