"""Correlated-diffusion ring (diff1d): the 1-D correlation-structure toy.

An ``n_sites`` field on a ring evolves by a stable, linear, damped diffusion
(linear conditional mean -- no chaos), driven by Gaussian forcing whose
covariance is genuinely low-rank-plus-diagonal and state-dependent:

    x_{t+1} = A x_t + eps_t ,    A = (1 - dt*gamma) I + dt*kappa * Lap1D   (stable)
    eps_t   ~ N(0, Sigma(x_t)) , Sigma(x_t) = G diag(sigma2(x_t)) G^T + delta^2 I
    sigma2_i(x_t) = sigma0^2 (1 + beta * |grad x_t|_i)   (forcing at fronts)

``A`` uses the 3-point ring Laplacian and ``G`` a Gaussian low-pass; the
off-diagonal correlation emerges from the smoother ``G``. ``Sigma(x)`` is known
exactly, so the oracle predictive ``N(A x, Sigma(x))`` is closed form
(:meth:`CorrelatedDiffusion1D.closed_form_cov`).

Twin of
:class:`~autosim.experimental.simulations.correlated_diffusion_2d.CorrelatedDiffusion2D`
on a 2-D torus; both share the generation and forcing machinery in
:class:`~autosim.experimental.simulations._stochastic_base.LinearDiffusionSimulator`.
"""

from __future__ import annotations

import numpy as np

from ._stochastic_base import LinearDiffusionSimulator


def _ring_operators(
    n_sites: int, dt: float, gamma: float, kappa: float, ell: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return the Fourier mean operator ``A_hat`` and smoother ``G_hat``.

    Uses a fixed reference discretisation: a 3-point Laplacian on the ring and a
    Gaussian low-pass (DC = 1).

    Args:
        n_sites: Number of ring sites.
        dt: Time step.
        gamma: Damping rate.
        kappa: Diffusion rate.
        ell: Gaussian smoothing length in sites.

    Returns:
        The per-mode mean operator ``A_hat`` and smoother ``G_hat``, each shape
        ``(n_sites,)``.
    """
    q = np.fft.fftfreq(n_sites) * 2.0 * np.pi  # angular wavenumbers
    lap = -4.0 * np.sin(q / 2.0) ** 2  # 3-point Laplacian eigenvalues, [-4, 0]
    a_hat = (1.0 - dt * gamma) + dt * kappa * lap  # mean operator A (per mode)
    g_hat = np.exp(-0.5 * ell**2 * q**2)  # Gaussian low-pass (DC = 1)
    return a_hat, g_hat


class CorrelatedDiffusion1D(LinearDiffusionSimulator):
    r"""Stable 1-D ring diffusion with a known low-rank-plus-diagonal forcing.

    The conditional mean is the linear, contractive operator
    ``A = (1 - dt*gamma) I + dt*kappa * Lap1D`` (no chaos); the forcing is
    ``eps_t ~ N(0, Sigma(x_t))`` with ``Sigma(x_t) = G diag(sigma2(x_t)) G^T +
    delta^2 I`` and ``sigma2_i = sigma0^2 (1 + beta |grad x|_i)``. The mean
    operator's spectral radius is checked ``< 1`` at construction (fail-early).

    Args:
        parameters_range: Placeholder bound for the (unused) sampled input;
            defaults to ``{"x0": (-1.0, 1.0)}`` for API parity with the OU twin.
            Trajectories are produced by ``forward_samples_spatiotemporal`` from a
            seeded burn-in, not from this input.
        output_names: Channel name; defaults to ``["x"]``.
        log_level: Logging verbosity passed to the base ``Simulator``.
        n_steps: Number of recorded steps per trajectory (``T``).
        burn: Burn-in steps discarded before recording.
        n_sites: Ring sites (``N``).
        dt: Time step of the damped-diffusion mean operator.
        gamma: Damping rate of the mean operator.
        kappa: Diffusion rate of the mean operator.
        ell: Gaussian smoothing length in sites (sets the effective rank).
        sigma0: Base forcing standard deviation.
        beta: Gradient/front sensitivity of the forcing amplitude.
        delta: Diagonal nugget standard deviation.
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
        """Initialize the ring diffusion integrator and validate its dynamics."""
        if parameters_range is None:
            parameters_range = {"x0": (-1.0, 1.0)}
        if output_names is None:
            output_names = ["x"]

        self.a_hat, self.g_hat = _ring_operators(n_sites, dt, gamma, kappa, ell)
        self._field_shape = (n_sites,)
        self._field_axes = (-1,)
        self._validate_dynamics()

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

    def _grad_mag(self, x: np.ndarray) -> np.ndarray:
        """Magnitude of the central-difference gradient on the ring."""
        return np.abs((np.roll(x, -1, -1) - np.roll(x, 1, -1)) * 0.5)
