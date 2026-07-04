"""Correlated-diffusion torus (diff2d): the 2-D correlation-structure toy.

The 2-D version of :mod:`~autosim.experimental.simulations.correlated_diffusion_1d`.
An ``n_side x n_side`` field on a torus evolves by a stable, linear, damped
diffusion (linear conditional mean -- no chaos), driven by Gaussian forcing whose
covariance is genuinely low-rank-plus-diagonal and state-dependent:

    x_{t+1} = A x_t + eps_t ,    A = (1 - dt*gamma) I + dt*kappa * Lap2D   (stable)
    eps_t   ~ N(0, Sigma(x_t)) , Sigma(x_t) = G diag(sigma2(x_t)) G^T + delta^2 I
    sigma2_p(x_t) = sigma0^2 (1 + beta * |grad x_t|_p)   (forcing at fronts)

``A`` uses the 5-point torus Laplacian and ``G`` a 2-D Gaussian low-pass; the
emergent off-diagonal correlation is sharper than the ring's. ``Sigma(x)`` is
known exactly, so the oracle predictive ``N(A x, Sigma(x))`` is closed form
(:meth:`CorrelatedDiffusion2D.closed_form_cov`).

Twin of
:class:`~autosim.experimental.simulations.correlated_diffusion_1d.CorrelatedDiffusion1D`
on a 2-D torus; both share the generation and forcing machinery in
:class:`~autosim.experimental.simulations._stochastic_base.LinearDiffusionSimulator`.
"""

from __future__ import annotations

import numpy as np

from ._stochastic_base import LinearDiffusionSimulator


def _torus_operators(
    n_side: int, dt: float, gamma: float, kappa: float, ell: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return the Fourier mean operator ``A_hat`` and smoother ``G_hat``.

    Uses a fixed reference discretisation: a 5-point torus Laplacian and a 2-D
    Gaussian low-pass (DC = 1). Both are ``(n_side, n_side)`` per-mode arrays.

    Args:
        n_side: Torus side length.
        dt: Time step.
        gamma: Damping rate.
        kappa: Diffusion rate.
        ell: Gaussian smoothing length in pixels.

    Returns:
        The per-mode mean operator ``A_hat`` and smoother ``G_hat``, each shape
        ``(n_side, n_side)``.
    """
    kx = np.fft.fftfreq(n_side) * 2.0 * np.pi  # angular wavenumbers
    qx, qy = np.meshgrid(kx, kx, indexing="ij")
    lap = -4.0 * (np.sin(qx / 2.0) ** 2 + np.sin(qy / 2.0) ** 2)  # 5-point Laplacian
    a_hat = (1.0 - dt * gamma) + dt * kappa * lap
    g_hat = np.exp(-0.5 * ell**2 * (qx**2 + qy**2))
    return a_hat, g_hat


class CorrelatedDiffusion2D(LinearDiffusionSimulator):
    r"""Stable 2-D torus diffusion with a known low-rank-plus-diagonal forcing.

    The 2-D analogue of :class:`CorrelatedDiffusion1D`: the conditional mean is
    the contractive ``A = (1 - dt*gamma) I + dt*kappa * Lap2D`` (no chaos), the
    forcing is ``eps_t ~ N(0, Sigma(x_t))`` with the smoothed state-dependent
    ``Sigma`` above. The mean operator's spectral radius is checked ``< 1`` at
    construction (fail-early).

    Args:
        parameters_range: Placeholder bound for the (unused) sampled input;
            defaults to ``{"x0": (-1.0, 1.0)}`` for API parity with the OU twin.
        output_names: Channel name; defaults to ``["x"]``.
        log_level: Logging verbosity passed to the base ``Simulator``.
        n_steps: Number of recorded steps per trajectory.
        burn: Burn-in steps discarded before recording.
        n_side: Torus side; the field has ``n_side*n_side`` sites.
        dt: Time step of the damped-diffusion mean operator.
        gamma: Damping rate of the mean operator.
        kappa: Diffusion rate of the mean operator.
        ell: Gaussian smoothing length in pixels.
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
        n_side: int = 8,
        dt: float = 1.0,
        gamma: float = 0.2,
        kappa: float = 0.05,
        ell: float = 1.5,
        sigma0: float = 0.35,
        beta: float = 1.5,
        delta: float = 0.12,
    ) -> None:
        """Initialize the torus diffusion integrator and validate its dynamics."""
        if parameters_range is None:
            parameters_range = {"x0": (-1.0, 1.0)}
        if output_names is None:
            output_names = ["x"]

        self.a_hat, self.g_hat = _torus_operators(n_side, dt, gamma, kappa, ell)
        self._field_shape = (n_side, n_side)
        self._field_axes = (-2, -1)
        self._validate_dynamics()

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

    def _grad_mag(self, x: np.ndarray) -> np.ndarray:
        """Magnitude of the central-difference gradient on the torus."""
        gx = (np.roll(x, -1, -2) - np.roll(x, 1, -2)) * 0.5
        gy = (np.roll(x, -1, -1) - np.roll(x, 1, -1)) * 0.5
        return np.sqrt(gx**2 + gy**2)
