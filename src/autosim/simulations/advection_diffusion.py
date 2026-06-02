"""Advection-diffusion simulator and finite-difference solver helpers."""

import warnings

import numpy as np
import scipy.sparse as sp
from scipy.fft import fft2, ifft2
from scipy.integrate import solve_ivp

from autosim.simulations.spatiotemporal import AdvectionDiffusionMultichannel
from autosim.types import NumpyLike

integrator_keywords = {}
integrator_keywords["rtol"] = 1e-6
integrator_keywords["method"] = "RK45"
integrator_keywords["atol"] = 1e-8


class AdvectionDiffusion(AdvectionDiffusionMultichannel):
    r"""Deprecated vorticity-only advection-diffusion simulator.

    This legacy class delegates to
    :class:`autosim.simulations.spatiotemporal.AdvectionDiffusionMultichannel` with
    ``output_indices=[0]``.

    Use ``AdvectionDiffusionMultichannel(output_indices=[0])`` for the canonical
    vorticity-only API.
    """

    def __init__(
        self,
        parameters_range: dict[str, tuple[float, float]] | None = None,
        output_names: list[str] | None = None,
        return_timeseries: bool = False,
        log_level: str = "progress_bar",
        n: int = 50,
        L: float = 10.0,
        T: float = 80.0,
        dt: float = 0.25,
        integrator_kwargs: dict | None = None,
    ):
        """Initialize the deprecated vorticity-only wrapper.

        Args:
            parameters_range: Mapping of input parameter names to (min, max) ranges.
            output_names: List of output parameter names.
            log_level: Logging level for the simulator.
            return_timeseries: Whether to return the full timeseries or just the final
                snapshot.
            n: Number of spatial points per direction.
            L: Domain size in X and Y directions.
            T: Total simulation time.
            dt: Time step size.
            integrator_kwargs: Extra keyword arguments forwarded to the canonical
                multichannel simulator.
        """
        warnings.warn(
            "AdvectionDiffusion is deprecated. Use "
            "AdvectionDiffusionMultichannel(output_indices=[0]) from "
            "autosim.simulations.spatiotemporal instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            parameters_range=parameters_range,
            output_names=output_names,
            output_indices=[0],
            return_timeseries=return_timeseries,
            log_level=log_level,
            n=n,
            L=L,
            T=T,
            dt=dt,
            integrator_kwargs=integrator_kwargs,
        )


def create_sparse_matrices(
    n: int, N: int
) -> tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
    """Create sparse matrices A, Dx, and Dy for finite-difference operators."""
    e1 = np.ones(N)
    e2 = np.ones(N)
    e4 = np.zeros(N)

    for j in range(1, n + 1):
        e2[n * j - 1] = 0
        e4[n * j - 1] = 1

    e3 = np.zeros(N)
    e3[1:] = e2[:-1]
    e3[0] = e2[-1]

    e5 = np.zeros(N)
    e5[1:] = e4[:-1]
    e5[0] = e4[-1]

    # Create Laplacian matrix A
    diagonals = [e1, e1, e5, e2, -4 * e1, e3, e4, e1, e1]
    offsets = [-(N - n), -n, -n + 1, -1, 0, 1, n - 1, n, N - n]
    A = sp.diags(diagonals, offsets, shape=(N, N), format="csr").tolil()  # type: ignore[arg-type]
    A[0, 0] = 2

    # Create Dx matrix
    diagonals_x = [e1, -e1, e1, -e1]
    offsets_x = [-(N - n), -n, n, N - n]
    Dx = sp.diags(diagonals_x, offsets_x, shape=(N, N), format="csr").tolil()  # type: ignore[arg-type]

    # Create Dy matrix
    diagonals_y = [e5, -e2, e3, -e4]
    offsets_y = [-n + 1, -1, 1, n - 1]
    Dy = sp.diags(diagonals_y, offsets_y, shape=(N, N), format="csr").tolil()  # type: ignore[arg-type]

    return A.tocsr(), Dx.tocsr(), Dy.tocsr()


def advection_diffusion(
    _t: float,
    w2: NumpyLike,
    A: sp.csr_matrix,
    Dx: sp.csr_matrix,
    Dy: sp.csr_matrix,
    nu: float,
    dx: float,
    n: int,
    N: int,
    K3: NumpyLike,
    mu: float,
) -> NumpyLike:
    r"""Define the advection-diffusion RHS used by the ODE integrator.

    The vorticity equation is approximated as:

    .. math::

        \begin{aligned}
        \partial_t \omega
            &= \nu \nabla^2 \omega
            - \mu (u \partial_x \omega + v \partial_y \omega)
        \end{aligned}

    Args:
        _t: Current time (unused).
        w2: Flattened vorticity field.
        A: Sparse Laplacian operator.
        Dx: Sparse derivative operator in the x direction.
        Dy: Sparse derivative operator in the y direction.
        nu: Viscosity coefficient.
        dx: Spatial step.
        n: Number of spatial points per direction.
        N: Total number of spatial grid points.
        K3: Inverse Laplacian in Fourier space.
        mu: Advection strength.
    """
    w_2d = w2.reshape(n, n)

    # Compute stream function using FFT (Poisson solver)
    psi_2d = np.real(np.asarray(ifft2(-fft2(w_2d) * K3)))  # type: ignore[arg-type]
    psi2 = psi_2d.reshape(N)

    # Diffusion term + nonlinear advection terms
    return np.asarray(
        (nu / dx**2) * (A @ w2)
        - (0.25 / dx**2) * (Dx @ psi2) * (Dy @ w2) * mu
        + (0.25 / dx**2) * (Dy @ psi2) * (Dx @ w2) * mu
    )


def simulate_advection_diffusion(
    x: NumpyLike,
    return_timeseries: bool = False,
    n: int = 50,
    L: float = 10.0,
    T: float = 80.0,
    dt: float = 0.25,
) -> NumpyLike:
    """Simulate the 2D vorticity equation (advection-diffusion).

    Args:
        x: [nu, mu] parameters.
        return_timeseries: Whether to return full timeseries or only final snapshot.
        n: Number of spatial points per direction.
        L: Domain length in each spatial direction.
        T: Total simulation time.
        dt: Time step for saved solver outputs.
    """
    warnings.warn(
        "simulate_advection_diffusion from autosim.simulations.advection_diffusion "
        "is deprecated. Use "
        "autosim.simulations.spatiotemporal.advection_diffusion_multichannel."
        "simulate_advection_diffusion instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    nu, mu = x

    # Time vector
    tspan = np.arange(0, T, dt)
    n_time = len(tspan)

    # Spatial grid
    x_grid = np.linspace(-L / 2, L / 2, n)
    dx = float(x_grid[1] - x_grid[0])
    y_grid = x_grid
    N = n * n

    # Initial conditions - Gaussian vortex
    X, Y = np.meshgrid(x_grid, y_grid)
    w_initial = np.exp(-(X**2) - Y**2 / 20)

    # Create sparse matrices for finite differences
    A, Dx, Dy = create_sparse_matrices(n, N)

    # Wavenumber grid for FFT (Poisson solver)
    k = (2 * np.pi / L) * np.concatenate([np.arange(0, n // 2), np.arange(-n // 2, 0)])
    k[0] = 1e-6  # Avoid division by zero
    KX, KY = np.meshgrid(k, k)
    K3 = 1.0 / (KX**2 + KY**2)

    # Reshape initial condition
    w2_initial = w_initial.reshape(N)

    # Solve the ODE system
    sol = solve_ivp(
        lambda t, w2: advection_diffusion(t, w2, A, Dx, Dy, nu, dx, n, N, K3, mu),
        [0, T],
        w2_initial,
        t_eval=tspan,
        **integrator_keywords,
    )

    if return_timeseries:
        return sol.y.T.reshape(n_time, n, n)
    return sol.y[:, -1].reshape(n, n)
