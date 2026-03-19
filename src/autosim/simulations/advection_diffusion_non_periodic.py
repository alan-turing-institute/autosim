from collections.abc import Callable

import numpy as np
import scipy.sparse as sp
import torch
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import factorized

from autosim.simulations.base import SpatioTemporalSimulator
from autosim.types import NumpyLike, TensorLike

integrator_keywords = {}
integrator_keywords["rtol"] = 1e-6
integrator_keywords["method"] = "RK45"
integrator_keywords["atol"] = 1e-8


class AdvectionDiffusionNonPeriodic(SpatioTemporalSimulator):
    """Simulate 2D vorticity with non-periodic (rigid) boundary conditions."""

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
    ):
        """
        Initialize the non-periodic AdvectionDiffusion simulator.

        Parameters
        ----------
        parameters_range: dict[str, tuple[float, float]]
            Mapping of input parameter names to (min, max) ranges.
        output_names: list[str]
            List of output parameter names.
        log_level: str
            Logging level for the simulator.
        return_timeseries: bool
            Whether to return the full timeseries or just the final snapshot.
        n: int
            Number of spatial points per direction.
        L: float
            Domain size in X and Y directions.
        T: float
            Total simulation time.
        dt: float
            Time step size.
        """
        if parameters_range is None:
            parameters_range = {
                "nu": (0.0001, 0.01),  # viscosity
                "mu": (0.5, 2.0),  # advection strength
            }
        if output_names is None:
            output_names = ["vorticity"]
        super().__init__(parameters_range, output_names, log_level)
        self.return_timeseries = return_timeseries
        self.n = n
        self.L = L
        self.T = T
        self.dt = dt

    def _forward(self, x: TensorLike) -> TensorLike:
        assert x.shape[0] == 1, (
            f"Simulator._forward expects a single input, got {x.shape[0]}"
        )

        vorticity_sol = simulate_advection_diffusion_non_periodic(
            x.cpu().numpy()[0], self.return_timeseries, self.n, self.L, self.T, self.dt
        )

        return torch.tensor(vorticity_sol.ravel(), dtype=torch.float32).reshape(1, -1)

    def forward_samples_spatiotemporal(
        self,
        n: int,
        random_seed: int | None = None,
        ensure_exact_n: bool = False,
    ) -> dict:
        """Reshape to spatiotemporal format and return data plus constants."""
        y, x = self._forward_batch_with_optional_retries(
            n=n,
            random_seed=random_seed,
            ensure_exact_n=ensure_exact_n,
        )

        if self.return_timeseries:
            n_time = int(self.T / self.dt)
            y_reshaped = y.reshape(y.shape[0], n_time, self.n, self.n, 1)
        else:
            y_reshaped = y.reshape(y.shape[0], 1, self.n, self.n, 1)

        return {
            "data": y_reshaped,
            "constant_scalars": x,
            "constant_fields": None,
        }


def create_sparse_matrices_non_periodic(
    n: int, N: int
) -> tuple[sp.dia_matrix, sp.dia_matrix, sp.dia_matrix]:
    """Create sparse matrices for finite differences without wrap-around."""
    e1 = np.ones(N)
    e2 = np.ones(N)

    for j in range(1, n + 1):
        e2[n * j - 1] = 0

    e3 = np.zeros(N)
    e3[1:] = e2[:-1]
    e3[0] = 0

    # Create Laplacian matrix A without wrap-around offsets
    diagonals = [e1, e2, -4 * e1, e3, e1]
    offsets = [-n, -1, 0, 1, n]
    A = sp.diags(diagonals=diagonals, offsets=offsets, shape=(N, N), format="csr")  # type: ignore since sp.diags supports sequences

    # Create Dx matrix without wrap-around
    diagonals_x = [-e1, e1]
    offsets_x = [-n, n]
    Dx = sp.diags(diagonals=diagonals_x, offsets=offsets_x, shape=(N, N), format="csr")  # type: ignore since sp.diags supports sequences

    # Create Dy matrix without wrap-around
    diagonals_y = [-e2, e3]
    offsets_y = [-1, 1]
    Dy = sp.diags(diagonals=diagonals_y, offsets=offsets_y, shape=(N, N), format="csr")  # type: ignore since sp.diags supports sequences

    return A, Dx, Dy


def advection_diffusion_non_periodic(
    _t: float,
    w2: NumpyLike,
    A: sp.dia_matrix,
    Dx: sp.dia_matrix,
    Dy: sp.dia_matrix,
    nu: float,
    dx: float,
    mu: float,
    solve_A: Callable,
) -> NumpyLike:
    """Define the advection-diffusion RHS without periodic boundaries."""
    # Inverse laplacian via direct sparse solver instead of FFT
    # Note: A is the discrete Laplacian, so A*psi = -w*dx^2 physically.
    # We solve A*psi_unit = -w, then multiply by dx^2 to get physical psi.
    psi2 = solve_A(-w2) * (dx**2)

    # Diffusion term + nonlinear advection terms
    return np.asarray(
        (nu / dx**2) * (A @ w2)
        - (0.25 / dx**2) * (Dx @ psi2) * (Dy @ w2) * mu
        + (0.25 / dx**2) * (Dy @ psi2) * (Dx @ w2) * mu
    )


def simulate_advection_diffusion_non_periodic(
    x: NumpyLike,
    return_timeseries: bool = False,
    n: int = 50,
    L: float = 10.0,
    T: float = 80.0,
    dt: float = 0.25,
) -> NumpyLike:
    """Simulate the 2D vorticity equation with non-periodic barriers."""
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
    A, Dx, Dy = create_sparse_matrices_non_periodic(n, N)

    # Pre-factorize the sparse negative Laplacian for the Poisson solver
    solve_A = factorized(A.tocsc())

    # Reshape initial condition
    w2_initial = w_initial.reshape(N)

    # Solve the ODE system
    sol = solve_ivp(
        lambda t, w2: advection_diffusion_non_periodic(
            t, w2, A, Dx, Dy, nu, dx, mu, solve_A
        ),
        [0, T],
        w2_initial,
        t_eval=tspan,
        **integrator_keywords,
    )

    if return_timeseries:
        return sol.y.T.reshape(n_time, n, n)
    return sol.y[:, -1].reshape(n, n)
