import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.ndimage import laplace

from autosim.simulations.base import SpatioTemporalSimulator
from autosim.types import NumpyLike, TensorLike

integrator_keywords = {}
integrator_keywords["rtol"] = 1e-12
integrator_keywords["method"] = "RK45"
integrator_keywords["atol"] = 1e-12


class ReactionDiffusionNonPeriodic(SpatioTemporalSimulator):
    """Simulate the reaction-diffusion PDE with non-periodic boundary conditions."""

    def __init__(
        self,
        parameters_range: dict[str, tuple[float, float]] | None = None,
        output_names: list[str] | None = None,
        return_timeseries: bool = False,
        log_level: str = "progress_bar",
        n: int = 32,
        L: int = 20,
        T: float = 10.0,
        dt: float = 0.1,
    ):
        """
        Initialize the ReactionDiffusionNonPeriodic simulator.

        Parameters
        ----------
        parameters_range: dict[str, tuple[float, float]]
            Dictionary mapping input parameter names to their (min, max) ranges.
        output_names: list[str]
            List of output parameters' names.
        log_level: str
            Logging level for the simulator.
        return_timeseries: bool
            Whether to return the full timeseries or just the spatial solution at the
            final time step. Defaults to False.
        n: int
            Number of spatial points in each direction.
        L: int
            Domain size in X and Y directions.
        T: float
            Total time to simulate.
        dt: float
            Time step size.
        """
        if parameters_range is None:
            parameters_range = {"beta": (1.0, 2.0), "d": (0.05, 0.3)}
        if output_names is None:
            output_names = ["u", "v"]
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
        u_sol, v_sol = simulate_reaction_diffusion_non_periodic(
            x.cpu().numpy()[0], self.return_timeseries, self.n, self.L, self.T, self.dt
        )

        # concatenate U and V arrays (flattened across time and space)
        concat_array = np.concatenate([u_sol.ravel(), v_sol.ravel()])

        # return tensor shape (1, 2*self.t*self.n*self.n)
        return torch.tensor(concat_array, dtype=torch.float32).reshape(1, -1)

    def forward_samples_spatiotemporal(
        self,
        n: int,
        random_seed: int | None = None,
        ensure_exact_n: bool = False,
    ) -> dict:
        """Reshape to spatiotemporal format.

        Parameters
        ----------
        n: int
            Number of samples to generate.
        random_seed: int | None
            Random seed for reproducibility. Defaults to None.

        Returns
        -------
        dict
            A dictionary containing the reshaped spatiotemporal data, constant scalars,
            and constant fields.
        """
        # Run simulation and optionally resample failed trajectories
        y, x = self._forward_batch_with_optional_retries(
            n=n,
            random_seed=random_seed,
            ensure_exact_n=ensure_exact_n,
        )

        # Reshape and permute output
        y_reshaped_permuted = y.reshape(
            y.shape[0], 2, int(self.T / self.dt), self.n, self.n
        ).permute(0, 2, 3, 4, 1)

        return {
            "data": y_reshaped_permuted,
            "constant_scalars": x,
            "constant_fields": None,
        }


def reaction_diffusion_fdm(
    t: float,  # noqa: ARG001
    uv: NumpyLike,
    d1: float,
    d2: float,
    beta: float,
    n: int,
    N: int,
    dx2: float,
):
    """Define the reaction-diffusion PDE in real space using Finite Differences."""
    u = uv[:N].reshape((n, n))
    v = uv[N:].reshape((n, n))

    # Non-periodic boundary conditions via mode='reflect' (Neumann/zero-flux)
    del2_u = laplace(u, mode="reflect") / dx2
    del2_v = laplace(v, mode="reflect") / dx2

    u3 = u**3
    v3 = v**3
    u2v = (u**2) * v
    uv2 = u * (v**2)

    du_dt = d1 * del2_u + u - u3 - uv2 + beta * u2v + beta * v3
    dv_dt = d2 * del2_v + v - u2v - v3 - beta * u3 - beta * uv2

    return np.concatenate([du_dt.ravel(), dv_dt.ravel()])


def simulate_reaction_diffusion_non_periodic(
    x: NumpyLike,
    return_timeseries: bool = False,
    n: int = 32,
    L: int = 20,
    T: float = 10.0,
    dt: float = 0.1,
) -> tuple[NumpyLike, NumpyLike]:
    """Simulate the reaction-diffusion PDE with non-periodic boundaries."""
    beta, d = x
    d1 = d2 = d

    t_eval = np.linspace(0, T, int(T / dt))

    N = n * n
    x_uniform = np.linspace(-L / 2, L / 2, n + 1)
    x_grid = x_uniform[:n]
    y_grid = x_uniform[:n]
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

    dx = L / n
    dx2 = dx**2

    m = 1

    # Initial conditions
    u0 = np.tanh(np.sqrt(X_grid**2 + Y_grid**2)) * np.cos(
        m * np.angle(X_grid + 1j * Y_grid) - (np.sqrt(X_grid**2 + Y_grid**2))
    )
    v0 = np.tanh(np.sqrt(X_grid**2 + Y_grid**2)) * np.sin(
        m * np.angle(X_grid + 1j * Y_grid) - (np.sqrt(X_grid**2 + Y_grid**2))
    )

    uv0 = np.concatenate([u0.ravel(), v0.ravel()])

    # Solve the PDE in real space
    uvsol = solve_ivp(
        reaction_diffusion_fdm,
        (t_eval[0], t_eval[-1]),
        y0=uv0,
        t_eval=t_eval,
        args=(d1, d2, beta, n, N, dx2),
        **integrator_keywords,
    )
    uvsol = uvsol.y

    u_sol = uvsol[:N, :]
    v_sol = uvsol[N:, :]

    if return_timeseries:
        u = u_sol.reshape((n, n, len(t_eval))).transpose((2, 0, 1))
        v = v_sol.reshape((n, n, len(t_eval))).transpose((2, 0, 1))
        return u, v

    u_last = u_sol[:, -1].reshape((n, n))
    v_last = v_sol[:, -1].reshape((n, n))
    return u_last, v_last
