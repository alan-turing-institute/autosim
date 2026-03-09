import numpy as np
import torch
from numpy.fft import fft2, ifft2
from scipy.integrate import solve_ivp

from autosim.simulations.base import SpatioTemporalSimulator
from autosim.types import NumpyLike, TensorLike

integrator_keywords = {"rtol": 1e-6, "atol": 1e-6, "method": "RK45"}


class ReactionDiffusion(SpatioTemporalSimulator):
    """Simulate the reaction-diffusion PDE for a given set of parameters."""

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
        Initialize the ReactionDiffusion simulator.

        Parameters
        ----------
        parameters_range: dict[str, tuple[float, float]]
            Dictionary mapping input parameter names to their (min, max) ranges.
        output_names: list[str]
            List of output parameters' names.
        log_level: str
            Logging level for the simulator. Can be one of:
            - "progress_bar": shows a progress bar during batch simulations
            - "debug": shows debug messages
            - "info": shows informational messages
            - "warning": shows warning messages
            - "error": shows error messages
            - "critical": shows critical messages
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
        u_sol, v_sol = simulate_reaction_diffusion(
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


def reaction_diffusion(
    t: float,  # noqa: ARG001
    uvt: NumpyLike,
    K22: NumpyLike,
    d1: float,
    d2: float,
    beta: float,
    n: int,
    N: int,
):
    """
    Define the reaction-diffusion PDE in the Fourier (kx, ky) space.

    Parameters
    ----------
    t: float
        The current time step (not used).
    uvt: NumpyLike
        Fourier transformed solution vector at current time step (length 2*N, 1-D).
    K22: NumpyLike
        Squared Fourier wavenumbers, shape (N,).
    d1: float
        The diffusion coefficient for species 1.
    d2: float
        The diffusion coefficient for species 2.
    beta: float
        The reaction coefficient controlling reaction between the two species.
    n: int
        Number of spatial points in each direction.
    N: int
        Total number of spatial grid points (n*n).
    """
    u = np.real(ifft2(uvt[:N].reshape(n, n)))
    v = np.real(ifft2(uvt[N:].reshape(n, n)))
    u2v = u * u * v
    uv2 = u * v * v
    rhs = np.empty(2 * N, dtype=complex)
    rhs[:N] = fft2(u - u**3 - uv2 + beta * (u2v + v**3)).ravel() - d1 * K22 * uvt[:N]
    rhs[N:] = fft2(v - u2v - v**3 - beta * (u**3 + uv2)).ravel() - d2 * K22 * uvt[N:]
    return rhs


def simulate_reaction_diffusion(
    x: NumpyLike,
    return_timeseries: bool = False,
    n: int = 32,
    L: int = 20,
    T: float = 10.0,
    dt: float = 0.1,
) -> tuple[NumpyLike, NumpyLike]:
    """
    Simulate the reaction-diffusion PDE for a given set of parameters.

    Parameters
    ----------
    x: NumpyLike
        The parameters of the reaction-diffusion model. The first element is the
        reaction coefficient (beta) and the second element is the diffusion
        coefficient (d).
    return_timeseries: bool
        Whether to return the full timeseries or just the spatial solution at the final
        time step. Defaults to False.
    n: int
        Number of spatial points in each direction. Defaults to 32.
    L: int
        Domain size in X and Y directions. Defaults to 20.
    T: float
        Total time to simulate. Defaults to 10.0.
    dt: float
        Time step size. Defaults to 0.1.

    Returns
    -------
    tuple[NumpyLike, NumpyLike]
        [u_sol, v_sol], the spatial solution of the reaction-diffusion PDE, either as a
        timeseries or at the final time point of `return_timeseries` is False.
    """
    beta, d = x
    d1 = d2 = d

    t = np.linspace(0, T, int(T / dt))

    N = n * n
    x_uniform = np.linspace(-L / 2, L / 2, n + 1)
    x_grid = x_uniform[:n]
    n2 = n // 2
    kx = (2 * np.pi / L) * np.hstack(
        (np.linspace(0, n2 - 1, n2), np.linspace(-n2, -1, n2))
    )
    X_grid, Y_grid = np.meshgrid(x_grid, x_grid)
    KX, KY = np.meshgrid(kx, kx)
    K2 = KX**2 + KY**2
    K22 = K2.ravel()

    r = np.sqrt(X_grid**2 + Y_grid**2)
    theta = np.angle(X_grid + 1j * Y_grid)
    u0 = np.tanh(r) * np.cos(theta - r)
    v0 = np.tanh(r) * np.sin(theta - r)

    uvt0 = np.hstack([fft2(u0).ravel(), fft2(v0).ravel()])

    uvsol = solve_ivp(
        reaction_diffusion,
        (t[0], t[-1]),
        y0=uvt0,
        t_eval=t,
        args=(K22, d1, d2, beta, n, N),
        **integrator_keywords,
    )
    uvsol = uvsol.y

    u_out = np.array(
        [np.real(ifft2(uvsol[:N, j].reshape(n, n))) for j in range(len(t))]
    )
    v_out = np.array(
        [np.real(ifft2(uvsol[N:, j].reshape(n, n))) for j in range(len(t))]
    )

    if return_timeseries:
        return u_out, v_out
    return u_out[-1], v_out[-1]
