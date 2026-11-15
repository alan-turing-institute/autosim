"""Operators, tendencies and time step of the shallow water equations on the sphere."""

from __future__ import annotations

import torch

from autosim.types import TensorLike

try:
    import torch_harmonics as th  # pyright: ignore[reportMissingImports]
except ImportError:  # optional dependency: the "spherical" extra
    th = None


TORCH_HARMONICS_MISSING = (
    "SphericalShallowWater requires torch-harmonics (extra 'spherical')."
)


def make_grid(grid: str, nlat: int | None, nlon: int | None, nside: int | None):
    """Build a torch-harmonics grid descriptor.

    Args:
        grid: "legendre-gauss", "equiangular", "lobatto" or "healpix".
        nlat: Rings, for regular grids.
        nlon: Points per ring, for regular grids.
        nside: Resolution, for "healpix".

    Returns:
        The grid descriptor.
    """
    if th is None:
        raise ImportError(TORCH_HARMONICS_MISSING)
    if grid == "healpix":
        return th.as_grid("healpix", nside=nside)
    return th.as_grid(grid, nlat=nlat, nlon=nlon)


class SphereShallowWaterDynamics:
    """Fixed operators for the shallow water equations on the sphere.

    Args:
        grid: torch-harmonics grid descriptor (any sampling: Gauss-Legendre,
            equiangular, HEALPix, ...).
        lmax: Spectral band limit L; degrees l = 0 .. L-1 are kept.
        radius: Planet radius a [m].
        omega: Rotation rate [1/s].
        gravity: Gravitational acceleration [m/s^2].
        mean_depth: Global mean fluid depth [m].
        dt: Solver time step [s].
        device: Torch device of the operators.
    """

    def __init__(
        self,
        grid,
        lmax: int,
        radius: float,
        omega: float,
        gravity: float,
        mean_depth: float,
        dt: float,
        device: str = "cuda",
    ) -> None:
        """Build the transforms, the Laplacian, Coriolis and the damping."""
        if th is None:
            raise ImportError(TORCH_HARMONICS_MISSING)
        self.grid = grid
        self.radius = radius
        self.gravity = gravity
        self.mean_depth = mean_depth
        self.dt = dt

        # The products of two fields of degree < L have degree < 2L - 1. Projecting them
        # back onto degree < L is exact only if the quadrature reaches (3L - 1) / 2.
        try:
            exact_degree = grid.max_exact_degree
        except NotImplementedError:  # HEALPix: no exact quadrature
            exact_degree = 0
        self.exact_products = 2 * exact_degree >= 3 * lmax - 1

        # Transforms between the grid and spherical-harmonic coefficients
        self.sht = th.RealSHT(grid, lmax=lmax, mmax=lmax, csphase=False).to(device)
        self.isht = th.InverseRealSHT(grid, lmax=lmax, mmax=lmax, csphase=False).to(
            device
        )
        self.vsht = th.RealVectorSHT(grid, lmax=lmax, mmax=lmax, csphase=False).to(
            device
        )
        self.ivsht = th.InverseRealVectorSHT(
            grid, lmax=lmax, mmax=lmax, csphase=False
        ).to(device)

        # Laplacian on the sphere: nabla^2 Y_l^m = -l(l+1)/a^2 Y_l^m
        degree = torch.arange(lmax, dtype=torch.float64, device=device)
        degree = degree.reshape(lmax, 1).expand(lmax, lmax)
        self.laplacian = -degree * (degree + 1) / radius**2
        self.inverse_laplacian = torch.zeros_like(self.laplacian)
        self.inverse_laplacian[1:] = 1.0 / self.laplacian[1:]  # l = 0: set to 0

        # Coriolis parameter f = 2 Omega sin(latitude) at every grid point
        colatitude = grid.coords[:, 0].to(device=device, dtype=torch.float64)
        latitude = torch.pi / 2 - colatitude
        self.coriolis = (2 * omega * torch.sin(latitude)).reshape(grid.shape)

        # Hyperdiffusion: e-folding time of 2 h at the largest kept degree
        damping_time = 2 * 3600.0
        scale = self.laplacian / self.laplacian[-1, 0]  # l(l+1) / ((L-1)L)
        self.hyperdiffusion = torch.exp(-(dt / damping_time) * scale**4)

        # Damping of one step E = exp(Lambda dt) for (geopotential, vorticity,
        # divergence); the geopotential is not damped.
        no_damping = torch.ones_like(self.hyperdiffusion)
        self.step_damping = torch.stack(
            [no_damping, self.hyperdiffusion, self.hyperdiffusion]
        )

    def to_grid(self, coefficients: TensorLike) -> TensorLike:
        """Scalar fields: coefficients (B, L, L) -> grid values (B, *grid.shape)."""
        return self.isht(coefficients)

    def to_coefficients(self, field: TensorLike) -> TensorLike:
        """Scalar fields: grid values (B, *grid.shape) -> coefficients (B, L, L)."""
        return self.sht(field)

    def wind(self, vorticity: TensorLike, divergence: TensorLike) -> TensorLike:
        """Vorticity and divergence coefficients -> wind (B, 2, *grid.shape).

        psi = nabla^-2 vorticity, chi = nabla^-2 divergence and
        u = k x grad(psi) + grad(chi). Component 0 is eastward, 1 is northward.
        """
        potentials = torch.stack([vorticity, divergence], dim=1)
        return self.ivsht(potentials * self.inverse_laplacian / self.radius)

    def vorticity_divergence(
        self, vector_field: TensorLike
    ) -> tuple[TensorLike, TensorLike]:
        """Vector fields (B, 2, *grid.shape) -> curl and divergence coefficients."""
        coefficients = self.laplacian * self.radius * self.vsht(vector_field)
        return coefficients[:, 0], coefficients[:, 1]


def shallow_water_tendencies(
    dynamics: SphereShallowWaterDynamics, state: TensorLike
) -> TensorLike:
    """Time derivatives of the (geopotential, vorticity, divergence) coefficients."""
    geopotential = state[:, 0]
    vorticity = state[:, 1]
    divergence = state[:, 2]

    # 1. Fields on the grid
    geopotential_grid = dynamics.to_grid(geopotential)
    vorticity_grid = dynamics.to_grid(vorticity)
    wind = dynamics.wind(vorticity, divergence)

    # 2. Products on the grid
    absolute_vorticity = vorticity_grid + dynamics.coriolis
    vorticity_flux = wind * absolute_vorticity.unsqueeze(1)  # (zeta + f) u
    geopotential_flux = wind * geopotential_grid.unsqueeze(1)  # Phi u
    kinetic_energy = 0.5 * (wind[:, 0] ** 2 + wind[:, 1] ** 2)
    bernoulli = geopotential_grid + kinetic_energy  # Phi + |u|^2 / 2

    # 3. Back to coefficients
    flux_curl, flux_divergence = dynamics.vorticity_divergence(vorticity_flux)
    _, geopotential_flux_divergence = dynamics.vorticity_divergence(geopotential_flux)
    bernoulli_coefficients = dynamics.to_coefficients(bernoulli)

    d_geopotential = -geopotential_flux_divergence
    d_vorticity = -flux_divergence
    d_divergence = flux_curl - dynamics.laplacian * bernoulli_coefficients
    return torch.stack([d_geopotential, d_vorticity, d_divergence], dim=1)


def adams_bashforth_step(
    dynamics: SphereShallowWaterDynamics,
    state: TensorLike,
    previous_tendencies: list[TensorLike],
) -> tuple[TensorLike, list[TensorLike]]:
    """One Adams-Bashforth 3 step with the hyperdiffusion as an integrating factor.

    s_{n+1} = E s_n + dt (23/12 E N_n - 16/12 E^2 N_{n-1} + 5/12 E^3 N_{n-2}),
    with E = exp(Lambda dt) the damping of one step. The first two steps use Euler
    and AB2 the same way.

    Args:
        dynamics: Fixed operators and time step.
        state: Spectral state (B, 3, L, L).
        previous_tendencies: Tendencies of earlier steps, most recent first (at
            most two).

    Returns:
        The new state and the updated list of previous tendencies.
    """
    dt = dynamics.dt
    E = dynamics.step_damping  # (3, L, L)
    tendency = shallow_water_tendencies(dynamics, state)

    if len(previous_tendencies) == 0:
        increment = E * tendency
    elif len(previous_tendencies) == 1:
        increment = 1.5 * E * tendency - 0.5 * E**2 * previous_tendencies[0]
    else:
        increment = (
            23 * E * tendency
            - 16 * E**2 * previous_tendencies[0]
            + 5 * E**3 * previous_tendencies[1]
        ) / 12

    new_state = E * state + dt * increment
    new_previous_tendencies = [tendency, *previous_tendencies[:1]]
    return new_state, new_previous_tendencies


def spectral_to_fields(
    dynamics: SphereShallowWaterDynamics, state: TensorLike
) -> TensorLike:
    """Spectral state -> h, u_phi, u_theta on the grid, shape (B, *grid.shape, 3).

    h is the height perturbation Phi / g - mean_depth [m], u_phi the eastward and
    u_theta the colatitude (southward) wind [m/s].
    """
    geopotential_grid = dynamics.to_grid(state[:, 0])
    wind = dynamics.wind(state[:, 1], state[:, 2])
    h = geopotential_grid / dynamics.gravity - dynamics.mean_depth
    return torch.stack([h, wind[:, 0], -wind[:, 1]], dim=-1)
