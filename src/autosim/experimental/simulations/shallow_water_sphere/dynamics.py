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
        sht_iterations: Jacobi iterations refining the analysis (grid values ->
            coefficients). 0 is exact on grids with an exact quadrature; HEALPix has
            none, and 3 iterations bring its analysis error from about 1e-2 to about
            1e-6 (torch-harmonics documentation) at four times the cost.
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
        sht_iterations: int = 0,
    ) -> None:
        """Build the transforms, the Laplacian, Coriolis and the damping."""
        if th is None:
            raise ImportError(TORCH_HARMONICS_MISSING)
        self.grid = grid
        self.lmax = lmax
        self.radius = radius
        self.omega = omega
        self.gravity = gravity
        self.mean_depth = mean_depth
        self.dt = dt

        # The product of two fields of degree <= L - 1 has degree <= 2L - 2. Projecting
        # it onto degree <= L - 1 integrates a polynomial of degree <= 3L - 3, and a
        # quadrature with exact_degree = n is exact up to degree 2n - 1. The projection
        # is therefore exact (free of aliasing) when 2n >= 3L - 2: T85 (L = 86) on 128
        # Gauss-Legendre rings. Checked numerically against a grid twice as fine.
        try:
            exact_degree = grid.max_exact_degree
        except NotImplementedError:  # HEALPix: no exact quadrature
            exact_degree = 0
        self.exact_products = 2 * exact_degree >= 3 * lmax - 2

        # Transforms between the grid and spherical-harmonic coefficients
        self.sht = th.RealSHT(
            grid, lmax=lmax, mmax=lmax, csphase=False, iter=sht_iterations
        ).to(device)
        self.isht = th.InverseRealSHT(grid, lmax=lmax, mmax=lmax, csphase=False).to(
            device
        )
        self.vsht = th.RealVectorSHT(
            grid, lmax=lmax, mmax=lmax, csphase=False, iter=sht_iterations
        ).to(device)
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

        # Latitude, longitude and quadrature weight (summing to 4 pi) of every point
        self.latitude = latitude.reshape(grid.shape)
        self.longitude = (
            grid.coords[:, 1].to(device=device, dtype=torch.float64).reshape(grid.shape)
        )
        self.quad_weights = grid.quad_weights.to(
            device=device, dtype=torch.float64
        ).reshape(grid.shape)

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


def coriolis_parameter(
    dynamics: SphereShallowWaterDynamics, axis_lon: TensorLike, axis_lat: TensorLike
) -> TensorLike:
    r"""Coriolis parameter for a rotation axis of any direction.

    :math:`f = 2\Omega\,(\hat{\mathbf{n}}\cdot\hat{\mathbf{r}})`, with
    :math:`\hat{\mathbf{n}}` the unit vector of the rotation pole and
    :math:`\hat{\mathbf{r}}` the local vertical. The geographic pole
    (``axis_lat = pi/2``) gives :math:`2\Omega\sin\varphi`. The pole at longitude
    :math:`\pi` and latitude :math:`\pi/2 - \alpha` gives W92 eq. (96).

    Args:
        dynamics: Fixed operators.
        axis_lon: Longitude of the rotation pole [rad], shape (B,).
        axis_lat: Latitude of the rotation pole [rad], shape (B,).

    Returns:
        f on the grid [1/s], shape (B, *grid.shape).
    """
    latitude = dynamics.latitude.unsqueeze(0)
    longitude = dynamics.longitude.unsqueeze(0)
    parameter_shape = (axis_lon.shape[0],) + (1,) * len(dynamics.grid.shape)
    axis_lon = axis_lon.to(latitude.device, torch.float64).reshape(parameter_shape)
    axis_lat = axis_lat.to(latitude.device, torch.float64).reshape(parameter_shape)

    # n . r, the cosine of the angle between the rotation pole and the grid point
    cosine = torch.sin(latitude) * torch.sin(axis_lat) + torch.cos(
        latitude
    ) * torch.cos(axis_lat) * torch.cos(longitude - axis_lon)
    return 2 * dynamics.omega * cosine


def shallow_water_tendencies(
    dynamics: SphereShallowWaterDynamics,
    state: TensorLike,
    coriolis: TensorLike | None = None,
    surface_geopotential: TensorLike | None = None,
) -> TensorLike:
    r"""Time derivatives of the (geopotential, vorticity, divergence) coefficients.

    The geopotential of the state is :math:`\Phi = g h^*`, with :math:`h^*` the fluid
    depth. Over an orography :math:`h_s` the free surface is :math:`h^* + h_s`, and
    the pressure gradient acts on it, so :math:`g h_s` is added to the Bernoulli
    function:

    .. math::

        \partial_t \Phi = -\nabla\cdot(\Phi\,\mathbf{u}), \qquad
        \partial_t \zeta = -\nabla\cdot\big((\zeta+f)\,\mathbf{u}\big), \qquad
        \partial_t \delta = \hat{\mathbf{k}}\cdot\nabla\times
            \big((\zeta+f)\,\mathbf{u}\big)
            - \nabla^2\big(\Phi + g h_s + \tfrac12|\mathbf{u}|^2\big).

    Args:
        dynamics: Fixed operators.
        state: Spectral state (B, 3, L, L).
        coriolis: Coriolis parameter on the grid, (B, *grid.shape). None is the
            polar one, ``dynamics.coriolis``.
        surface_geopotential: Coefficients of :math:`g h_s`, (B, L, L). None is a flat
            bottom.

    Returns:
        Tendencies (B, 3, L, L).
    """
    if coriolis is None:
        coriolis = dynamics.coriolis
    geopotential = state[:, 0]
    vorticity = state[:, 1]
    divergence = state[:, 2]

    # 1. Fields on the grid
    geopotential_grid = dynamics.to_grid(geopotential)
    vorticity_grid = dynamics.to_grid(vorticity)
    wind = dynamics.wind(vorticity, divergence)

    # 2. Products on the grid
    absolute_vorticity = vorticity_grid + coriolis
    vorticity_flux = wind * absolute_vorticity.unsqueeze(1)  # (zeta + f) u
    geopotential_flux = wind * geopotential_grid.unsqueeze(1)  # Phi u
    kinetic_energy = 0.5 * (wind[:, 0] ** 2 + wind[:, 1] ** 2)
    bernoulli = geopotential_grid + kinetic_energy  # Phi + |u|^2 / 2

    # 3. Back to coefficients
    flux_curl, flux_divergence = dynamics.vorticity_divergence(vorticity_flux)
    _, geopotential_flux_divergence = dynamics.vorticity_divergence(geopotential_flux)
    bernoulli_coefficients = dynamics.to_coefficients(bernoulli)
    if surface_geopotential is not None:
        bernoulli_coefficients = bernoulli_coefficients + surface_geopotential

    d_geopotential = -geopotential_flux_divergence
    d_vorticity = -flux_divergence
    d_divergence = flux_curl - dynamics.laplacian * bernoulli_coefficients
    return torch.stack([d_geopotential, d_vorticity, d_divergence], dim=1)


def adams_bashforth_step(
    dynamics: SphereShallowWaterDynamics,
    state: TensorLike,
    previous_tendencies: list[TensorLike],
    coriolis: TensorLike | None = None,
    surface_geopotential: TensorLike | None = None,
    step_damping: TensorLike | None = None,
    equilibrium: TensorLike | None = None,
) -> tuple[TensorLike, list[TensorLike]]:
    """One Adams-Bashforth 3 step with the linear damping as an integrating factor.

    s_{n+1} = E s_n + dt (23/12 E N_n - 16/12 E^2 N_{n-1} + 5/12 E^3 N_{n-2}),
    with E = exp(Lambda dt) the damping of one step. The first two steps use Euler
    and AB2 the same way. When the damping relaxes the state towards an equilibrium
    s_eq, it acts on the departure from it: E s_n becomes s_eq + E (s_n - s_eq).

    Args:
        dynamics: Fixed operators and time step.
        state: Spectral state (B, 3, L, L).
        previous_tendencies: Tendencies of earlier steps, most recent first (at
            most two).
        coriolis: Coriolis parameter (B, *grid.shape); None is the polar one.
        surface_geopotential: Coefficients of g h_s, (B, L, L); None is a flat bottom.
        step_damping: Damping E of one step for each simulation, (B, 3, L, L), see
            ``step_damping_factors``. None is the hyperdiffusion alone,
            ``dynamics.step_damping``.
        equilibrium: State towards which the damping relaxes, (B, 3, L, L). None is
            a damping towards zero.

    Returns:
        The new state and the updated list of previous tendencies.
    """
    dt = dynamics.dt
    E = dynamics.step_damping if step_damping is None else step_damping
    tendency = shallow_water_tendencies(dynamics, state, coriolis, surface_geopotential)

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
    if equilibrium is not None:
        new_state = new_state + (1 - E) * equilibrium
    new_previous_tendencies = [tendency, *previous_tendencies[:1]]
    return new_state, new_previous_tendencies


def step_damping_factors(
    dynamics: SphereShallowWaterDynamics,
    drag_time: TensorLike,
    relaxation_time: TensorLike,
) -> TensorLike:
    r"""Damping of one step with linear drag and Newtonian relaxation.

    The vorticity and the divergence are damped by the hyperdiffusion and by a linear
    drag, :math:`\partial_t \zeta = -\zeta / \tau_\text{drag}` (the same for
    :math:`\delta`). The geopotential relaxes towards an equilibrium,
    :math:`\partial_t \Phi = -(\Phi - \Phi_\text{eq}) / \tau_\text{relax}`. Over one
    step these are the factors :math:`\exp(-\Delta t / \tau)`, exact for any
    :math:`\Delta t`.

    Args:
        dynamics: Fixed operators and time step.
        drag_time: Drag time [s], shape (B,); infinity is no drag.
        relaxation_time: Relaxation time [s], shape (B,); infinity is no relaxation.

    Returns:
        Factors E of (geopotential, vorticity, divergence), shape (B, 3, L, L).
    """
    device = dynamics.step_damping.device
    drag = torch.exp(-dynamics.dt / drag_time.to(device, torch.float64))
    relaxation = torch.exp(-dynamics.dt / relaxation_time.to(device, torch.float64))
    batch = drag.shape[0]
    E = dynamics.step_damping.unsqueeze(0).repeat(batch, 1, 1, 1)
    E[:, 0] = E[:, 0] * relaxation.reshape(batch, 1, 1)
    E[:, 1] = E[:, 1] * drag.reshape(batch, 1, 1)
    E[:, 2] = E[:, 2] * drag.reshape(batch, 1, 1)
    return E


def total_energy(
    dynamics: SphereShallowWaterDynamics,
    state: TensorLike,
    surface_geopotential: TensorLike | None = None,
) -> TensorLike:
    r"""Energy of the layer per unit area, global mean.

    .. math::

        E = \frac{1}{4\pi}\int \Big( \tfrac12 h^* |\mathbf{u}|^2
            + \tfrac12 g\,(h - H)^2 \Big)\, d\Omega,

    with :math:`h^* = \Phi / g` the fluid depth, :math:`h = h^* + h_s` the free surface
    and :math:`H` the mean depth: the kinetic energy weighted by the depth, plus the
    potential energy of the displaced free surface. This is the energy of the planar
    simulator. It differs from the invariant of ``diagnostics.invariants`` by terms
    that are constant when the mass is conserved.

    Args:
        dynamics: Fixed operators.
        state: Spectral state (B, 3, L, L).
        surface_geopotential: Coefficients of g h_s, (B, L, L); None is a flat bottom.

    Returns:
        Energy [m^3/s^2], shape (B,).
    """
    depth = dynamics.to_grid(state[:, 0]) / dynamics.gravity
    wind = dynamics.wind(state[:, 1], state[:, 2])
    speed_squared = wind[:, 0] ** 2 + wind[:, 1] ** 2
    free_surface = depth
    if surface_geopotential is not None:
        free_surface = depth + dynamics.to_grid(surface_geopotential) / dynamics.gravity
    displacement = free_surface - dynamics.mean_depth
    energy = 0.5 * depth * speed_squared + 0.5 * dynamics.gravity * displacement**2

    grid_dims = tuple(range(-dynamics.quad_weights.ndim, 0))
    return (energy * dynamics.quad_weights).sum(dim=grid_dims) / (4 * torch.pi)


def spectral_to_fields(
    dynamics: SphereShallowWaterDynamics,
    state: TensorLike,
    surface_geopotential: TensorLike | None = None,
) -> TensorLike:
    """Spectral state -> h, u_phi, u_theta on the grid, shape (B, *grid.shape, 3).

    h is the height of the free surface above the mean depth,
    (Phi + g h_s) / g - mean_depth [m], which is Phi / g - mean_depth over a flat
    bottom. u_phi is the eastward and u_theta the colatitude (southward) wind [m/s].

    Args:
        dynamics: Fixed operators.
        state: Spectral state (B, 3, L, L).
        surface_geopotential: Coefficients of g h_s, (B, L, L); None is a flat bottom.

    Returns:
        Fields (B, *grid.shape, 3).
    """
    geopotential = state[:, 0]
    if surface_geopotential is not None:
        geopotential = geopotential + surface_geopotential
    geopotential_grid = dynamics.to_grid(geopotential)
    wind = dynamics.wind(state[:, 1], state[:, 2])
    h = geopotential_grid / dynamics.gravity - dynamics.mean_depth
    return torch.stack([h, wind[:, 0], -wind[:, 1]], dim=-1)


def increment_to_fields(
    dynamics: SphereShallowWaterDynamics, increment: TensorLike
) -> TensorLike:
    """Increment of the spectral state -> increments of h, u_phi, u_theta.

    The same conversion as ``spectral_to_fields`` for a difference of two states: no
    mean depth is removed.

    Args:
        dynamics: Fixed operators.
        increment: Difference of two spectral states, (B, 3, L, L).

    Returns:
        Increments of the fields, (B, *grid.shape, 3).
    """
    height = dynamics.to_grid(increment[:, 0]) / dynamics.gravity
    wind = dynamics.wind(increment[:, 1], increment[:, 2])
    return torch.stack([height, wind[:, 0], -wind[:, 1]], dim=-1)


def fields_to_spectral(
    dynamics: SphereShallowWaterDynamics,
    fields: TensorLike,
    surface_geopotential: TensorLike | None = None,
) -> TensorLike:
    """Saved fields h, u_phi, u_theta -> spectral state: inverse of spectral_to_fields.

    The result keeps the degrees below the band limit of ``dynamics``. On a grid with
    an exact quadrature, fields produced by ``spectral_to_fields`` give back their
    state to round-off (of the field precision).

    Args:
        dynamics: Fixed operators.
        fields: Fields (B, *grid.shape, 3) in the order h, u_phi, u_theta.
        surface_geopotential: Coefficients of g h_s, (B, L, L); None is a flat bottom.

    Returns:
        Spectral state (B, 3, L, L).
    """
    fields = fields.to(dynamics.coriolis.device, torch.float64)
    h = fields[..., 0]
    eastward = fields[..., 1]
    northward = -fields[..., 2]

    geopotential_grid = dynamics.gravity * (h + dynamics.mean_depth)
    geopotential = dynamics.to_coefficients(geopotential_grid)
    if surface_geopotential is not None:
        geopotential = geopotential - surface_geopotential
    wind = torch.stack([eastward, northward], dim=1)
    vorticity, divergence = dynamics.vorticity_divergence(wind)

    state = torch.stack([geopotential, vorticity, divergence], dim=1)
    return torch.tril(state)  # keep m <= l
