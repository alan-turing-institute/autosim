r"""Shallow water equations on the rotating sphere, spectral method (torch-harmonics).

The state of ``B`` simulations is one complex tensor of shape ``(B, 3, L, L)``:
index 0 is the geopotential :math:`\Phi = g h`, 1 the relative vorticity
:math:`\zeta` and 2 the divergence :math:`\delta`, all as spherical-harmonic
coefficients with :math:`m \le l < L`. The grid is only used to form the nonlinear
products:

.. math::

    \partial_t \Phi = -\nabla\cdot(\Phi\,\mathbf{u}), \qquad
    \partial_t \zeta = -\nabla\cdot\big((\zeta+f)\,\mathbf{u}\big), \qquad
    \partial_t \delta = \hat{\mathbf{k}}\cdot\nabla\times\big((\zeta+f)\,\mathbf{u}\big)
        - \nabla^2\big(\Phi + \tfrac12|\mathbf{u}|^2\big).

Time stepping is third-order Adams-Bashforth with the hyperdiffusion as an integrating
factor. The initial condition is the barotropically unstable jet of Galewsky, Scott and
Polvani (2004, Tellus A 56, 429-440), with sampled parameters and a small height
noise that seeds the chaos. The tendencies equal torch-harmonics'
``ShallowWaterSolver.dudtspec``.

Requires the optional ``spherical`` extra (torch-harmonics with grid descriptors).
"""

from __future__ import annotations

import math

import torch

from autosim.simulations.base import SpatioTemporalSimulator
from autosim.types import TensorLike

try:
    import torch_harmonics as th
except ImportError:  # optional dependency: the "spherical" extra
    th = None

GALEWSKY_RANGES: dict[str, tuple[float, float]] = {
    "jet_speed": (60.0, 100.0),  # u_max [m/s], paper: 80
    "height_perturbation_m": (60.0, 180.0),  # bump amplitude [m], paper: 120
    "perturbation_lon": (0.0, 2 * math.pi),  # bump longitude [rad], paper: 0
    "perturbation_lat": (math.pi / 5, math.pi / 3),  # bump latitude [rad], paper: pi/4
}


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


def galewsky_initial_condition(
    dynamics: SphereShallowWaterDynamics,
    jet_speed: TensorLike,
    height_perturbation: TensorLike,
    perturbation_lon: TensorLike,
    perturbation_lat: TensorLike,
    noise_amplitude: float,
    noise_lmax: int,
    device: str = "cuda",
) -> TensorLike:
    """Galewsky et al. (2004) jet, bump and height noise for B simulations.

    Jet (their eq. 2), balanced height with a global mean depth ``mean_depth``
    (eq. 3, computed spectrally from the steady divergence equation), bump
    h' = h_hat cos(phi) exp(-(dlon/alpha)^2) exp(-((phi_2 - phi)/beta)^2) (eq. 4) with
    alpha = 1/3, beta = 1/15, and a random height noise below degree ``noise_lmax``
    drawn from the global torch generator.

    Args:
        dynamics: Fixed operators.
        jet_speed: Maximum jet speed u_max [m/s], shape (B,).
        height_perturbation: Bump amplitude h_hat [m], shape (B,).
        perturbation_lon: Bump centre longitude [rad], shape (B,).
        perturbation_lat: Bump centre latitude phi_2 [rad], shape (B,).
        noise_amplitude: Root-mean-square of the height noise [m].
        noise_lmax: The noise has degrees and orders below this value.
        device: Torch device of the state.

    Returns:
        Spectral state (B, 3, L, L).
    """
    batch = jet_speed.shape[0]
    lmax = dynamics.laplacian.shape[0]
    gravity = dynamics.gravity

    # Latitude and longitude of every grid point, shape (1, *grid.shape)
    grid_shape = dynamics.grid.shape
    coords = dynamics.grid.coords.to(device=device, dtype=torch.float64)
    latitude = (torch.pi / 2 - coords[:, 0]).reshape(1, *grid_shape)
    longitude = coords[:, 1].reshape(1, *grid_shape)

    # Parameters as shape (B, 1, ...) so they broadcast over the grid
    parameter_shape = (batch,) + (1,) * len(grid_shape)
    jet_speed = jet_speed.to(device, torch.float64).reshape(parameter_shape)
    height_perturbation = height_perturbation.to(device, torch.float64).reshape(
        parameter_shape
    )
    perturbation_lon = perturbation_lon.to(device, torch.float64).reshape(
        parameter_shape
    )
    perturbation_lat = perturbation_lat.to(device, torch.float64).reshape(
        parameter_shape
    )

    # 1. Zonal jet, eq. (2)
    phi0 = torch.pi / 7
    phi1 = torch.pi / 2 - phi0
    en = math.exp(-4 / (phi1 - phi0) ** 2)
    inside = (latitude > phi0) & (latitude < phi1)
    product = torch.where(inside, (latitude - phi0) * (latitude - phi1), -1.0)
    u = torch.where(inside, jet_speed / en * torch.exp(1.0 / product), 0.0)
    wind = torch.stack([u, torch.zeros_like(u)], dim=1)
    vorticity, divergence = dynamics.vorticity_divergence(wind)

    # 2. Balanced geopotential, eq. (3): lap^-1 curl((zeta + f) u) - |u|^2 / 2 + const
    absolute_vorticity = dynamics.to_grid(vorticity) + dynamics.coriolis
    flux_curl, _ = dynamics.vorticity_divergence(wind * absolute_vorticity.unsqueeze(1))
    geopotential = dynamics.inverse_laplacian * flux_curl - dynamics.to_coefficients(
        0.5 * u**2
    )
    geopotential[:, 0, 0] = math.sqrt(4 * math.pi) * gravity * dynamics.mean_depth

    # 3. Bump, eq. (4)
    alpha = 1 / 3
    beta = 1 / 15
    delta_lon = (
        torch.remainder(longitude - perturbation_lon + torch.pi, 2 * torch.pi)
        - torch.pi
    )
    bump = (
        height_perturbation
        * torch.cos(latitude)
        * torch.exp(-((delta_lon / alpha) ** 2))
        * torch.exp(-(((perturbation_lat - latitude) / beta) ** 2))
    )
    geopotential = geopotential + gravity * dynamics.to_coefficients(bump)

    # 4. Height noise below degree noise_lmax (global RNG, on CPU)
    k = min(noise_lmax, lmax)
    noise = torch.randn(batch, k, k, dtype=torch.complex128)
    scale = gravity * noise_amplitude * math.sqrt(4 * math.pi / (k * (k + 1)))
    geopotential[:, :k, :k] += scale * noise.to(device)

    state = torch.stack([geopotential, vorticity, divergence], dim=1)
    return torch.tril(state)  # keep m <= l


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


class SphericalShallowWater(SpatioTemporalSimulator):
    """Shallow water on the rotating sphere from a Galewsky jet (torch-harmonics).

    Outputs ``[h, u_phi, u_theta]`` on the simulation grid, with data of shape
    ``(n, time, *grid.shape, 3)``: ``(nlat, nlon)`` on regular grids and ``(npix,)``
    on HEALPix. Simulations run ``batch_size`` at a time in parallel on ``device``.

    Args:
        lmax: Spectral band limit L.
        grid: "legendre-gauss", "equiangular", "lobatto" or "healpix".
        nlat: Rings, for regular grids.
        nlon: Points per ring, for regular grids.
        nside: Resolution, for "healpix".
        parameters_range: Must contain the four keys of ``GALEWSKY_RANGES``; equal
            bounds fix a parameter.
        output_names: Channel names, default ``["h", "u_phi", "u_theta"]``.
        log_level: Logging verbosity.
        T: Total simulated time [hours].
        dt_save: Time between snapshots [hours].
        skip_nt: Number of leading snapshots dropped (spin-up).
        courant: Fraction of the gravity-wave stability limit used for the step.
        radius: Planet radius [m].
        omega: Rotation rate [1/s].
        gravity: Gravitational acceleration [m/s^2].
        mean_depth: Global mean depth [m].
        noise_amplitude: Root-mean-square of the seeding height noise [m].
        noise_lmax: The noise has degrees below this value.
        batch_size: Simulations run in parallel on the device.
        device: Torch device.
    """

    def __init__(
        self,
        lmax: int = 256,
        grid: str = "legendre-gauss",
        nlat: int | None = 384,
        nlon: int | None = 768,
        nside: int | None = None,
        parameters_range: dict[str, tuple[float, float]] | None = None,
        output_names: list[str] | None = None,
        log_level: str = "progress_bar",
        T: float = 360.0,
        dt_save: float = 3.0,
        skip_nt: int = 61,
        courant: float = 0.4,
        radius: float = 6.37122e6,
        omega: float = 7.292e-5,
        gravity: float = 9.80616,
        mean_depth: float = 1.0e4,
        noise_amplitude: float = 1e-3,
        noise_lmax: int = 120,
        batch_size: int = 8,
        device: str = "cuda",
    ) -> None:
        """Store the configuration and build the spectral operators once."""
        if th is None:
            msg = "SphericalShallowWater requires torch-harmonics (extra 'spherical')."
            raise ImportError(msg)
        if parameters_range is None:
            parameters_range = dict(GALEWSKY_RANGES)
        if output_names is None:
            output_names = ["h", "u_phi", "u_theta"]
        super().__init__(parameters_range, output_names, log_level)

        if grid == "healpix":
            self.grid = th.as_grid("healpix", nside=nside)
        else:
            self.grid = th.as_grid(grid, nlat=nlat, nlon=nlon)

        self.lmax = lmax
        self.T = T
        self.dt_save = dt_save
        self.skip_nt = skip_nt
        self.courant = courant
        self.noise_amplitude = noise_amplitude
        self.noise_lmax = noise_lmax
        self.batch_size = batch_size
        self.device = device

        # Courant step, adjusted so that snapshots land exactly on dt_save
        wave_speed = math.sqrt(gravity * mean_depth)
        max_wind = 80.0
        dt = courant * (radius / lmax) / (wave_speed + max_wind)
        self.steps_per_save = math.ceil(dt_save * 3600 / dt)
        self.dt = dt_save * 3600 / self.steps_per_save
        self.n_frames = round(T / dt_save) + 1 - skip_nt  # saved snapshots
        if self.n_frames < 1:
            msg = (
                f"skip_nt={skip_nt} leaves no snapshot for T={T} and dt_save={dt_save}."
            )
            raise ValueError(msg)

        self.dynamics = SphereShallowWaterDynamics(
            self.grid, lmax, radius, omega, gravity, mean_depth, self.dt, device
        )

    def forward_batch(
        self,
        x: TensorLike,
        allow_failures: bool = True,
        *,
        num_workers: int | None = None,
        random_seed: int | None = None,
    ) -> tuple[TensorLike, TensorLike]:
        """Run one simulation per row of x, ``batch_size`` of them at a time.

        Args:
            x: Parameter rows (n, 4), columns ordered as ``param_names``.
            allow_failures: Kept for the autosim interface; non-finite simulations
                are always dropped.
            num_workers: Not supported: use ``batch_size``.
            random_seed: Optional seed of the global torch generator (noise).

        Returns:
            (y, x_valid) with y of shape (n_valid, n_frames * npoints * 3).
        """
        del allow_failures
        if num_workers is not None:
            msg = "SphericalShallowWater runs simulations in parallel with batch_size."
            raise ValueError(msg)
        if random_seed is not None:
            torch.manual_seed(random_seed)

        results = []
        valid_rows = []
        for start in range(0, x.shape[0], self.batch_size):
            rows = x[start : start + self.batch_size]
            state = galewsky_initial_condition(
                self.dynamics,
                jet_speed=rows[:, self.get_parameter_idx("jet_speed")],
                height_perturbation=rows[
                    :, self.get_parameter_idx("height_perturbation_m")
                ],
                perturbation_lon=rows[:, self.get_parameter_idx("perturbation_lon")],
                perturbation_lat=rows[:, self.get_parameter_idx("perturbation_lat")],
                noise_amplitude=self.noise_amplitude,
                noise_lmax=self.noise_lmax,
                device=self.device,
            )
            frames = []
            if self.skip_nt == 0:
                frames.append(spectral_to_fields(self.dynamics, state).float().cpu())
            previous_tendencies: list[TensorLike] = []
            for save in range(1, round(self.T / self.dt_save) + 1):
                for _ in range(self.steps_per_save):
                    state, previous_tendencies = adams_bashforth_step(
                        self.dynamics, state, previous_tendencies
                    )
                if save >= self.skip_nt:
                    fields = spectral_to_fields(self.dynamics, state)
                    frames.append(fields.float().cpu())

            trajectories = torch.stack(frames, dim=1)  # (b, n_frames, *grid.shape, 3)
            finite = torch.isfinite(trajectories.flatten(start_dim=1)).all(dim=1)
            results.append(trajectories[finite].flatten(start_dim=1))
            valid_rows.append(rows[finite])

        return torch.cat(results), torch.cat(valid_rows)

    def _forward(self, x: TensorLike) -> TensorLike | None:
        """One simulation (the autosim single-sample contract)."""
        y, _ = self.forward_batch(x)
        if y.shape[0] == 0:
            return None
        return y

    def forward_samples_spatiotemporal(
        self,
        n: int,
        random_seed: int | None = None,
        ensure_exact_n: bool = False,
    ) -> dict:
        """Run n simulations and return them in the autosim spatiotemporal format.

        Args:
            n: Number of simulations.
            random_seed: Seed of the parameter sample and of the noise.
            ensure_exact_n: Replace failed simulations until n succeed.

        Returns:
            ``{"data": (n, time, *grid.shape, 3), "constant_scalars": (n, 4),
            "constant_fields": None}``.
        """
        y, x = self._forward_batch_with_optional_retries(
            n=n, random_seed=random_seed, ensure_exact_n=ensure_exact_n
        )
        data = y.reshape(
            y.shape[0], self.n_frames, *self.grid.shape, len(self.output_names)
        )
        return {"data": data, "constant_scalars": x, "constant_fields": None}
