"""The spherical shallow water simulator (autosim interface)."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import torch

from autosim.simulations.base import SpatioTemporalSimulator
from autosim.types import TensorLike

from .dynamics import (
    SphereShallowWaterDynamics,
    coriolis_parameter,
    fields_to_spectral,
    make_grid,
)
from .forcing import ForcingSettings
from .initial_conditions import (
    GALEWSKY_RANGES,
    GALEWSKY_UNPERTURBED_RANGES,
    MOUNTAIN_RANGES,
    RANDOM_BALANCED_RANGES,
    ROSSBY_HAURWITZ_RANGES,
    WILLIAMSON2_RANGES,
    balanced_geopotential,
    galewsky_initial_condition,
    mountain_initial_condition,
    random_balanced_initial_condition,
    rossby_haurwitz_initial_condition,
    williamson2_initial_condition,
    zonal_jet,
)
from .integration import ENERGY_BUDGET_NAMES, integrate_trajectories

INITIAL_CONDITIONS = (
    "galewsky",
    "galewsky_unperturbed",
    "williamson2",
    "rossby_haurwitz",
    "mountain",
    "random_balanced",
    "restart",
    "from_fields",
)
FORCING_TYPES = ("none", "vortical", "momentum", "mass_pulses")
BACKSCATTER_MODES = ("global", "local")
BRANCH_FORCING_STATES = ("redraw", "carry")

# Default sampling ranges of each family; their keys are the required parameters.
DEFAULT_PARAMETER_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "galewsky": GALEWSKY_RANGES,
    "galewsky_unperturbed": GALEWSKY_UNPERTURBED_RANGES,
    "williamson2": WILLIAMSON2_RANGES,
    "rossby_haurwitz": ROSSBY_HAURWITZ_RANGES,
    "mountain": MOUNTAIN_RANGES,
    "random_balanced": RANDOM_BALANCED_RANGES,
    "restart": {},  # the state is supplied: no sampled parameter
    "from_fields": {},
}

# Constructor arguments that can also be keys of parameters_range: they are then
# sampled for each simulation and returned in constant_scalars.
SAMPLED_SCALARS = (
    "relaxation_time",
    "relaxation_jet_speed",
    "drag_time",
    "forcing_energy_rate",
    "forcing_correlation_time",
    "forcing_spectral_slope",
    "forcing_divergent_fraction",
    "forcing_backscatter_fraction",
    "pulse_rate",
    "pulse_amplitude",
    "pulse_width",
)

# Independent random streams of one simulation
INITIAL_NOISE_STREAM = 0
FORCING_STREAM = 1
BASELINE_STREAM = 2
REFERENCE_STREAM = 100  # member j of a reference ensemble uses stream 100 + j


def sample_generators(
    entropy: int, rows: list[int], stream: int, device: str
) -> list[torch.Generator]:
    """One random generator per simulation, seeded from (entropy, row, stream).

    The seed of a simulation depends on its row in the batch of parameters and not on
    ``batch_size``, so its random draws are the same however the rows are grouped.
    Different streams of the same simulation are independent.

    Args:
        entropy: Seed of the whole call.
        rows: Row index of each simulation in the parameter tensor.
        stream: Purpose of the generators (initial noise, forcing, ...).
        device: Torch device of the generators.

    Returns:
        One generator per row.
    """
    generators = []
    for row in rows:
        seed_sequence = np.random.SeedSequence([entropy, row, stream])
        seed = int(seed_sequence.generate_state(1, np.uint64)[0])
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        generators.append(generator)
    return generators


class SphericalShallowWater(SpatioTemporalSimulator):
    """Shallow water on the rotating sphere (torch-harmonics).

    Outputs ``[h, u_phi, u_theta]`` on the simulation grid, with data of shape
    ``(n, time, *grid.shape, 3)``: ``(nlat, nlon)`` on regular grids and ``(npix,)``
    on HEALPix. ``h`` is the height of the free surface above ``mean_depth``.
    Simulations run ``batch_size`` at a time in parallel on ``device``.

    The initial condition is one of these families, each with its own parameters in
    ``parameters_range`` (see the ``*_RANGES`` dictionaries of ``initial_conditions``):

    - ``"galewsky"``: barotropically unstable jet with a height bump and a small
      height noise (Galewsky et al. 2004). Parameters ``jet_speed``,
      ``height_perturbation_m``, ``perturbation_lon``, ``perturbation_lat``.
    - ``"galewsky_unperturbed"``: the same jet alone, which stays zonal. Parameter
      ``jet_speed``.
    - ``"williamson2"``: steady zonal flow about an axis tilted by ``tilt``, with the
      Coriolis parameter of the same axis (W92 case 2). Parameters ``flow_speed``,
      ``tilt``.
    - ``"rossby_haurwitz"``: Rossby-Haurwitz wave (W92 case 6). Parameters
      ``wavenumber`` (rounded to the nearest integer; the rounded value is returned
      in ``constant_scalars``), ``wave_amplitude``, ``mean_depth``.
    - ``"mountain"``: zonal flow over an isolated mountain (W92 case 5). Parameters
      ``flow_speed``, ``mountain_height``, ``mountain_radius``, ``mountain_lon``,
      ``mountain_lat``.
    - ``"random_balanced"``: random flow without divergence and its balanced height,
      the spherical counterpart of the planar ``balanced_random_pv``. Parameters
      ``amp`` (root-mean-square speed), ``initial_wavenumber``, ``initial_bandwidth``.

    - ``"restart"``: the state ``initial_state``, saved fields ``[h, u_phi,
      u_theta]`` of this or another run. No parameter. All the simulations of a call
      start from this one state, so with stochastic forcing they are the members of
      an ensemble forecast, as in the planar simulator.
    - ``"from_fields"``: the same after a balance step, which removes the divergence
      and replaces the height by the one in nonlinear balance with the wind. This is
      the interface of W92 case 7 for analysed fields.

    The arguments listed in ``SAMPLED_SCALARS`` (drag and relaxation times, forcing
    rate, decorrelation time, slope, fractions, pulse statistics) can also be keys of
    ``parameters_range``. They are then sampled for each simulation, and the
    constructor value is not used.

    Three physical processes can be added to the conservative dynamics (module
    ``integration`` gives their order in a step, module ``forcing`` the definitions):

    - Newtonian relaxation of the height towards the balanced height of a zonal jet,
      and linear drag on the vorticity and the divergence. With both, the flow
      reaches a statistically steady state with a jet and eddies.
    - Stochastic forcing, ``forcing_type``: ``"vortical"`` (streamfunction pattern,
      Berner et al. 2009), ``"momentum"`` (the same with a divergent part) or
      ``"mass_pulses"`` (Gaussian height bumps at Poisson times). The energy rate of
      the first two can include a backscatter of the diagnosed dissipation, global
      or local.

    Args:
        lmax: Spectral band limit L.
        grid: "legendre-gauss", "equiangular", "lobatto" or "healpix".
        nlat: Rings, for regular grids.
        nlon: Points per ring, for regular grids.
        nside: Resolution, for "healpix".
        parameters_range: Bounds of the parameters of the family; equal bounds fix a
            parameter. None takes the default ranges of the family.
        output_names: Channel names, default ``["h", "u_phi", "u_theta"]``.
        log_level: Logging verbosity.
        T: Total simulated time [hours].
        dt_save: Time between snapshots [hours].
        skip_nt: Number of leading snapshots dropped (spin-up).
        courant: Fraction of the gravity-wave stability limit used for the step.
        radius: Planet radius [m].
        omega: Rotation rate [1/s].
        gravity: Gravitational acceleration [m/s^2].
        mean_depth: Reference depth [m]: the global mean depth of the Galewsky jet,
            h_0 of W92 cases 2 and 5, and the level from which the output ``h`` is
            measured. It also gives the gravity-wave speed of the time step.
        noise_amplitude: Root-mean-square of the seeding height noise of
            ``"galewsky"`` [m]. Two simulations with the same parameters differ
            only by this noise. The default of 10 m is the smallest of the tested
            values that separates them by the end of the 183 h spin-up: at lmax = 86
            the distance between their eddy winds (1 when decorrelated) is then 0.80
            with 10 m, 0.37 with 1 m and 0.00 with 1 mm.
        noise_lmax: The noise has degrees below this value.
        batch_size: Simulations run in parallel on the device.
        device: Torch device.
        initial_condition: Family of the initial condition, see above.
        rotation_axis: Longitude and latitude [rad] of the rotation pole, for the
            Coriolis parameter. None is the geographic pole. ``"williamson2"`` sets
            the axis from its tilt.
        return_constant_fields: Return the Coriolis parameter f [1/s] and the
            orography h_s [m] on the grid as ``constant_fields``, shape
            ``(n, *grid.shape, 2)``. An emulator needs them as inputs: a rotated
            problem has a rotated f.
        sht_iterations: Jacobi iterations refining the analysis on grids without an
            exact quadrature (HEALPix), see ``SphereShallowWaterDynamics``.
        relaxation_time: Time of the relaxation of the height towards its
            equilibrium [hours]. None is no relaxation.
        relaxation_jet_speed: Maximum speed [m/s] of the zonal jet that defines the
            equilibrium height (its balanced height, profile of Galewsky et al.).
        relaxation_jet_latitude: Latitude of that jet [rad].
        drag_time: Time of the linear drag on vorticity and divergence [hours]. None
            is no drag.
        forcing_type: "none", "vortical", "momentum" or "mass_pulses".
        forcing_energy_rate: Kinetic energy per unit mass injected per unit time by
            "vortical" and "momentum" forcing [m^2/s^3]: the growth rate of the
            global mean of |u|^2 / 2 under a free random walk of the increments.
        forcing_correlation_time: Decorrelation time of the forcing pattern [hours].
            Zero is white noise.
        forcing_spectral_slope: Exponent p of the noise amplitude g_l = b l^p. The
            kinetic-energy spectrum of the forcing follows l^(2p+3).
        forcing_degree_min: Lowest forced degree.
        forcing_degree_max: Highest forced degree. None is L - 1.
        forcing_divergent_fraction: Fraction of the energy rate of "momentum"
            forcing given to the divergent part.
        forcing_backscatter_fraction: Fraction of the diagnosed kinetic-energy
            dissipation of the hyperdiffusion added to the energy rate. Zero keeps
            the fixed rate.
        forcing_backscatter_mode: "global" adds the global mean of the dissipation
            to the rate. "local" multiplies the pattern on the grid by the square
            root of the local rate (Berner et al. 2009, eq. 13), which makes the
            forcing depend on the flow.
        backscatter_include_drag: Whether the loss to the drag counts as
            dissipation in the backscatter.
        pulse_rate: Mean number of mass pulses per hour over the sphere.
        pulse_amplitude: Height of a mass pulse [m].
        pulse_width: Angular width of a mass pulse [rad].
        pulse_remove_mean: Remove the global mean of each pulse, so that the mass
            of the fluid is unchanged.
        return_additional_input_fields: Return, as ``additional_input_fields``, the
            increments of h, u_phi and u_theta added by the forcing between each
            saved frame and the next (zero at the last frame).
        return_energy_budget: Return, as ``energy_budget``, the energy at each saved
            frame and the energy change of each process until the next frame, with
            the columns ``energy_budget_names``.
        return_forcing_state: Return, as ``forcing_state``, the wind tendency of the
            forcing pattern at each saved frame, for a unit energy rate. The pattern
            has memory that the saved fields do not show.
        initial_state: Fields ``[h, u_phi, u_theta]`` of shape ``(*grid.shape, 3)``,
            for ``"restart"`` and ``"from_fields"``.
        initial_constant_fields: Coriolis parameter and orography of that state,
            shape ``(*grid.shape, 2)``, as returned in ``constant_fields``. Needed
            when the state has an orography (h is the free surface) or a rotation
            axis that ``rotation_axis`` does not give. None keeps a flat bottom and
            the axis of ``rotation_axis``.
        initial_forcing_state: Forcing pattern of that state, shape
            ``(*grid.shape, 2)``, as returned in ``forcing_state``, for
            ``branch_forcing_state="carry"``.
        branch_forcing_state: What happens to the forcing pattern, which has memory,
            where a forecast branches (a restart, or the first saved frame of a
            generated run). ``"redraw"`` draws it from its stationary distribution:
            the forecast then depends on the saved fields alone, and its
            distribution is p(x_{n+1} | x_n). ``"carry"`` keeps the pattern: the
            members share it at the branch point, and an emulator needs it as an
            extra input (``forcing_state``) to see what the simulator sees.
        reference_members: Number N of extra members branched from the first saved
            frame of each simulation, with independent forcing. They are returned as
            ``reference_ensemble``, shape ``(n, N, time, *grid.shape, 3)``. The
            simulation itself (``data``) is one more member, the truth.
        output_lmax: Band limit of the saved fields, below ``lmax``: the state is
            truncated to it before it is saved (coarse-graining). None saves the
            full state on the simulation grid.
        output_nlat: Rings of the grid of the saved fields, with ``output_lmax``.
        output_nlon: Points per ring of that grid.
        output_nside: Resolution of that grid, for "healpix".
        return_baseline: With ``output_lmax``, also run the solver at the band limit
            ``output_lmax`` from the truncated first saved frame, and return it as
            ``baseline``: the forecast of the low-resolution physics, which an
            emulator at that resolution has to beat.
    """

    def __init__(  # noqa: PLR0912, PLR0915
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
        noise_amplitude: float = 10.0,
        noise_lmax: int = 120,
        batch_size: int = 8,
        device: str = "cuda",
        initial_condition: str = "galewsky",
        rotation_axis: tuple[float, float] | None = None,
        return_constant_fields: bool = False,
        sht_iterations: int = 0,
        relaxation_time: float | None = None,
        relaxation_jet_speed: float = 80.0,
        relaxation_jet_latitude: float = math.pi / 4,
        drag_time: float | None = None,
        forcing_type: str = "none",
        forcing_energy_rate: float = 0.0,
        forcing_correlation_time: float = 0.0,
        forcing_spectral_slope: float = -1.27,
        forcing_degree_min: int = 1,
        forcing_degree_max: int | None = None,
        forcing_divergent_fraction: float = 0.5,
        forcing_backscatter_fraction: float = 0.0,
        forcing_backscatter_mode: str = "global",
        backscatter_include_drag: bool = False,
        pulse_rate: float = 0.0,
        pulse_amplitude: float = 0.0,
        pulse_width: float = 0.1,
        pulse_remove_mean: bool = True,
        return_additional_input_fields: bool = False,
        return_energy_budget: bool = False,
        return_forcing_state: bool = False,
        initial_state: TensorLike | None = None,
        initial_constant_fields: TensorLike | None = None,
        initial_forcing_state: TensorLike | None = None,
        branch_forcing_state: str = "redraw",
        reference_members: int = 0,
        output_lmax: int | None = None,
        output_nlat: int | None = None,
        output_nlon: int | None = None,
        output_nside: int | None = None,
        return_baseline: bool = False,
    ) -> None:
        """Store the configuration and build the spectral operators once."""
        sphere_grid = make_grid(grid, nlat, nlon, nside)  # needs torch-harmonics
        if initial_condition not in INITIAL_CONDITIONS:
            msg = f"initial_condition must be one of {INITIAL_CONDITIONS}"
            raise ValueError(msg)
        if forcing_type not in FORCING_TYPES:
            msg = f"forcing_type must be one of {FORCING_TYPES}"
            raise ValueError(msg)
        if forcing_backscatter_mode not in BACKSCATTER_MODES:
            msg = f"forcing_backscatter_mode must be one of {BACKSCATTER_MODES}"
            raise ValueError(msg)
        kinetic_forcing = forcing_type in ("vortical", "momentum")
        if forcing_backscatter_fraction < 0:
            msg = "forcing_backscatter_fraction must be non-negative"
            raise ValueError(msg)
        if forcing_backscatter_fraction > 0 and not kinetic_forcing:
            msg = "backscatter requires forcing_type 'vortical' or 'momentum'"
            raise ValueError(msg)
        if return_forcing_state and not kinetic_forcing:
            msg = "return_forcing_state requires forcing_type 'vortical' or 'momentum'"
            raise ValueError(msg)
        if forcing_energy_rate < 0 or forcing_correlation_time < 0:
            msg = "forcing_energy_rate and forcing_correlation_time must be >= 0"
            raise ValueError(msg)
        if not 0 <= forcing_divergent_fraction <= 1:
            msg = "forcing_divergent_fraction must be between 0 and 1"
            raise ValueError(msg)
        if branch_forcing_state not in BRANCH_FORCING_STATES:
            msg = f"branch_forcing_state must be one of {BRANCH_FORCING_STATES}"
            raise ValueError(msg)
        if reference_members < 0:
            msg = "reference_members must be non-negative"
            raise ValueError(msg)
        supplied_state = initial_condition in ("restart", "from_fields")
        if supplied_state and initial_state is None:
            msg = (
                f"initial_state is required for initial_condition='{initial_condition}'"
            )
            raise ValueError(msg)
        if not supplied_state and initial_state is not None:
            msg = (
                "initial_state is only valid for initial_condition='restart' or "
                "'from_fields'"
            )
            raise ValueError(msg)
        if initial_state is not None:
            initial_state = torch.as_tensor(initial_state).detach()
            expected_shape = (*sphere_grid.shape, 3)
            if tuple(initial_state.shape) == (1, *expected_shape):
                initial_state = initial_state[0]
            if tuple(initial_state.shape) != expected_shape:
                msg = f"initial_state must have shape {expected_shape}"
                raise ValueError(msg)
            if not torch.isfinite(initial_state).all():
                msg = "initial_state must be finite"
                raise ValueError(msg)
        if initial_constant_fields is not None:
            initial_constant_fields = torch.as_tensor(initial_constant_fields).detach()
            expected_shape = (*sphere_grid.shape, 2)
            if tuple(initial_constant_fields.shape) != expected_shape:
                msg = f"initial_constant_fields must have shape {expected_shape}"
                raise ValueError(msg)
        if initial_forcing_state is not None:
            initial_forcing_state = torch.as_tensor(initial_forcing_state).detach()
            expected_shape = (*sphere_grid.shape, 2)
            if tuple(initial_forcing_state.shape) != expected_shape:
                msg = f"initial_forcing_state must have shape {expected_shape}"
                raise ValueError(msg)
        carries_pattern = branch_forcing_state == "carry" and kinetic_forcing
        if supplied_state and carries_pattern and initial_forcing_state is None:
            msg = "branch_forcing_state='carry' requires initial_forcing_state"
            raise ValueError(msg)
        if output_lmax is not None and output_lmax > lmax:
            msg = "output_lmax must not exceed lmax"
            raise ValueError(msg)
        if return_baseline and output_lmax is None:
            msg = "return_baseline requires output_lmax"
            raise ValueError(msg)
        if forcing_degree_max is None:
            forcing_degree_max = lmax - 1
        if not 1 <= forcing_degree_min <= forcing_degree_max <= lmax - 1:
            msg = "forcing degrees must satisfy 1 <= min <= max <= lmax - 1"
            raise ValueError(msg)
        if initial_condition == "williamson2" and rotation_axis is not None:
            msg = "initial_condition='williamson2' sets the rotation axis from its tilt"
            raise ValueError(msg)
        if parameters_range is None:
            parameters_range = dict(DEFAULT_PARAMETER_RANGES[initial_condition])
        required_parameters = list(DEFAULT_PARAMETER_RANGES[initial_condition])
        missing = [name for name in required_parameters if name not in parameters_range]
        if missing:
            msg = (
                f"parameters_range must contain {missing} for "
                f"initial_condition='{initial_condition}'"
            )
            raise ValueError(msg)
        unknown = [
            name
            for name in parameters_range
            if name not in required_parameters and name not in SAMPLED_SCALARS
        ]
        if unknown:
            msg = (
                f"parameters_range has unknown keys {unknown} for "
                f"initial_condition='{initial_condition}'"
            )
            raise ValueError(msg)
        if output_names is None:
            output_names = ["h", "u_phi", "u_theta"]
        super().__init__(parameters_range, output_names, log_level)

        self.grid = sphere_grid
        self.lmax = lmax
        self.T = T
        self.dt_save = dt_save
        self.skip_nt = skip_nt
        self.courant = courant
        self.noise_amplitude = noise_amplitude
        self.noise_lmax = noise_lmax
        self.batch_size = batch_size
        self.device = device
        self.initial_condition = initial_condition
        self.rotation_axis = rotation_axis
        self.return_constant_fields = return_constant_fields
        self.relaxation_time = relaxation_time
        self.relaxation_jet_speed = relaxation_jet_speed
        self.relaxation_jet_latitude = relaxation_jet_latitude
        self.drag_time = drag_time
        self.forcing_type = forcing_type
        self.forcing_energy_rate = forcing_energy_rate
        self.forcing_correlation_time = forcing_correlation_time
        self.forcing_spectral_slope = forcing_spectral_slope
        self.forcing_degree_min = forcing_degree_min
        self.forcing_degree_max = forcing_degree_max
        self.forcing_divergent_fraction = forcing_divergent_fraction
        self.forcing_backscatter_fraction = forcing_backscatter_fraction
        self.forcing_backscatter_mode = forcing_backscatter_mode
        self.backscatter_include_drag = backscatter_include_drag
        self.pulse_rate = pulse_rate
        self.pulse_amplitude = pulse_amplitude
        self.pulse_width = pulse_width
        self.pulse_remove_mean = pulse_remove_mean
        self.return_additional_input_fields = return_additional_input_fields
        self.return_energy_budget = return_energy_budget
        self.return_forcing_state = return_forcing_state
        self.initial_state = initial_state
        self.initial_constant_fields = initial_constant_fields
        self.initial_forcing_state = initial_forcing_state
        self.branch_forcing_state = branch_forcing_state
        self.reference_members = reference_members
        self.return_baseline = return_baseline
        self.additional_input_names = ["forcing_h", "forcing_u_phi", "forcing_u_theta"]
        self.energy_budget_names = list(ENERGY_BUDGET_NAMES)
        self.forcing_state_names = ["forcing_state_u_phi", "forcing_state_u_theta"]

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
            self.grid,
            lmax,
            radius,
            omega,
            gravity,
            mean_depth,
            self.dt,
            device,
            sht_iterations,
        )

        # Grid of the saved fields. With output_lmax the state is truncated to this
        # band limit before it is saved, and the solver of the baseline runs on it,
        # with its own Courant step.
        self.output_grid = self.grid
        self.output_dynamics = self.dynamics
        self.output_steps_per_save = self.steps_per_save
        if output_lmax is not None:
            self.output_grid = make_grid(grid, output_nlat, output_nlon, output_nside)
            output_dt = courant * (radius / output_lmax) / (wave_speed + max_wind)
            self.output_steps_per_save = math.ceil(dt_save * 3600 / output_dt)
            self.output_dynamics = SphereShallowWaterDynamics(
                self.output_grid,
                output_lmax,
                radius,
                omega,
                gravity,
                mean_depth,
                dt_save * 3600 / self.output_steps_per_save,
                device,
                sht_iterations,
            )

        # Content of one flat output row: (key, shape of one simulation). The extra
        # outputs travel in the row so that the retries of the base class keep them
        # aligned with the data.
        output_shape = self.output_grid.shape
        self.row_layout: list[tuple[str, tuple[int, ...]]] = [
            ("data", (self.n_frames, *output_shape, len(self.output_names)))
        ]
        if return_constant_fields:
            self.row_layout.append(("constant_fields", (*output_shape, 2)))
        if return_additional_input_fields:
            shape = (self.n_frames, *output_shape, 3)
            self.row_layout.append(("additional_input_fields", shape))
        if return_energy_budget:
            shape = (self.n_frames, len(self.energy_budget_names))
            self.row_layout.append(("energy_budget", shape))
        if return_forcing_state:
            shape = (self.n_frames, *output_shape, 2)
            self.row_layout.append(("forcing_state", shape))
        if reference_members > 0:
            shape = (reference_members, self.n_frames, *output_shape, 3)
            self.row_layout.append(("reference_ensemble", shape))
        if return_baseline:
            shape = (self.n_frames, *output_shape, 3)
            self.row_layout.append(("baseline", shape))

    def sampled_or_fixed(self, rows: TensorLike, name: str, value: float) -> TensorLike:
        """Value of a scalar of ``SAMPLED_SCALARS`` for each simulation of a batch.

        The value is the column ``name`` of ``rows`` when ``name`` is a key of
        ``parameters_range``, and ``value``, the constructor argument, otherwise.

        Args:
            rows: Parameter rows (B, n_parameters).
            name: Name of the scalar.
            value: Constructor argument of that scalar.

        Returns:
            Values (B,), float64.
        """
        if name in self.param_names:
            return rows[:, self.get_parameter_idx(name)].to(torch.float64)
        return torch.full((rows.shape[0],), value, dtype=torch.float64)

    def forcing_settings(self, rows: TensorLike) -> ForcingSettings | None:
        """Settings of the stochastic forcing of one batch, None without forcing.

        Times are converted from hours to seconds and the pulse rate from pulses per
        hour to pulses per second.

        Args:
            rows: Parameter rows (B, n_parameters).

        Returns:
            The settings, with one value per simulation for the sampled scalars.
        """
        if self.forcing_type == "none":
            return None
        energy_rate = self.sampled_or_fixed(
            rows, "forcing_energy_rate", self.forcing_energy_rate
        )
        correlation_hours = self.sampled_or_fixed(
            rows, "forcing_correlation_time", self.forcing_correlation_time
        )
        spectral_slope = self.sampled_or_fixed(
            rows, "forcing_spectral_slope", self.forcing_spectral_slope
        )
        divergent_fraction = self.sampled_or_fixed(
            rows, "forcing_divergent_fraction", self.forcing_divergent_fraction
        )
        if self.forcing_type == "vortical":
            divergent_fraction = torch.zeros_like(divergent_fraction)
        backscatter_fraction = self.sampled_or_fixed(
            rows, "forcing_backscatter_fraction", self.forcing_backscatter_fraction
        )
        pulses_per_hour = self.sampled_or_fixed(rows, "pulse_rate", self.pulse_rate)
        pulse_amplitude = self.sampled_or_fixed(
            rows, "pulse_amplitude", self.pulse_amplitude
        )
        pulse_width = self.sampled_or_fixed(rows, "pulse_width", self.pulse_width)
        return ForcingSettings(
            forcing_type=self.forcing_type,
            energy_rate=energy_rate,
            correlation_time=3600 * correlation_hours,
            spectral_slope=spectral_slope,
            degree_min=self.forcing_degree_min,
            degree_max=self.forcing_degree_max,
            divergent_fraction=divergent_fraction,
            backscatter_fraction=backscatter_fraction,
            backscatter_mode=self.forcing_backscatter_mode,
            backscatter_include_drag=self.backscatter_include_drag,
            pulse_rate=pulses_per_hour / 3600,
            pulse_amplitude=pulse_amplitude,
            pulse_width=pulse_width,
            pulse_remove_mean=self.pulse_remove_mean,
        )

    def damping_settings(
        self, rows: TensorLike, coriolis: TensorLike | None
    ) -> tuple[TensorLike | None, TensorLike | None, TensorLike | None]:
        """Drag time, relaxation time and equilibrium geopotential of one batch.

        The equilibrium is the balanced geopotential of a zonal jet of speed
        ``relaxation_jet_speed`` at ``relaxation_jet_latitude``.

        Args:
            rows: Parameter rows (B, n_parameters).
            coriolis: Coriolis parameter (B, *grid.shape); None is the polar one.

        Returns:
            The drag time [s] (B,), the relaxation time [s] (B,) and the coefficients
            of the equilibrium geopotential (B, L, L). Each is None when its process
            is absent.
        """
        batch = rows.shape[0]
        drag_time = None
        if "drag_time" in self.param_names:
            drag_time = 3600 * rows[:, self.get_parameter_idx("drag_time")]
        elif self.drag_time is not None:
            drag_time = torch.full((batch,), 3600 * self.drag_time)

        relaxation_time = None
        equilibrium_geopotential = None
        if "relaxation_time" in self.param_names:
            relaxation_time = 3600 * rows[:, self.get_parameter_idx("relaxation_time")]
        elif self.relaxation_time is not None:
            relaxation_time = torch.full((batch,), 3600 * self.relaxation_time)
        if relaxation_time is not None:
            jet_speed = self.sampled_or_fixed(
                rows, "relaxation_jet_speed", self.relaxation_jet_speed
            )
            jet_latitude = torch.full(
                (batch,), self.relaxation_jet_latitude, dtype=torch.float64
            )
            vorticity, divergence = zonal_jet(self.dynamics, jet_speed, jet_latitude)
            equilibrium_geopotential = balanced_geopotential(
                self.dynamics, vorticity, divergence, coriolis
            )
        return drag_time, relaxation_time, equilibrium_geopotential

    def supplied_problem(
        self, batch: int, coriolis: TensorLike | None
    ) -> tuple[TensorLike, TensorLike | None, TensorLike | None]:
        """Initial state and fixed fields of ``"restart"`` and ``"from_fields"``.

        Every simulation of the batch starts from ``initial_state``. With
        ``initial_constant_fields`` the Coriolis parameter and the orography are
        those of the supplied problem. ``"from_fields"`` then applies its balance
        step.

        Args:
            batch: Number of simulations B.
            coriolis: Coriolis parameter of ``rotation_axis``, (B, *grid.shape), or
                None for the polar one. Replaced by the supplied one, if any.

        Returns:
            The spectral state (B, 3, L, L), the Coriolis parameter or None, and the
            coefficients of the surface geopotential or None.
        """
        if self.initial_state is None:
            msg = "initial_state is required"  # checked by the constructor
            raise ValueError(msg)
        grid_shape = self.grid.shape
        fields = self.initial_state.to(self.device, torch.float64)
        fields = fields.expand(batch, *grid_shape, 3)

        surface_geopotential = None
        if self.initial_constant_fields is not None:
            constant = self.initial_constant_fields.to(self.device, torch.float64)
            coriolis = constant[..., 0].expand(batch, *grid_shape)
            surface_height = constant[..., 1].expand(batch, *grid_shape)
            surface_coefficients = self.dynamics.to_coefficients(surface_height)
            surface_geopotential = torch.tril(
                self.dynamics.gravity * surface_coefficients
            )
        state = fields_to_spectral(self.dynamics, fields, surface_geopotential)

        if self.initial_condition == "from_fields":
            # Balance step: no divergence, and the free surface in nonlinear balance
            # with the wind, with the mean of the supplied free surface
            vorticity = state[:, 1]
            divergence = torch.zeros_like(vorticity)
            free_surface = balanced_geopotential(
                self.dynamics, vorticity, divergence, coriolis
            )
            supplied_free_surface = state[:, 0]
            if surface_geopotential is not None:
                supplied_free_surface = state[:, 0] + surface_geopotential
            free_surface[:, 0, 0] = supplied_free_surface[:, 0, 0]
            geopotential = free_surface
            if surface_geopotential is not None:
                geopotential = free_surface - surface_geopotential
            state = torch.stack([geopotential, vorticity, divergence], dim=1)
        return state, coriolis, surface_geopotential

    def initial_problem(
        self, rows: TensorLike, noise_generators: list[torch.Generator]
    ) -> tuple[TensorLike, TensorLike | None, TensorLike | None]:
        """Build the initial state and the fixed fields of one batch.

        Args:
            rows: Parameter rows (B, n_parameters), columns ordered as ``param_names``.
            noise_generators: One generator per simulation, for the initial noise.

        Returns:
            The spectral state (B, 3, L, L), the Coriolis parameter (B, *grid.shape)
            or None for the polar one, and the coefficients of the surface
            geopotential g h_s (B, L, L) or None for a flat bottom.
        """
        batch = rows.shape[0]
        coriolis = None
        surface_geopotential = None
        if self.rotation_axis is not None:
            axis_lon = torch.full((batch,), self.rotation_axis[0], dtype=torch.float64)
            axis_lat = torch.full((batch,), self.rotation_axis[1], dtype=torch.float64)
            coriolis = coriolis_parameter(self.dynamics, axis_lon, axis_lat)

        if self.initial_condition == "galewsky":
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
                generators=noise_generators,
            )
        elif self.initial_condition == "galewsky_unperturbed":
            state = galewsky_initial_condition(
                self.dynamics,
                jet_speed=rows[:, self.get_parameter_idx("jet_speed")],
                height_perturbation=torch.zeros(batch),
                perturbation_lon=torch.zeros(batch),
                perturbation_lat=torch.full((batch,), math.pi / 4),
                noise_amplitude=0.0,
                noise_lmax=self.noise_lmax,
                device=self.device,
                generators=noise_generators,
            )
        elif self.initial_condition == "williamson2":
            tilt = rows[:, self.get_parameter_idx("tilt")]
            state = williamson2_initial_condition(
                self.dynamics,
                flow_speed=rows[:, self.get_parameter_idx("flow_speed")],
                tilt=tilt,
            )
            # Rotation pole of W92 eq. (96): longitude pi, latitude pi/2 - tilt. In
            # float64: a pole misplaced by the float32 rounding of pi (1e-7 rad)
            # unbalances the flow at the 1e-8 level.
            axis_lon = torch.full((batch,), math.pi, dtype=torch.float64)
            axis_lat = math.pi / 2 - tilt.to(torch.float64)
            coriolis = coriolis_parameter(self.dynamics, axis_lon, axis_lat)
        elif self.initial_condition == "rossby_haurwitz":
            state = rossby_haurwitz_initial_condition(
                self.dynamics,
                wavenumber=rows[:, self.get_parameter_idx("wavenumber")],
                wave_amplitude=rows[:, self.get_parameter_idx("wave_amplitude")],
                mean_depth=rows[:, self.get_parameter_idx("mean_depth")],
            )
        elif self.initial_condition in ("restart", "from_fields"):
            state, coriolis, surface_geopotential = self.supplied_problem(
                batch, coriolis
            )
        elif self.initial_condition == "random_balanced":
            state = random_balanced_initial_condition(
                self.dynamics,
                amp=rows[:, self.get_parameter_idx("amp")],
                initial_wavenumber=rows[
                    :, self.get_parameter_idx("initial_wavenumber")
                ],
                initial_bandwidth=rows[:, self.get_parameter_idx("initial_bandwidth")],
                generators=noise_generators,
                coriolis=coriolis,
            )
        else:  # "mountain"
            state, surface_geopotential = mountain_initial_condition(
                self.dynamics,
                flow_speed=rows[:, self.get_parameter_idx("flow_speed")],
                mountain_lon=rows[:, self.get_parameter_idx("mountain_lon")],
                mountain_lat=rows[:, self.get_parameter_idx("mountain_lat")],
                mountain_height=rows[:, self.get_parameter_idx("mountain_height")],
                mountain_radius=rows[:, self.get_parameter_idx("mountain_radius")],
            )
        return state, coriolis, surface_geopotential

    def constant_fields_on_grid(
        self,
        batch: int,
        coriolis: TensorLike | None,
        surface_geopotential: TensorLike | None,
    ) -> TensorLike:
        """Coriolis parameter and orography on the grid of the saved fields.

        Args:
            batch: Number of simulations B.
            coriolis: Coriolis parameter (B, *grid.shape); None is the polar one.
            surface_geopotential: Coefficients of g h_s, (B, L, L); None is a flat
                bottom.

        Returns:
            f and h_s, shape (B, *output_grid.shape, 2).
        """
        output_lmax = self.output_dynamics.lmax
        if coriolis is None:
            coriolis = self.dynamics.coriolis.expand(batch, *self.grid.shape)
        if self.output_dynamics is not self.dynamics:
            coefficients = self.dynamics.to_coefficients(coriolis)
            coriolis = self.output_dynamics.to_grid(
                coefficients[..., :output_lmax, :output_lmax]
            )
        if surface_geopotential is None:
            surface_height = torch.zeros_like(coriolis)
        else:
            truncated = surface_geopotential[..., :output_lmax, :output_lmax]
            surface_height = (
                self.output_dynamics.to_grid(truncated) / self.dynamics.gravity
            )
        return torch.stack([coriolis, surface_height], dim=-1)

    def initial_forcing_pattern(self, batch: int) -> TensorLike | None:
        """Forcing pattern of a restart that carries it, None otherwise.

        ``initial_forcing_state`` is the wind tendency of the pattern for a unit
        energy rate, the Laplacian of the pattern divided by the time step. Its
        vorticity and divergence give back the streamfunction and the velocity
        potential of the pattern.

        Args:
            batch: Number of simulations B.

        Returns:
            Pattern coefficients (B, P, L, L), or None when the pattern is drawn.
        """
        if self.initial_forcing_state is None or self.branch_forcing_state != "carry":
            return None
        tendency = self.initial_forcing_state.to(self.device, torch.float64)
        tendency = tendency.expand(batch, *self.grid.shape, 2)
        wind = torch.stack([tendency[..., 0], -tendency[..., 1]], dim=1)
        vorticity, divergence = self.dynamics.vorticity_divergence(wind)
        scale = self.dynamics.inverse_laplacian * self.dynamics.dt
        if self.forcing_type == "vortical":
            pattern = (scale * vorticity).unsqueeze(1)
        else:
            pattern = torch.stack([scale * vorticity, scale * divergence], dim=1)
        return torch.tril(pattern)

    def forward_batch(  # noqa: PLR0912, PLR0915
        self,
        x: TensorLike,
        allow_failures: bool = True,
        *,
        num_workers: int | None = None,
        random_seed: int | None = None,
    ) -> tuple[TensorLike, TensorLike]:
        """Run one simulation per row of x, ``batch_size`` of them at a time.

        Every random draw of a simulation comes from generators seeded from
        ``random_seed`` and the row of the simulation in x. Without ``random_seed``
        the seed is one integer drawn from the global torch generator, which
        ``sample_inputs`` has seeded with the seed of the sample.

        Args:
            x: Parameter rows (n, n_parameters), columns ordered as ``param_names``.
            allow_failures: Kept for the autosim interface; non-finite simulations
                are always dropped.
            num_workers: Not supported: use ``batch_size``.
            random_seed: Optional seed of the random draws.

        Returns:
            (y, x_valid) with y of shape (n_valid, row size): the flattened outputs
            of ``row_layout`` side by side.
        """
        del allow_failures
        if num_workers is not None:
            msg = "SphericalShallowWater runs simulations in parallel with batch_size."
            raise ValueError(msg)
        if random_seed is None:
            entropy = int(torch.randint(0, 2**31 - 1, (1,)).item())
        else:
            entropy = random_seed

        x = x.clone()
        if self.initial_condition == "rossby_haurwitz":
            column = self.get_parameter_idx("wavenumber")
            x[:, column] = x[:, column].round()  # the wavenumber is an integer

        results = []
        valid_rows = []
        for start in range(0, x.shape[0], self.batch_size):
            rows = x[start : start + self.batch_size]
            batch = rows.shape[0]
            row_numbers = list(range(start, start + batch))
            noise_generators = sample_generators(
                entropy, row_numbers, INITIAL_NOISE_STREAM, self.device
            )
            state, coriolis, surface_geopotential = self.initial_problem(
                rows, noise_generators
            )
            drag_time, relaxation_time, equilibrium = self.damping_settings(
                rows, coriolis
            )
            forcing = self.forcing_settings(rows)
            forcing_generators = sample_generators(
                entropy, row_numbers, FORCING_STREAM, self.device
            )
            n_saves = round(self.T / self.dt_save)
            carry = self.branch_forcing_state == "carry"
            output = integrate_trajectories(
                self.dynamics,
                state,
                n_saves=n_saves,
                steps_per_save=self.steps_per_save,
                skip_nt=self.skip_nt,
                coriolis=coriolis,
                surface_geopotential=surface_geopotential,
                drag_time=drag_time,
                relaxation_time=relaxation_time,
                equilibrium_geopotential=equilibrium,
                forcing=forcing,
                forcing_generators=forcing_generators,
                return_additional_input_fields=self.return_additional_input_fields,
                return_energy_budget=self.return_energy_budget,
                return_forcing_state=self.return_forcing_state,
                initial_pattern=self.initial_forcing_pattern(batch),
                redraw_pattern_at_first_save=not carry,
                output_dynamics=self.output_dynamics,
            )

            # Reference ensemble: N more members from the first saved frame, each
            # with its own forcing stream. They continue the time stepping of the
            # truth (same history), and their pattern is drawn anew or inherited.
            if self.reference_members > 0:
                members = []
                for member in range(self.reference_members):
                    member_generators = sample_generators(
                        entropy, row_numbers, REFERENCE_STREAM + member, self.device
                    )
                    member_pattern = None  # drawn from its stationary distribution
                    if carry:
                        member_pattern = output["branch_pattern"]
                    member_output = integrate_trajectories(
                        self.dynamics,
                        output["branch_state"],
                        n_saves=n_saves - self.skip_nt,
                        steps_per_save=self.steps_per_save,
                        skip_nt=0,
                        coriolis=coriolis,
                        surface_geopotential=surface_geopotential,
                        drag_time=drag_time,
                        relaxation_time=relaxation_time,
                        equilibrium_geopotential=equilibrium,
                        forcing=forcing,
                        forcing_generators=member_generators,
                        previous_tendencies=output["branch_tendencies"],
                        initial_pattern=member_pattern,
                        output_dynamics=self.output_dynamics,
                    )
                    members.append(member_output["data"])
                output["reference_ensemble"] = torch.stack(members, dim=1)

            # Baseline: the solver at the band limit of the saved fields, from the
            # truncated first saved frame
            if self.return_baseline:
                low = self.output_dynamics
                low_coriolis = None
                if coriolis is not None:
                    coefficients = self.dynamics.to_coefficients(coriolis)
                    low_coriolis = low.to_grid(
                        coefficients[..., : low.lmax, : low.lmax]
                    )
                low_surface = None
                if surface_geopotential is not None:
                    low_surface = surface_geopotential[..., : low.lmax, : low.lmax]
                low_equilibrium = None
                if equilibrium is not None:
                    low_equilibrium = equilibrium[..., : low.lmax, : low.lmax]
                # The forced degrees stop at the band limit of the baseline
                low_forcing = None
                if forcing is not None:
                    low_degree_max = min(forcing.degree_max, low.lmax - 1)
                    low_forcing = dataclasses.replace(
                        forcing, degree_max=low_degree_max
                    )
                baseline_generators = sample_generators(
                    entropy, row_numbers, BASELINE_STREAM, self.device
                )
                baseline_output = integrate_trajectories(
                    low,
                    output["branch_state"][..., : low.lmax, : low.lmax],
                    n_saves=n_saves - self.skip_nt,
                    steps_per_save=self.output_steps_per_save,
                    skip_nt=0,
                    coriolis=low_coriolis,
                    surface_geopotential=low_surface,
                    drag_time=drag_time,
                    relaxation_time=relaxation_time,
                    equilibrium_geopotential=low_equilibrium,
                    forcing=low_forcing,
                    forcing_generators=baseline_generators,
                )
                output["baseline"] = baseline_output["data"]
            if self.return_constant_fields:
                fields = self.constant_fields_on_grid(
                    batch, coriolis, surface_geopotential
                )
                output["constant_fields"] = fields.float().cpu()

            flat_outputs = []
            for key, _ in self.row_layout:
                flat_outputs.append(output[key].float().flatten(start_dim=1))
            flat_rows = torch.cat(flat_outputs, dim=1)
            finite = torch.isfinite(flat_rows).all(dim=1)
            results.append(flat_rows[finite])
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
            random_seed: Seed of the parameter sample and of every random draw.
            ensure_exact_n: Replace failed simulations until n succeed.

        Returns:
            ``{"data": (n, time, *grid.shape, 3), "constant_scalars": (n,
            n_parameters), "constant_fields": (n, *grid.shape, 2) or None}``, and
            when requested ``"additional_input_fields"`` (n, time, *grid.shape, 3),
            ``"energy_budget"`` (n, time, 7), ``"forcing_state"`` (n, time,
            *grid.shape, 2), ``"reference_ensemble"`` (n, N, time, *grid.shape, 3)
            and ``"baseline"`` (n, time, *grid.shape, 3). With ``output_lmax`` the
            grid is that of the saved fields.
        """
        y, x = self._forward_batch_with_optional_retries(
            n=n, random_seed=random_seed, ensure_exact_n=ensure_exact_n
        )
        payload: dict = {"constant_scalars": x, "constant_fields": None}
        offset = 0
        for key, shape in self.row_layout:
            size = math.prod(shape)
            payload[key] = y[:, offset : offset + size].reshape(y.shape[0], *shape)
            offset += size
        return payload
