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
    make_grid,
)
from .era5 import (
    ERA5_CLIMATOLOGY_STORE,
    ERA5_LEVEL,
    ERA5_MIN_DEPTH,
    ERA5_OROGRAPHY_DEGREE,
    ERA5_RANGES,
    ERA5_STEP_HOURS,
    ERA5_STORE,
    era5_initial_condition,
    era5_source_dynamics,
    era5_time_index,
    load_era5_climatology,
    load_era5_fields,
)
from .forcing import (
    FORCING_TYPES,
    ForcingSettings,
    PeriodicForcingSettings,
    SpptSettings,
    periodic_height_forcing,
)
from .initial_conditions import (
    GALEWSKY_RANGES,
    MOUNTAIN_RANGES,
    WILLIAMSON3_RANGES,
    balanced_geopotential,
    galewsky_initial_condition,
    mountain_initial_condition,
    tilted_pole_coriolis,
    williamson3_initial_condition,
    zonal_jet,
)
from .integration import ENERGY_BUDGET_NAMES, integrate_trajectories
from .shamir_paldor import SHAMIR_PALDOR_RANGES, shamir_paldor_initial_condition

INITIAL_CONDITIONS = (
    "galewsky",
    "williamson3",
    "shamir_paldor",
    "mountain",
    "era5",
)
BACKSCATTER_MODES = ("global", "local")
RELAXATION_TARGETS = ("jet", "era5_climatology")

# Default sampling ranges of each family; their keys are the required parameters.
DEFAULT_PARAMETER_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "galewsky": GALEWSKY_RANGES,
    "williamson3": WILLIAMSON3_RANGES,
    "shamir_paldor": SHAMIR_PALDOR_RANGES,
    "mountain": MOUNTAIN_RANGES,
    "era5": ERA5_RANGES,
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
)

# Independent random streams of one simulation
INITIAL_NOISE_STREAM = 0
FORCING_STREAM = 1
BASELINE_STREAM = 2
SPPT_STREAM = 3
BASELINE_SPPT_STREAM = 4
REFERENCE_STREAM = 100  # member j of a reference ensemble uses stream 100 + j
REFERENCE_SPPT_STREAM = 200  # and 200 + j for its SPPT pattern


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
    - ``"williamson3"``: steady zonal jet with compact support about an axis tilted
      by ``tilt``, with the Coriolis parameter of the same axis (W92 case 3).
      Parameters ``flow_speed``, ``tilt``.
    - ``"shamir_paldor"``: one Rossby or inertia-gravity wave of Shamir and Paldor
      (2016) at the rest depth ``mean_depth``. Parameters ``wave_type`` (0 Rossby,
      1 eastward, 2 westward inertia-gravity), ``zonal_wavenumber``, ``mode_number``
      (all three rounded to the nearest integer; the rounded values are returned in
      ``constant_scalars``) and ``wave_amplitude``, the largest wind speed [m/s].
    - ``"mountain"``: zonal flow over an isolated Gaussian mountain (W92 case 5 with
      the mountain of DCMIP-2008). Parameters ``flow_speed``, ``mountain_height``,
      ``mountain_width``, ``mountain_lon``, ``mountain_lat``.
    - ``"era5"``: the ERA5 state at 500 hPa over the Earth's orography, after a
      balance step (W92 case 7), read from the WeatherBench2 store ``era5_store``.
      The orography is smoothed (``era5_orography_degree``), and a date whose
      balanced depth falls below ``era5_min_depth`` is rejected: its row is dropped,
      and ``ensure_exact_n`` draws another date.
      Parameter ``era5_time``, hours since 1959-01-01T00, rounded to the 6-hourly
      times of the store; the rounded value is returned in ``constant_scalars``.

    The arguments listed in ``SAMPLED_SCALARS`` (drag and relaxation times, forcing
    rate, decorrelation time, slope, fractions) can also be keys of
    ``parameters_range``. They are then sampled for each simulation, and the
    constructor value is not used.

    Physical processes that can be added to the conservative dynamics, alone or
    together (module ``integration`` gives their order in a step, module ``forcing``
    the definitions):

    - Newtonian relaxation of the height towards the balanced height of a zonal jet,
      and linear drag on the vorticity and the divergence. With both, the flow
      reaches a statistically steady state with a jet and eddies.
    - Stochastic kinetic-energy forcing, ``forcing_type``: ``"vortical"``
      (streamfunction pattern, Berner et al. 2009) or ``"momentum"`` (the same with a
      divergent part). The energy rate can include a backscatter of the diagnosed
      dissipation, global or local.
    - Stochastically perturbed physics tendencies, ``sppt_amplitudes``: the increment
      of the drag and the relaxation multiplied by a random pattern (ECMWF's SPPT).
    - The daily and yearly height forcing of The Well's planetswe,
      ``periodic_forcing_amplitude``.

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
            Coriolis parameter. None is the geographic pole. ``"williamson3"`` sets
            the axis from its tilt, and ``"era5"`` uses the geographic pole.
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
        relaxation_target: Equilibrium of the relaxation: ``"jet"``, the balanced
            height of the jet above, or ``"era5_climatology"`` (family ``"era5"``
            only), the ERA5 1990-2019 climatology of the free surface at the day of
            year and hour of the start date, minus the orography, held fixed.
        drag_time: Time of the linear drag on vorticity and divergence [hours]. None
            is no drag.
        forcing_type: "none", "vortical" or "momentum".
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
        sppt_amplitudes: Grid-point standard deviation of each field of the SPPT
            pattern. None is no SPPT. Requires drag or relaxation.
        sppt_length_scales: Correlation length of each field [m].
        sppt_correlation_times: Decorrelation time of each field [hours].
        sppt_clip: Bound of the summed pattern.
        periodic_forcing_amplitude: Amplitude h_f0 of The Well's height forcing
            [m/hour]. Zero is no such forcing.
        periodic_forcing_width: Width sigma of its latitude profile [rad].
        periodic_forcing_declination: Maximum latitude of its centre [rad].
        periodic_forcing_day: Period of the longitude of its centre [hours].
        periodic_forcing_year: Period of the latitude of its centre [hours].
        era5_store: WeatherBench2 zarr store of ``"era5"``.
        era5_level: Pressure level of ``"era5"`` [hPa].
        era5_orography_degree: Degree l_0 of the orography filter exp(-(l/l_0)^4).
        era5_min_depth: Smallest accepted balanced fluid depth of ``"era5"`` [m].
        era5_climatology_store: WeatherBench2 climatology store, for
            ``relaxation_target="era5_climatology"``.
        return_periodic_forcing: Return, as ``periodic_forcing``, The Well's height
            forcing F [m/hour] at the time of each saved frame, shape
            ``(n, time, *grid.shape, 1)``: a known input of the emulator.
        return_additional_input_fields: Return, as ``additional_input_fields``, the
            increments of h, u_phi and u_theta added by all the forcings between each
            saved frame and the next (zero at the last frame).
        return_energy_budget: Return, as ``energy_budget``, the energy at each saved
            frame and the energy change of each process until the next frame, with
            the columns ``energy_budget_names``.
        return_forcing_state: Return, as ``forcing_state``, the wind tendency of the
            kinetic forcing pattern at each saved frame, for a unit energy rate. The
            pattern has memory that the saved fields do not show.
        reference_members: Number N of extra members branched from the first saved
            frame of each simulation, with independent forcing. They are returned as
            ``reference_ensemble``, shape ``(n, N, time, *grid.shape, 3)``. The
            simulation itself (``data``) is one more member, the truth. At that frame
            the random patterns of every member, the truth included, are drawn from
            their stationary distribution, so the forecast depends on the saved
            fields alone: its distribution is p(x_{n+1} | x_n).
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
        relaxation_target: str = "jet",
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
        sppt_amplitudes: list[float] | None = None,
        sppt_length_scales: list[float] | None = None,
        sppt_correlation_times: list[float] | None = None,
        sppt_clip: float = 1.0,
        periodic_forcing_amplitude: float = 0.0,
        periodic_forcing_width: float = 2 * math.pi / 3,
        periodic_forcing_declination: float = 0.4,
        periodic_forcing_day: float = 24.0,
        periodic_forcing_year: float = 1008.0,
        era5_store: str = ERA5_STORE,
        era5_level: int = ERA5_LEVEL,
        era5_orography_degree: float = ERA5_OROGRAPHY_DEGREE,
        era5_min_depth: float = ERA5_MIN_DEPTH,
        era5_climatology_store: str = ERA5_CLIMATOLOGY_STORE,
        return_periodic_forcing: bool = False,
        return_additional_input_fields: bool = False,
        return_energy_budget: bool = False,
        return_forcing_state: bool = False,
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
        kinetic_forcing = forcing_type != "none"
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
        if reference_members < 0:
            msg = "reference_members must be non-negative"
            raise ValueError(msg)
        sppt = None
        if sppt_amplitudes is not None:
            sppt = SpptSettings(
                amplitudes=tuple(float(value) for value in sppt_amplitudes),
                length_scales=tuple(float(value) for value in sppt_length_scales or []),
                correlation_times=tuple(
                    3600 * float(value) for value in sppt_correlation_times or []
                ),
                clip=float(sppt_clip),
            )
            n_fields = len(sppt.amplitudes)
            if n_fields == 0 or {
                len(sppt.length_scales),
                len(sppt.correlation_times),
            } != {n_fields}:
                msg = (
                    "sppt_amplitudes, sppt_length_scales and sppt_correlation_times "
                    "must have the same length, at least 1"
                )
                raise ValueError(msg)
            if min(sppt.amplitudes) <= 0 or min(sppt.length_scales) <= 0:
                msg = "sppt amplitudes and length scales must be positive"
                raise ValueError(msg)
            if min(sppt.correlation_times) < 0 or sppt.clip <= 0:
                msg = "sppt correlation times must be >= 0 and sppt_clip positive"
                raise ValueError(msg)
            has_physics = (
                relaxation_time is not None
                or drag_time is not None
                or "relaxation_time" in (parameters_range or {})
                or "drag_time" in (parameters_range or {})
            )
            if not has_physics:
                msg = "sppt requires relaxation_time or drag_time"
                raise ValueError(msg)
        if periodic_forcing_amplitude < 0:
            msg = "periodic_forcing_amplitude must be non-negative"
            raise ValueError(msg)
        periodic = None
        if periodic_forcing_amplitude > 0:
            if min(periodic_forcing_width, periodic_forcing_day) <= 0:
                msg = "periodic_forcing_width and periodic_forcing_day must be positive"
                raise ValueError(msg)
            if periodic_forcing_year <= 0:
                msg = "periodic_forcing_year must be positive"
                raise ValueError(msg)
            periodic = PeriodicForcingSettings(
                amplitude=periodic_forcing_amplitude / 3600,
                width=periodic_forcing_width,
                declination=periodic_forcing_declination,
                day=3600 * periodic_forcing_day,
                year=3600 * periodic_forcing_year,
            )
        if return_periodic_forcing and periodic is None:
            msg = "return_periodic_forcing requires periodic_forcing_amplitude > 0"
            raise ValueError(msg)
        if relaxation_target not in RELAXATION_TARGETS:
            msg = f"relaxation_target must be one of {RELAXATION_TARGETS}"
            raise ValueError(msg)
        if relaxation_target == "era5_climatology" and initial_condition != "era5":
            msg = "relaxation_target='era5_climatology' needs initial_condition='era5'"
            raise ValueError(msg)
        if era5_orography_degree <= 0:
            msg = "era5_orography_degree must be positive"
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
        if initial_condition in ("williamson3", "era5") and rotation_axis is not None:
            msg = f"initial_condition='{initial_condition}' sets its own rotation axis"
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
        self.return_additional_input_fields = return_additional_input_fields
        self.return_energy_budget = return_energy_budget
        self.return_forcing_state = return_forcing_state
        self.sppt = sppt
        self.periodic = periodic
        self.era5_store = era5_store
        self.era5_level = era5_level
        self.era5_orography_degree = era5_orography_degree
        self.era5_min_depth = era5_min_depth
        self.era5_climatology_store = era5_climatology_store
        self.relaxation_target = relaxation_target
        self.return_periodic_forcing = return_periodic_forcing
        self.era5_source: SphereShallowWaterDynamics | None = None  # built on use
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
        if return_periodic_forcing:
            shape = (self.n_frames, *output_shape, 1)
            self.row_layout.append(("periodic_forcing", shape))
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

        Times are converted from hours to seconds.

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
        )

    def damping_settings(
        self,
        rows: TensorLike,
        coriolis: TensorLike | None,
        surface_geopotential: TensorLike | None = None,
    ) -> tuple[TensorLike | None, TensorLike | None, TensorLike | None]:
        """Drag time, relaxation time and equilibrium geopotential of one batch.

        With ``relaxation_target="jet"`` the equilibrium free surface is the balanced
        geopotential of a zonal jet of speed ``relaxation_jet_speed`` at
        ``relaxation_jet_latitude``. With ``"era5_climatology"`` it is the climatology
        of ``era5_climatology_store`` at the start time of each simulation, analysed
        as the ERA5 fields. Both are returned minus the surface geopotential, as the
        geopotential of the fluid depth.

        Args:
            rows: Parameter rows (B, n_parameters).
            coriolis: Coriolis parameter (B, *grid.shape); None is the polar one.
            surface_geopotential: Coefficients of g h_s, (B, L, L); None is a flat
                bottom.

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
        if relaxation_time is not None and self.relaxation_target == "era5_climatology":
            hours = rows[:, self.get_parameter_idx("era5_time")]
            climatology = load_era5_climatology(
                hours, self.era5_climatology_store, self.era5_level
            )
            if self.era5_source is None:
                msg = "the climatology target needs the ERA5 grid of the initial state"
                raise ValueError(msg)
            free_surface = self.era5_source.to_coefficients(
                climatology.to(self.device, torch.float64)
            )
            equilibrium_geopotential = torch.tril(free_surface)
            if surface_geopotential is not None:
                equilibrium_geopotential = (
                    equilibrium_geopotential - surface_geopotential
                )
        elif relaxation_time is not None:
            jet_speed = self.sampled_or_fixed(
                rows, "relaxation_jet_speed", self.relaxation_jet_speed
            )
            jet_latitude = torch.full(
                (batch,), self.relaxation_jet_latitude, dtype=torch.float64
            )
            vorticity, divergence = zonal_jet(self.dynamics, jet_speed, jet_latitude)
            # Balanced free surface of the jet; the state holds the fluid depth
            equilibrium_geopotential = balanced_geopotential(
                self.dynamics, vorticity, divergence, coriolis
            )
            if surface_geopotential is not None:
                equilibrium_geopotential = (
                    equilibrium_geopotential - surface_geopotential
                )
        return drag_time, relaxation_time, equilibrium_geopotential

    def era5_problem(
        self, rows: TensorLike
    ) -> tuple[TensorLike, TensorLike | None, TensorLike]:
        """Initial state and orography of ``"era5"`` for one batch.

        The fields of the sampled times are read from ``era5_store``. The transforms
        on their grid are built at the first call and kept.

        Args:
            rows: Parameter rows (B, n_parameters); ``era5_time`` is already rounded.

        Returns:
            The spectral state (B, 3, L, L), None (the polar Coriolis parameter) and
            the coefficients of the surface geopotential (B, L, L). The state of a
            rejected date (balanced depth below ``era5_min_depth``) is NaN, so its row
            is dropped.
        """
        hours = rows[:, self.get_parameter_idx("era5_time")]
        fields = load_era5_fields(hours, self.era5_store, self.era5_level)
        if self.era5_source is None:
            self.era5_source = era5_source_dynamics(self.dynamics, fields)
        state, surface_geopotential, valid = era5_initial_condition(
            self.dynamics,
            self.era5_source,
            fields,
            self.era5_orography_degree,
            self.era5_min_depth,
        )
        if not valid.all():
            self.logger.warning(
                "Rejected %d ERA5 date(s) with a fluid depth below %.0f m",
                int((~valid).sum()),
                self.era5_min_depth,
            )
            state[~valid] = torch.nan
        return state, None, surface_geopotential

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
        elif self.initial_condition == "williamson3":
            tilt = rows[:, self.get_parameter_idx("tilt")]
            state = williamson3_initial_condition(
                self.dynamics,
                flow_speed=rows[:, self.get_parameter_idx("flow_speed")],
                tilt=tilt,
            )
            coriolis = tilted_pole_coriolis(self.dynamics, tilt)  # W92 eq. (96)
        elif self.initial_condition == "shamir_paldor":
            state = shamir_paldor_initial_condition(
                self.dynamics,
                wave_type=rows[:, self.get_parameter_idx("wave_type")],
                zonal_wavenumber=rows[:, self.get_parameter_idx("zonal_wavenumber")],
                mode_number=rows[:, self.get_parameter_idx("mode_number")],
                wave_amplitude=rows[:, self.get_parameter_idx("wave_amplitude")],
            )
        elif self.initial_condition == "era5":
            state, coriolis, surface_geopotential = self.era5_problem(rows)
        else:  # "mountain"
            state, surface_geopotential = mountain_initial_condition(
                self.dynamics,
                flow_speed=rows[:, self.get_parameter_idx("flow_speed")],
                mountain_lon=rows[:, self.get_parameter_idx("mountain_lon")],
                mountain_lat=rows[:, self.get_parameter_idx("mountain_lat")],
                mountain_height=rows[:, self.get_parameter_idx("mountain_height")],
                mountain_width=rows[:, self.get_parameter_idx("mountain_width")],
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

    def periodic_forcing_frames(self, batch: int) -> TensorLike:
        """The Well's height forcing at the time of each saved frame [m/hour].

        The forcing as the solver applies it: projected on the degrees below
        ``lmax``, then truncated to the grid of the saved fields.

        Args:
            batch: Number of simulations B.

        Returns:
            F, shape (B, n_frames, *output_grid.shape, 1), float32 on the CPU.
        """
        if self.periodic is None:
            msg = "periodic_forcing_frames requires the periodic forcing"
            raise ValueError(msg)
        output_lmax = self.output_dynamics.lmax
        frames = []
        for frame in range(self.n_frames):
            time = 3600 * self.dt_save * (self.skip_nt + frame)
            forcing = periodic_height_forcing(self.dynamics, self.periodic, time)
            coefficients = torch.tril(self.dynamics.to_coefficients(forcing[None]))
            truncated = coefficients[..., :output_lmax, :output_lmax]
            field = 3600 * self.output_dynamics.to_grid(truncated)[0]
            frames.append(field.float().cpu())
        fields = torch.stack(frames).unsqueeze(-1)  # (n_frames, *grid, 1)
        # The same forcing for every simulation: a broadcast view, not B copies
        return fields.expand(batch, *fields.shape)

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
        if self.initial_condition == "shamir_paldor":
            for name in ("wave_type", "zonal_wavenumber", "mode_number"):
                column = self.get_parameter_idx(name)
                x[:, column] = x[:, column].round()  # integer parameters
        if self.initial_condition == "era5":
            column = self.get_parameter_idx("era5_time")
            index = era5_time_index(x[:, column])  # the times of the store
            x[:, column] = (ERA5_STEP_HOURS * index).to(x.dtype)

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
                rows, coriolis, surface_geopotential
            )
            forcing = self.forcing_settings(rows)
            forcing_generators = sample_generators(
                entropy, row_numbers, FORCING_STREAM, self.device
            )
            sppt_generators = sample_generators(
                entropy, row_numbers, SPPT_STREAM, self.device
            )
            n_saves = round(self.T / self.dt_save)
            branch_time = 3600 * self.dt_save * self.skip_nt  # first saved frame [s]
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
                sppt=self.sppt,
                sppt_generators=sppt_generators,
                periodic=self.periodic,
                return_additional_input_fields=self.return_additional_input_fields,
                return_energy_budget=self.return_energy_budget,
                return_forcing_state=self.return_forcing_state,
                output_dynamics=self.output_dynamics,
            )

            # Reference ensemble: N more members from the first saved frame, each
            # with its own random streams. They continue the time stepping of the
            # truth (same history), with patterns drawn from their stationary
            # distribution, as the truth's are at that frame.
            if self.reference_members > 0:
                members = []
                for member in range(self.reference_members):
                    member_generators = sample_generators(
                        entropy, row_numbers, REFERENCE_STREAM + member, self.device
                    )
                    member_sppt_generators = sample_generators(
                        entropy,
                        row_numbers,
                        REFERENCE_SPPT_STREAM + member,
                        self.device,
                    )
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
                        sppt=self.sppt,
                        sppt_generators=member_sppt_generators,
                        periodic=self.periodic,
                        start_time=branch_time,
                        previous_tendencies=output["branch_tendencies"],
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
                baseline_sppt_generators = sample_generators(
                    entropy, row_numbers, BASELINE_SPPT_STREAM, self.device
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
                    sppt=self.sppt,
                    sppt_generators=baseline_sppt_generators,
                    periodic=self.periodic,
                    start_time=branch_time,
                )
                output["baseline"] = baseline_output["data"]
            if self.return_periodic_forcing:
                output["periodic_forcing"] = self.periodic_forcing_frames(batch)
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
            ``"energy_budget"`` (n, time, 9), ``"forcing_state"`` (n, time,
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
