"""The spherical shallow water simulator (autosim interface)."""

from __future__ import annotations

import math
from typing import overload

import numpy as np
import torch

from autosim.era5 import (
    ERA5_STEP_HOURS,
    era5_time_index,
    load_era5_climatology,
    load_era5_fields,
)
from autosim.simulations.base import SpatioTemporalSimulator
from autosim.types import TensorLike

from .dynamics import (
    SphereShallowWaterDynamics,
    coriolis_parameter,
    make_grid,
)
from .forcing import SkebsForcing, periodic_height_forcing
from .initial_conditions import (
    ERA5_RANGES,
    GALEWSKY_RANGES,
    MOUNTAIN_RANGES,
    balanced_geopotential,
    era5_initial_condition,
    era5_source_dynamics,
    galewsky_initial_condition,
    mountain_initial_condition,
    zonal_jet,
)
from .integration import ENERGY_BUDGET_NAMES, integrate_trajectories
from .settings import (
    SAMPLED_FIELDS,
    DampingSettings,
    Era5Settings,
    GlobalBackscatterSettings,
    LocalBackscatterSettings,
    PeriodicForcingSettings,
    SkebsSettings,
    SpptSettings,
)

INITIAL_CONDITIONS = (
    "galewsky",
    "mountain",
    "era5",
)

# Default sampling ranges of each family; their keys are the required parameters.
DEFAULT_PARAMETER_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "galewsky": GALEWSKY_RANGES,
    "mountain": MOUNTAIN_RANGES,
    "era5": ERA5_RANGES,
}

# Settings fields that can also be keys of parameters_range, by their dotted name:
# they are then sampled for each simulation and returned in constant_scalars.
SAMPLED_SCALARS = tuple(
    f"{argument}.{field}"
    for argument, fields in SAMPLED_FIELDS.items()
    for field in fields
)

# Independent random streams of one simulation
INITIAL_NOISE_STREAM = 0
SKEBS_STREAM = 1
BASELINE_STREAM = 2
SPPT_STREAM = 3
BASELINE_SPPT_STREAM = 4


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

    **Initial conditions.** ``initial_condition`` selects a family, each with its own
    parameters in ``parameters_range`` (see the ``*_RANGES`` dictionaries of
    ``initial_conditions`` and ``era5``):

    - ``"galewsky"``: barotropically unstable jet with a height bump and a small
      height noise (Galewsky et al. 2004). Parameters ``jet_speed``,
      ``height_perturbation_m``, ``perturbation_lon``, ``perturbation_lat``.
    - ``"mountain"``: zonal flow over an isolated Gaussian mountain (W92 case 5 with
      the mountain of DCMIP-2008). Parameters ``flow_speed``, ``mountain_height``,
      ``mountain_width``, ``mountain_lon``, ``mountain_lat``.
    - ``"era5"``: the ERA5 state at 500 hPa over the Earth's orography, after a
      balance step (W92 case 7), read from the WeatherBench2 store of ``era5``
      (``Era5Settings``, which also sets the orography compression of The Well's
      PlanetSWE). Parameter ``era5_time``, hours since 1959-01-01T00, rounded to the
      6-hourly times of the store; the rounded value is returned in
      ``constant_scalars``.

    **Physical processes.** Each process is one argument, a dataclass of module
    ``settings``; None leaves it out. The processes can be combined, and module
    ``integration`` gives their order within a step.

    - Relaxation and drag, ``damping`` (``DampingSettings``): Newtonian relaxation of
      the height towards the balanced height of a zonal jet or towards the ERA5
      climatology, and linear drag on the vorticity and the divergence. With both,
      the flow reaches a statistically steady state with a jet and eddies, the
      forced-dissipative setting of Scott and Polvani (2007, 2008).
    - Stochastic kinetic-energy backscatter, ``skebs`` (``SkebsSettings``): a
      streamfunction pattern whose spherical-harmonic coefficients follow an AR(1)
      process, scaled to a kinetic-energy injection rate (Berner et al. 2009).
    - Backscatter of the dissipation, ``global_backscatter``
      (``GlobalBackscatterSettings``) and ``local_backscatter``
      (``LocalBackscatterSettings``): a share of the kinetic energy removed by the
      hyperdiffusion added to the SKEBS energy rate, as its global mean or as a
      smoothed field on the grid (Berner et al. 2009, section 2b and eq. 13). Both
      need ``skebs`` and can be combined.
    - Stochastically perturbed physics tendencies, ``sppt`` (``SpptSettings``): the
      increment of the drag and the relaxation multiplied by 1 + r, with r a random
      pattern (Buizza et al. 1999; Palmer et al. 2009; Leutbecher et al. 2017). Needs
      ``damping``.
    - Periodic height forcing, ``periodic`` (``PeriodicForcingSettings``): the daily
      and yearly height source and sink of The Well's PlanetSWE (Ohana et al. 2024).

    **Sampled settings.** The fields listed in ``settings.SAMPLED_FIELDS`` (drag and
    relaxation times, speed of the target jet, SKEBS rate, decorrelation time and
    slope, backscatter fractions) can be sampled for each simulation: their dotted
    name, ``"<argument>.<field>"`` such as ``"skebs.energy_rate"`` or
    ``"damping.drag_time"``, is then a key of ``parameters_range``. Their argument
    must be given, its value of that field is then not used, and the sampled values
    are returned in ``constant_scalars``.

    Args:
        lmax: Spectral band limit L.
        grid: "legendre-gauss", "equiangular", "lobatto" or "healpix".
        nlat: Rings, for regular grids.
        nlon: Points per ring, for regular grids.
        nside: Resolution, for "healpix".
        parameters_range: Bounds of the parameters of the family and of the sampled
            settings; equal bounds fix a parameter. None takes the default ranges of
            the family.
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
            Coriolis parameter. None is the geographic pole. ``"era5"`` uses the
            geographic pole.
        return_constant_fields: Return the Coriolis parameter f [1/s] and the
            orography h_s [m] on the grid as ``constant_fields``, shape
            ``(n, *grid.shape, 2)``. An emulator needs them as inputs: a rotated
            problem has a rotated f.
        sht_iterations: Jacobi iterations refining the analysis on grids without an
            exact quadrature (HEALPix), see ``SphereShallowWaterDynamics``.
        damping: Relaxation and drag, see above. None is neither.
        skebs: SKEBS pattern, see above. None is no SKEBS.
        global_backscatter: Global backscatter of the dissipation into the SKEBS
            rate. None is none.
        local_backscatter: Local backscatter of the dissipation into the SKEBS rate.
            None is none.
        sppt: SPPT, see above. None is no SPPT.
        periodic: The Well's periodic height forcing. None is no such forcing.
        era5: Store, level and orography of ``"era5"``. None takes the defaults of
            ``Era5Settings`` with that family; other families reject it.
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
            SKEBS pattern at each saved frame, for a unit energy rate. The pattern has
            memory that the saved fields do not show.
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
        damping: DampingSettings | None = None,
        skebs: SkebsSettings | None = None,
        global_backscatter: GlobalBackscatterSettings | None = None,
        local_backscatter: LocalBackscatterSettings | None = None,
        sppt: SpptSettings | None = None,
        periodic: PeriodicForcingSettings | None = None,
        era5: Era5Settings | None = None,
        return_periodic_forcing: bool = False,
        return_additional_input_fields: bool = False,
        return_energy_budget: bool = False,
        return_forcing_state: bool = False,
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
        if (global_backscatter or local_backscatter) and skebs is None:
            msg = "global_backscatter and local_backscatter require skebs"
            raise ValueError(msg)
        if return_forcing_state and skebs is None:
            msg = "return_forcing_state requires skebs"
            raise ValueError(msg)
        if return_periodic_forcing and periodic is None:
            msg = "return_periodic_forcing requires periodic"
            raise ValueError(msg)
        if era5 is not None and initial_condition != "era5":
            msg = "era5 settings require initial_condition='era5'"
            raise ValueError(msg)
        if initial_condition == "era5" and era5 is None:
            era5 = Era5Settings()
        if (
            damping is not None
            and damping.relaxation_target == "era5_climatology"
            and initial_condition != "era5"
        ):
            msg = "relaxation_target='era5_climatology' needs initial_condition='era5'"
            raise ValueError(msg)
        if output_lmax is not None and output_lmax > lmax:
            msg = "output_lmax must not exceed lmax"
            raise ValueError(msg)
        if return_baseline and output_lmax is None:
            msg = "return_baseline requires output_lmax"
            raise ValueError(msg)
        skebs_degree_max = None
        if skebs is not None:
            skebs_degree_max = (
                lmax - 1 if skebs.degree_max is None else skebs.degree_max
            )
            if not 1 <= skebs.degree_min <= skebs_degree_max <= lmax - 1:
                msg = "skebs degrees must satisfy 1 <= min <= max <= lmax - 1"
                raise ValueError(msg)
        if initial_condition == "era5" and rotation_axis is not None:
            msg = f"initial_condition='{initial_condition}' sets its own rotation axis"
            raise ValueError(msg)

        # Parameters of the family, then the sampled settings fields
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
        settings_arguments = {
            "damping": damping,
            "skebs": skebs,
            "global_backscatter": global_backscatter,
            "local_backscatter": local_backscatter,
        }
        for name in parameters_range:
            argument = name.split(".")[0]
            if name in SAMPLED_SCALARS and settings_arguments[argument] is None:
                msg = f"parameters_range samples {name!r}, which requires {argument}"
                raise ValueError(msg)
        if sppt is not None:
            has_physics = damping is not None and (
                damping.relaxation_time is not None
                or damping.drag_time is not None
                or "damping.relaxation_time" in parameters_range
                or "damping.drag_time" in parameters_range
            )
            if not has_physics:
                msg = "sppt requires damping with a relaxation_time or a drag_time"
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
        self.damping = damping
        self.skebs = skebs
        self.skebs_degree_max = skebs_degree_max
        self.global_backscatter = global_backscatter
        self.local_backscatter = local_backscatter
        self.sppt = sppt
        self.periodic = periodic
        self.era5 = era5
        self.return_periodic_forcing = return_periodic_forcing
        self.return_additional_input_fields = return_additional_input_fields
        self.return_energy_budget = return_energy_budget
        self.return_forcing_state = return_forcing_state
        self.era5_source: SphereShallowWaterDynamics | None = None  # built on use
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
        if return_baseline:
            shape = (self.n_frames, *output_shape, 3)
            self.row_layout.append(("baseline", shape))

    @overload
    def sampled_or_fixed(
        self, rows: TensorLike, name: str, value: float
    ) -> TensorLike: ...

    @overload
    def sampled_or_fixed(
        self, rows: TensorLike, name: str, value: float | None
    ) -> TensorLike | None: ...

    def sampled_or_fixed(
        self, rows: TensorLike, name: str, value: float | None
    ) -> TensorLike | None:
        """Value of one settings field for each simulation of a batch.

        The base class turns every key of ``parameters_range`` into a column of the
        parameter rows, and ``sample_inputs`` draws those rows, so the autosim command
        line samples these fields like the parameters of the family. A field of
        ``settings.SAMPLED_FIELDS`` is such a column when its dotted name (for example
        ``"skebs.energy_rate"``) is a key of ``parameters_range``. This method returns
        that column, and otherwise the value of the settings dataclass for every
        simulation, so the code that builds a batch reads one value per simulation in
        both cases. ``skebs_forcing`` and ``damping_settings`` call it for every sampled
        field; the planar ``ShallowWater2D`` makes the same choice inline for its
        scalars.

        Args:
            rows: Parameter rows (B, n_parameters).
            name: Dotted name of the field, ``"<argument>.<field>"``.
            value: Value of the field in its settings dataclass; None (no drag, no
                relaxation) stays None when the field is not sampled.

        Returns:
            Values (B,), float64, or None.
        """
        if name in self.param_names:
            return rows[:, self.get_parameter_idx(name)].to(torch.float64)
        if value is None:
            return None
        return torch.full((rows.shape[0],), value, dtype=torch.float64)

    def skebs_forcing(
        self,
        rows: TensorLike,
        dynamics: SphereShallowWaterDynamics,
        generators: list[torch.Generator],
        degree_max: int | None = None,
    ) -> SkebsForcing | None:
        """SKEBS forcing of one batch, with the values of its simulations in SI units.

        The sampled fields of ``skebs`` and of the backscatter settings take one value
        per simulation, and the decorrelation time is converted from hours to seconds.
        The forcing draws its initial pattern from ``generators``.

        Args:
            rows: Parameter rows (B, n_parameters).
            dynamics: Operators of the solver that the forcing acts on.
            generators: One random generator per simulation.
            degree_max: Highest forced degree; None is that of ``skebs``.

        Returns:
            The forcing, or None without SKEBS.
        """
        if self.skebs is None:
            return None
        assert self.skebs_degree_max is not None  # set with skebs in __init__
        correlation_hours = self.sampled_or_fixed(
            rows, "skebs.correlation_time", self.skebs.correlation_time
        )
        global_fraction = local_fraction = None
        if self.global_backscatter is not None:
            global_fraction = self.sampled_or_fixed(
                rows, "global_backscatter.fraction", self.global_backscatter.fraction
            )
        if self.local_backscatter is not None:
            local_fraction = self.sampled_or_fixed(
                rows, "local_backscatter.fraction", self.local_backscatter.fraction
            )
        return SkebsForcing(
            dynamics,
            generators,
            energy_rate=self.sampled_or_fixed(
                rows, "skebs.energy_rate", self.skebs.energy_rate
            ),
            correlation_time=3600 * correlation_hours,
            spectral_slope=self.sampled_or_fixed(
                rows, "skebs.spectral_slope", self.skebs.spectral_slope
            ),
            degree_min=self.skebs.degree_min,
            degree_max=self.skebs_degree_max if degree_max is None else degree_max,
            global_fraction=global_fraction,
            global_include_drag=bool(
                self.global_backscatter and self.global_backscatter.include_drag
            ),
            local_fraction=local_fraction,
            local_include_drag=bool(
                self.local_backscatter and self.local_backscatter.include_drag
            ),
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
        of the ERA5 climatology store at the start time of each simulation, analysed
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
        if self.damping is None:
            return None, None, None
        drag_hours = self.sampled_or_fixed(
            rows, "damping.drag_time", self.damping.drag_time
        )
        drag_time = None if drag_hours is None else 3600 * drag_hours
        relaxation_hours = self.sampled_or_fixed(
            rows, "damping.relaxation_time", self.damping.relaxation_time
        )
        relaxation_time = None if relaxation_hours is None else 3600 * relaxation_hours

        equilibrium_geopotential = None
        if relaxation_time is not None and (
            self.damping.relaxation_target == "era5_climatology"
        ):
            assert self.era5 is not None  # __init__ requires the era5 family
            hours = rows[:, self.get_parameter_idx("era5_time")]
            climatology = load_era5_climatology(
                hours, self.era5.climatology_store, self.era5.level
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
                rows, "damping.relaxation_jet_speed", self.damping.relaxation_jet_speed
            )
            jet_latitude = torch.full(
                (rows.shape[0],),
                self.damping.relaxation_jet_latitude,
                dtype=torch.float64,
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

        The fields of the sampled times are read from the store of ``era5``. The
        transforms on their grid are built at the first call and kept.

        Args:
            rows: Parameter rows (B, n_parameters); ``era5_time`` is already rounded.

        Returns:
            The spectral state (B, 3, L, L), None (the polar Coriolis parameter) and
            the coefficients of the surface geopotential (B, L, L).
        """
        assert self.era5 is not None  # set by __init__ for the era5 family
        hours = rows[:, self.get_parameter_idx("era5_time")]
        fields = load_era5_fields(hours, self.era5.store, self.era5.level)
        if self.era5_source is None:
            self.era5_source = era5_source_dynamics(self.dynamics, fields)
        state, surface_geopotential = era5_initial_condition(
            self.dynamics,
            self.era5_source,
            fields,
            self.era5.orography_exponent,
            self.era5.orography_shift,
        )
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
            skebs = self.skebs_forcing(
                rows,
                self.dynamics,
                sample_generators(entropy, row_numbers, SKEBS_STREAM, self.device),
            )
            sppt_generators = sample_generators(
                entropy, row_numbers, SPPT_STREAM, self.device
            )
            n_saves = round(self.T / self.dt_save)
            first_frame_time = 3600 * self.dt_save * self.skip_nt  # [s]
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
                skebs=skebs,
                sppt=self.sppt,
                sppt_generators=sppt_generators,
                periodic=self.periodic,
                return_additional_input_fields=self.return_additional_input_fields,
                return_energy_budget=self.return_energy_budget,
                return_forcing_state=self.return_forcing_state,
                output_dynamics=self.output_dynamics,
            )

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
                low_skebs = None
                if skebs is not None and skebs.degree_min <= low.lmax - 1:
                    low_skebs = self.skebs_forcing(
                        rows,
                        low,
                        sample_generators(
                            entropy, row_numbers, BASELINE_STREAM, self.device
                        ),
                        degree_max=min(skebs.degree_max, low.lmax - 1),
                    )
                baseline_sppt_generators = sample_generators(
                    entropy, row_numbers, BASELINE_SPPT_STREAM, self.device
                )
                baseline_output = integrate_trajectories(
                    low,
                    output["first_frame_state"][..., : low.lmax, : low.lmax],
                    n_saves=n_saves - self.skip_nt,
                    steps_per_save=self.output_steps_per_save,
                    skip_nt=0,
                    coriolis=low_coriolis,
                    surface_geopotential=low_surface,
                    drag_time=drag_time,
                    relaxation_time=relaxation_time,
                    equilibrium_geopotential=low_equilibrium,
                    skebs=low_skebs,
                    sppt=self.sppt,
                    sppt_generators=baseline_sppt_generators,
                    periodic=self.periodic,
                    start_time=first_frame_time,
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
            *grid.shape, 2), ``"periodic_forcing"`` (n, time, *grid.shape, 1) and
            ``"baseline"`` (n, time, *grid.shape, 3). With ``output_lmax`` the grid
            is that of the saved fields.
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
