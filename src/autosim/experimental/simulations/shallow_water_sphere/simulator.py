"""The spherical shallow water simulator (autosim interface)."""

from __future__ import annotations

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
from .initial_conditions import (
    GALEWSKY_RANGES,
    GALEWSKY_UNPERTURBED_RANGES,
    MOUNTAIN_RANGES,
    ROSSBY_HAURWITZ_RANGES,
    WILLIAMSON2_RANGES,
    galewsky_initial_condition,
    mountain_initial_condition,
    rossby_haurwitz_initial_condition,
    williamson2_initial_condition,
)
from .integration import integrate_trajectories

INITIAL_CONDITIONS = (
    "galewsky",
    "galewsky_unperturbed",
    "williamson2",
    "rossby_haurwitz",
    "mountain",
)

# Default sampling ranges of each family; their keys are the required parameters.
DEFAULT_PARAMETER_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "galewsky": GALEWSKY_RANGES,
    "galewsky_unperturbed": GALEWSKY_UNPERTURBED_RANGES,
    "williamson2": WILLIAMSON2_RANGES,
    "rossby_haurwitz": ROSSBY_HAURWITZ_RANGES,
    "mountain": MOUNTAIN_RANGES,
}

# Independent random streams of one simulation
INITIAL_NOISE_STREAM = 0


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
            ``"galewsky"`` [m].
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
        initial_condition: str = "galewsky",
        rotation_axis: tuple[float, float] | None = None,
        return_constant_fields: bool = False,
        sht_iterations: int = 0,
    ) -> None:
        """Store the configuration and build the spectral operators once."""
        sphere_grid = make_grid(grid, nlat, nlon, nside)  # needs torch-harmonics
        if initial_condition not in INITIAL_CONDITIONS:
            msg = f"initial_condition must be one of {INITIAL_CONDITIONS}"
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

        # Content of one flat output row: (key, shape of one simulation). The extra
        # outputs travel in the row so that the retries of the base class keep them
        # aligned with the data.
        self.row_layout: list[tuple[str, tuple[int, ...]]] = [
            ("data", (self.n_frames, *self.grid.shape, len(self.output_names)))
        ]
        if return_constant_fields:
            self.row_layout.append(("constant_fields", (*self.grid.shape, 2)))

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
        """Coriolis parameter and orography on the grid, shape (B, *grid.shape, 2)."""
        if coriolis is None:
            coriolis = self.dynamics.coriolis.expand(batch, *self.grid.shape)
        if surface_geopotential is None:
            surface_height = torch.zeros_like(coriolis)
        else:
            surface_height = (
                self.dynamics.to_grid(surface_geopotential) / self.dynamics.gravity
            )
        return torch.stack([coriolis, surface_height], dim=-1)

    def forward_batch(
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
            output = integrate_trajectories(
                self.dynamics,
                state,
                n_saves=round(self.T / self.dt_save),
                steps_per_save=self.steps_per_save,
                skip_nt=self.skip_nt,
                coriolis=coriolis,
                surface_geopotential=surface_geopotential,
            )
            if self.return_constant_fields:
                fields = self.constant_fields_on_grid(
                    batch, coriolis, surface_geopotential
                )
                output["constant_fields"] = fields.float().cpu()

            flat_outputs = []
            for key, _ in self.row_layout:
                flat_outputs.append(output[key].flatten(start_dim=1))
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
            n_parameters), "constant_fields": (n, *grid.shape, 2) or None}``.
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
