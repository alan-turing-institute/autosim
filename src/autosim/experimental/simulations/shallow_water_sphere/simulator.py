"""The spherical shallow water simulator (autosim interface)."""

from __future__ import annotations

import math

import torch

from autosim.simulations.base import SpatioTemporalSimulator
from autosim.types import TensorLike

from .dynamics import (
    SphereShallowWaterDynamics,
    adams_bashforth_step,
    make_grid,
    spectral_to_fields,
)
from .initial_conditions import GALEWSKY_RANGES, galewsky_initial_condition


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
        sphere_grid = make_grid(grid, nlat, nlon, nside)  # needs torch-harmonics
        if parameters_range is None:
            parameters_range = dict(GALEWSKY_RANGES)
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
