from __future__ import annotations

import math
import os
from typing import Any

import numpy as np
import torch

from autosim.simulations.base import SpatioTemporalSimulator
from autosim.types import TensorLike

DEFAULT_HEIGHT_PERTURBATION_RANGE: tuple[float, float] = (80.0, 160.0)

EARTH_RADIUS_M = 6.37122e6
EARTH_OMEGA_PER_S = 7.292e-5
EARTH_GRAVITY_MPS2 = 9.80616
DEFAULT_MEAN_HEIGHT_M = 1.0e4
DEFAULT_HYPERVISCOSITY_M2PS = 1.0e5
DEFAULT_HYPERVISCOSITY_MATCH_MODE = 32

# Default per-trajectory diurnal/seasonal forcing of the height field, modelled on
# the PlanetSWE dataset (Polymathic AI, "The Well"): a Gaussian heating/cooling
# tendency that tracks the sub-solar longitude (daily) and latitude (seasonally).
DEFAULT_FORCING_RATE_M_PER_HOUR = 2.0
DEFAULT_FORCING_DAY_HOURS = 24.0
DEFAULT_FORCING_YEAR_HOURS = 1008.0  # 42-hour-per-day "year" used by PlanetSWE
DEFAULT_FORCING_DECLINATION_RAD = 0.4
DEFAULT_FORCING_SIGMA_RAD = 2.0 * math.pi / 3.0

IC_MODES = ("galewsky", "random_jet")
DEFAULT_N_JET_MODES = 4


def _import_dedalus() -> Any:
    """Import Dedalus lazily so the base autosim install stays lightweight."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
    try:
        import dedalus.public as d3  # noqa: PLC0415
    except ImportError as exc:
        msg = (
            "ShallowWaterSphereDedalus requires the optional Dedalus dependency. "
            "Dedalus must be installed in an environment with MPI, FFTW, HDF5, "
            "and an OpenMP-capable compiler. See the README section on the "
            "optional spherical shallow-water backend before running "
            "`uv sync --extra spherical`."
        )
        raise ImportError(msg) from exc
    return d3


class ShallowWaterSphereDedalus(SpatioTemporalSimulator):
    """Spherical shallow-water simulator backed by Dedalus.

    Python/Dedalus implementation of the spherical shallow-water equations,
    adapted from the Dedalus ``ivp_sphere_shallow_water`` example. Outputs are
    ``[h, u_phi, u_theta]`` on a longitude/colatitude grid, where ``h`` is the
    height perturbation in metres and the velocity components are in m/s.
    ``u_theta`` is the colatitude component and is positive southward; use
    ``-u_theta`` for northward meridional velocity. With ``include_vorticity=True``
    a fourth channel ``zeta`` (relative vorticity in 1/s) is appended, computed
    spectrally inside Dedalus as ``-div(skew(u))`` -- the same diagnostic the
    Dedalus example writes -- which is the natural field for seeing eddies,
    Rossby waves, and the Galewsky wave-breaking.

    Two initial-condition recipes are supported via ``ic_mode``:

    - ``"galewsky"`` (default): a single balanced mid-latitude jet plus a
      localized height bump (the classic Galewsky barotropic-instability test).
    - ``"random_jet"``: a randomly structured zonal jet built from low-order
      latitude and longitude modes, inspired by the PDEArena ``shallowwater``
      ``:random2`` setup. Each call draws a new jet, so a batch produces a
      diverse ensemble.

    Optionally (``forcing=True``) a PlanetSWE-style diurnal + seasonal heating
    term is added to the height equation, which keeps the flow statistically
    active over long integrations (jets, Rossby waves, eddies) rather than just
    decaying after the initial transient. This is not a reproduction of the full
    PlanetSWE generator, which also uses ERA5-derived initial conditions,
    topography, CFL-controlled stepping, and other dataset-specific choices.

    Notes
    -----
    The unforced ``"galewsky"`` case only develops its characteristic
    wave-breaking / vortex roll-up after roughly six days, so ``T`` should be at
    least ``144`` hours (the canonical Dedalus example runs ``360``). Resolution
    and hyperviscosity interact: ``hyperviscosity_match_mode`` is the spherical-
    harmonic degree at which the biharmonic damping timescale equals
    ``hyperviscosity_m2ps``-derived units; keep it well below the truncation
    degree (``~nlat``), e.g. ``nlat // 2``, or the whole resolved spectrum gets
    over-damped and the fields look bland.
    """

    def __init__(
        self,
        parameters_range: dict[str, tuple[float, float]] | None = None,
        output_names: list[str] | None = None,
        return_timeseries: bool = False,
        log_level: str = "progress_bar",
        nlon: int = 128,
        nlat: int = 64,
        T: float = 144.0,
        dt_save: float = 1.0,
        skip_nt: int = 0,
        timestep_seconds: float = 600.0,
        dealias: float = 1.5,
        radius_m: float = EARTH_RADIUS_M,
        omega_per_s: float = EARTH_OMEGA_PER_S,
        gravity_mps2: float = EARTH_GRAVITY_MPS2,
        mean_height_m: float = DEFAULT_MEAN_HEIGHT_M,
        umax_mps: float = 80.0,
        height_perturbation_m: float = 120.0,
        perturbation_lon: float = 0.0,
        perturbation_lat: float = math.pi / 4.0,
        perturbation_lon_width: float = 1.0 / 3.0,
        perturbation_lat_width: float = 1.0 / 15.0,
        perturbation_zonal_mode: int = 0,
        ic_mode: str = "galewsky",
        n_jet_modes: int = DEFAULT_N_JET_MODES,
        hyperviscosity_m2ps: float = DEFAULT_HYPERVISCOSITY_M2PS,
        hyperviscosity_match_mode: int = DEFAULT_HYPERVISCOSITY_MATCH_MODE,
        forcing: bool = False,
        forcing_day_hours: float = DEFAULT_FORCING_DAY_HOURS,
        forcing_year_hours: float = DEFAULT_FORCING_YEAR_HOURS,
        forcing_declination_rad: float = DEFAULT_FORCING_DECLINATION_RAD,
        forcing_sigma_rad: float = DEFAULT_FORCING_SIGMA_RAD,
        forcing_rate_m_per_hour: float = DEFAULT_FORCING_RATE_M_PER_HOUR,
        include_vorticity: bool = False,
        dtype: type[np.floating[Any]] = np.float64,
    ) -> None:
        if parameters_range is None:
            parameters_range = {
                "height_perturbation_m": DEFAULT_HEIGHT_PERTURBATION_RANGE
            }
        if output_names is None:
            output_names = ["h", "u_phi", "u_theta"]
            if include_vorticity:
                output_names = [*output_names, "zeta"]

        super().__init__(parameters_range, output_names, log_level)
        _validate_constructor_settings(
            skip_nt=skip_nt,
            ic_mode=ic_mode,
            n_jet_modes=n_jet_modes,
            perturbation_zonal_mode=perturbation_zonal_mode,
        )

        self.return_timeseries = return_timeseries
        self.nlon = nlon
        self.nlat = nlat
        self.T = T
        self.dt_save = dt_save
        self.skip_nt = skip_nt
        self.timestep_seconds = timestep_seconds
        self.dealias = dealias
        self.radius_m = radius_m
        self.omega_per_s = omega_per_s
        self.gravity_mps2 = gravity_mps2
        self.mean_height_m = mean_height_m
        self.umax_mps = umax_mps
        self.height_perturbation_m = height_perturbation_m
        self.perturbation_lon = perturbation_lon
        self.perturbation_lat = perturbation_lat
        self.perturbation_lon_width = perturbation_lon_width
        self.perturbation_lat_width = perturbation_lat_width
        self.perturbation_zonal_mode = perturbation_zonal_mode
        self.ic_mode = ic_mode
        self.n_jet_modes = n_jet_modes
        self.hyperviscosity_m2ps = hyperviscosity_m2ps
        self.hyperviscosity_match_mode = hyperviscosity_match_mode
        self.forcing = forcing
        self.forcing_rate_m_per_hour = forcing_rate_m_per_hour
        self.forcing_day_hours = forcing_day_hours
        self.forcing_year_hours = forcing_year_hours
        self.forcing_declination_rad = forcing_declination_rad
        self.forcing_sigma_rad = forcing_sigma_rad
        self.include_vorticity = include_vorticity
        self.dtype = dtype

    def _get_parameter_or_default(
        self, x: TensorLike, name: str, default: float
    ) -> float:
        """Read a sampled scalar parameter if configured, otherwise use default."""
        if name not in self.param_names:
            return default
        return float(x[0, self.get_parameter_idx(name)].item())

    def _forward(self, x: TensorLike) -> TensorLike:
        if x.shape[0] != 1:
            msg = "Simulator._forward expects a single input (batch size 1)"
            raise ValueError(msg)
        if x.shape[1] != self.in_dim:
            msg = (
                f"Input dim {x.shape[1]} does not match "
                f"parameters_range length {self.in_dim}"
            )
            raise ValueError(msg)

        height_perturbation_m = self._get_parameter_or_default(
            x, "height_perturbation_m", self.height_perturbation_m
        )
        umax_mps = self._get_parameter_or_default(x, "umax_mps", self.umax_mps)
        perturbation_lon = self._get_parameter_or_default(
            x, "perturbation_lon", self.perturbation_lon
        )
        forcing_rate_m_per_hour = self._get_parameter_or_default(
            x, "forcing_rate_m_per_hour", self.forcing_rate_m_per_hour
        )

        y = simulate_swe_sphere_dedalus(
            height_perturbation_m=height_perturbation_m,
            return_timeseries=self.return_timeseries,
            nlon=self.nlon,
            nlat=self.nlat,
            T=self.T,
            dt_save=self.dt_save,
            skip_nt=self.skip_nt,
            timestep_seconds=self.timestep_seconds,
            dealias=self.dealias,
            radius_m=self.radius_m,
            omega_per_s=self.omega_per_s,
            gravity_mps2=self.gravity_mps2,
            mean_height_m=self.mean_height_m,
            umax_mps=umax_mps,
            perturbation_lon=perturbation_lon,
            perturbation_lat=self.perturbation_lat,
            perturbation_lon_width=self.perturbation_lon_width,
            perturbation_lat_width=self.perturbation_lat_width,
            perturbation_zonal_mode=self.perturbation_zonal_mode,
            ic_mode=self.ic_mode,
            n_jet_modes=self.n_jet_modes,
            hyperviscosity_m2ps=self.hyperviscosity_m2ps,
            hyperviscosity_match_mode=self.hyperviscosity_match_mode,
            forcing=self.forcing,
            forcing_rate_m_per_hour=forcing_rate_m_per_hour,
            forcing_day_hours=self.forcing_day_hours,
            forcing_year_hours=self.forcing_year_hours,
            forcing_declination_rad=self.forcing_declination_rad,
            forcing_sigma_rad=self.forcing_sigma_rad,
            include_vorticity=self.include_vorticity,
            dtype=self.dtype,
        )
        return y.flatten().unsqueeze(0)

    def forward_samples_spatiotemporal(
        self,
        n: int,
        random_seed: int | None = None,
        ensure_exact_n: bool = False,
    ) -> dict:
        """Run sampled spherical trajectories as ``[batch,time,lon,lat,channels]``."""
        y, x = self._forward_batch_with_optional_retries(
            n=n,
            random_seed=random_seed,
            ensure_exact_n=ensure_exact_n,
        )
        n_valid = y.shape[0]
        channels = 4 if self.include_vorticity else 3
        features_per_step = self.nlon * self.nlat * channels

        if self.return_timeseries:
            total = y.shape[1]
            n_time = total // features_per_step
            y = y.reshape(n_valid, n_time, self.nlon, self.nlat, channels)
        else:
            y = y.reshape(n_valid, 1, self.nlon, self.nlat, channels)

        return {
            "data": y,
            "constant_scalars": x,
            "constant_fields": None,
        }


def _galewsky_jet(umax: float, lat: np.ndarray) -> np.ndarray:
    """Balanced Galewsky mid-latitude jet velocity on the colatitude grid."""
    lat0 = np.pi / 7.0
    lat1 = np.pi / 2.0 - lat0
    en = np.exp(-4.0 / (lat1 - lat0) ** 2)
    jet = (lat0 <= lat) & (lat <= lat1)
    u_phi = np.zeros_like(lat)
    u_phi[jet] = umax / en * np.exp(1.0 / (lat[jet] - lat0) / (lat[jet] - lat1))
    return u_phi


def _save_times_hours(T: float, dt_save: float) -> list[float]:
    """Return saved output times, always including the terminal time ``T``."""
    n_regular = math.floor(T / dt_save)
    save_times = [i * dt_save for i in range(n_regular + 1)]
    if not math.isclose(save_times[-1], T, rel_tol=1e-12, abs_tol=1e-12):
        save_times.append(T)
    else:
        save_times[-1] = T
    return save_times


def _validate_constructor_settings(
    skip_nt: int,
    ic_mode: str,
    n_jet_modes: int,
    perturbation_zonal_mode: int,
) -> None:
    if skip_nt < 0:
        msg = "skip_nt must be non-negative"
        raise ValueError(msg)
    if ic_mode not in IC_MODES:
        msg = f"ic_mode must be one of {IC_MODES}, got {ic_mode!r}"
        raise ValueError(msg)
    if n_jet_modes < 1:
        msg = "n_jet_modes must be >= 1"
        raise ValueError(msg)
    if perturbation_zonal_mode < 0:
        msg = "perturbation_zonal_mode must be >= 0"
        raise ValueError(msg)


def _random_zonal_jet(umax: float, lat: np.ndarray, n_jet_modes: int) -> np.ndarray:
    """Random smooth zonal jet from low-order longitude and latitude modes.

    Inspired by the PDEArena ``shallowwater`` ``:random2`` initial condition:
    low-order random modes produce longitudinal variability without injecting
    grid-scale roughness. Uses the global NumPy RNG so the surrounding batch
    seeding makes it reproducible.
    """
    lat_frac = (lat + np.pi / 2.0) / np.pi  # 0 at south pole, 1 at north pole
    nlon_local = lat.shape[0]
    lon = np.linspace(0.0, 2.0 * np.pi, nlon_local, endpoint=False)[:, None]
    max_lon_mode = max(1, min(n_jet_modes, nlon_local // 2))
    u_phi = np.zeros_like(lat)
    for lat_mode in range(n_jet_modes):
        lon_structure = np.random.standard_normal() * np.ones((nlon_local, 1))
        for lon_mode in range(1, max_lon_mode + 1):
            scale = 1.0 / lon_mode
            lon_structure = lon_structure + scale * (
                np.random.standard_normal() * np.cos(lon_mode * lon)
                + np.random.standard_normal() * np.sin(lon_mode * lon)
            )
        u_phi = u_phi + lon_structure * np.sin((lat_mode + 1) * np.pi * lat_frac)
    u_phi = u_phi * np.cos(lat)  # vanish at the poles
    peak = float(np.max(np.abs(u_phi))) + 1e-30
    return u_phi * (umax / peak)


def simulate_swe_sphere_dedalus(  # noqa: PLR0912, PLR0915
    height_perturbation_m: float,
    return_timeseries: bool,
    nlon: int,
    nlat: int,
    T: float,
    dt_save: float,
    timestep_seconds: float,
    dealias: float,
    radius_m: float,
    omega_per_s: float,
    gravity_mps2: float,
    mean_height_m: float,
    umax_mps: float,
    perturbation_lon: float,
    perturbation_lat: float,
    perturbation_lon_width: float,
    perturbation_lat_width: float,
    hyperviscosity_m2ps: float,
    hyperviscosity_match_mode: int,
    perturbation_zonal_mode: int = 0,
    ic_mode: str = "galewsky",
    n_jet_modes: int = DEFAULT_N_JET_MODES,
    forcing: bool = False,
    forcing_day_hours: float = DEFAULT_FORCING_DAY_HOURS,
    forcing_year_hours: float = DEFAULT_FORCING_YEAR_HOURS,
    forcing_declination_rad: float = DEFAULT_FORCING_DECLINATION_RAD,
    forcing_sigma_rad: float = DEFAULT_FORCING_SIGMA_RAD,
    forcing_rate_m_per_hour: float = DEFAULT_FORCING_RATE_M_PER_HOUR,
    include_vorticity: bool = False,
    dtype: type[np.floating[Any]] = np.float64,
    skip_nt: int = 0,
) -> torch.Tensor:
    """Integrate spherical shallow-water equations with Dedalus.

    Time is configured in hours for ``T`` and ``dt_save``; physical constants and
    output fields are expressed in SI units. With ``forcing=True`` a PlanetSWE-
    style diurnal + seasonal heating tendency is added to the height equation.
    With ``include_vorticity=True`` a fourth output channel ``zeta`` (relative
    vorticity in 1/s) is appended, evaluated as ``-div(skew(u))``.
    """
    if nlon <= 0 or nlat <= 0:
        msg = "nlon and nlat must be positive"
        raise ValueError(msg)
    if T < 0:
        msg = "T must be non-negative"
        raise ValueError(msg)
    if dt_save <= 0:
        msg = "dt_save must be positive"
        raise ValueError(msg)
    if timestep_seconds <= 0:
        msg = "timestep_seconds must be positive"
        raise ValueError(msg)
    if skip_nt < 0:
        msg = "skip_nt must be non-negative"
        raise ValueError(msg)
    if ic_mode not in IC_MODES:
        msg = f"ic_mode must be one of {IC_MODES}, got {ic_mode!r}"
        raise ValueError(msg)
    if hyperviscosity_match_mode <= 0:
        msg = "hyperviscosity_match_mode must be positive"
        raise ValueError(msg)
    if forcing_day_hours <= 0:
        msg = "forcing_day_hours must be positive"
        raise ValueError(msg)
    if forcing_year_hours <= 0:
        msg = "forcing_year_hours must be positive"
        raise ValueError(msg)
    if forcing_sigma_rad <= 0:
        msg = "forcing_sigma_rad must be positive"
        raise ValueError(msg)

    save_times = _save_times_hours(T, dt_save)
    expected_frames = len(save_times)
    if return_timeseries and skip_nt >= expected_frames:
        msg = (
            "skip_nt is too large for the available trajectory length; "
            f"skip_nt={skip_nt}, available_frames={expected_frames}."
        )
        raise ValueError(msg)

    d3 = _import_dedalus()

    meter = 1.0 / radius_m
    hour = 1.0
    second = hour / 3600.0

    radius = radius_m * meter
    omega = omega_per_s / second
    gravity = gravity_mps2 * meter / second**2
    mean_height = mean_height_m * meter
    timestep = timestep_seconds * second
    stop_sim_time = T * hour
    hyperviscosity = (
        hyperviscosity_m2ps * meter**2 / second / hyperviscosity_match_mode**2
    )
    umax = umax_mps * meter / second
    height_perturbation = height_perturbation_m * meter

    coords = d3.S2Coordinates("phi", "theta")
    dist = d3.Distributor(coords, dtype=dtype)
    basis = d3.SphereBasis(
        coords, (nlon, nlat), radius=radius, dealias=dealias, dtype=dtype
    )

    u = dist.VectorField(coords, name="u", bases=basis)
    h = dist.Field(name="h", bases=basis)

    def zcross(field):
        return d3.MulCosine(d3.skew(field))

    phi, theta = dist.local_grids(basis)
    lat = np.pi / 2.0 - theta + 0.0 * phi

    # Initial zonal jet.
    if ic_mode == "galewsky":
        u["g"][0] = _galewsky_jet(umax, lat)
    else:  # "random_jet"
        u["g"][0] = _random_zonal_jet(umax, lat, n_jet_modes)

    # Balanced height field: solve g*lap(h) + c = -div(u.grad(u) + 2 Omega zcross(u)).
    c = dist.Field(name="c")
    balance_problem = d3.LBVP([h, c], namespace=locals())
    balance_problem.add_equation(
        "gravity*lap(h) + c = - div(u@grad(u) + 2*omega*zcross(u))"
    )
    balance_problem.add_equation("ave(h) = 0")
    balance_solver = balance_problem.build_solver()
    balance_solver.solve()

    # Height perturbation: either a single localized bump (Galewsky) or a
    # cos(m*phi) zonal wave modulated by a mid-latitude Gaussian envelope.
    lat_envelope = np.cos(lat) * np.exp(
        -(((perturbation_lat - lat) / perturbation_lat_width) ** 2)
    )
    if perturbation_zonal_mode >= 1:
        h["g"] += (
            height_perturbation * np.cos(perturbation_zonal_mode * phi) * lat_envelope
        )
    else:
        lon_delta = np.angle(np.exp(1j * (phi - perturbation_lon)))
        h["g"] += (
            height_perturbation
            * np.exp(-((lon_delta / perturbation_lon_width) ** 2))
            * lat_envelope
        )

    # Evolution problem.
    if forcing:
        # PlanetSWE-style heating: a Gaussian couplet (warm on the day side,
        # cool on the night side) centred on the sub-solar point, which drifts
        # in longitude every "day" and in latitude every "year".
        day = forcing_day_hours * hour
        year = forcing_year_hours * hour
        sigma = forcing_sigma_rad
        max_declination = forcing_declination_rad
        forcing_rate = forcing_rate_m_per_hour * meter / hour

        t = dist.Field(name="t")
        phi_field = dist.Field(name="phi_field", bases=basis)
        phi_field["g"] += phi + 0.0 * lat
        lat_field = dist.Field(name="lat_field", bases=basis)
        lat_field["g"] += lat + 0.0 * phi

        lon_center = (t / day) * 2.0 * np.pi
        lat_center = np.sin((t / year) * 2.0 * np.pi) * max_declination
        h_forcing = (
            forcing_rate
            * np.cos(phi_field - lon_center)
            * np.exp(-((lat_field - lat_center) ** 2) / sigma**2)
        )

        problem = d3.IVP([u, h], namespace=locals(), time=t)
        problem.add_equation(
            "dt(u) + hyperviscosity*lap(lap(u)) + gravity*grad(h) "
            "+ 2*omega*zcross(u) = - u@grad(u)"
        )
        problem.add_equation(
            "dt(h) + hyperviscosity*lap(lap(h)) + mean_height*div(u) "
            "= - div(h*u) + h_forcing"
        )
    else:
        problem = d3.IVP([u, h], namespace=locals())
        problem.add_equation(
            "dt(u) + hyperviscosity*lap(lap(u)) + gravity*grad(h) "
            "+ 2*omega*zcross(u) = - u@grad(u)"
        )
        problem.add_equation(
            "dt(h) + hyperviscosity*lap(lap(h)) + mean_height*div(u) = - div(h*u)"
        )

    solver = problem.build_solver(d3.RK222)
    solver.stop_sim_time = stop_sim_time

    velocity_unit = meter / second
    vorticity_unit = 1.0 / second
    vorticity_op = -d3.div(d3.skew(u)) if include_vorticity else None

    def snapshot() -> torch.Tensor:
        h.change_scales(1)
        u.change_scales(1)
        h_m = np.asarray(h["g"] / meter)
        u_phi_mps = np.asarray(u["g"][0] / velocity_unit)
        u_theta_mps = np.asarray(u["g"][1] / velocity_unit)
        fields = [h_m, u_phi_mps, u_theta_mps]
        if vorticity_op is not None:
            zeta = vorticity_op.evaluate()
            zeta.change_scales(1)
            fields.append(np.asarray(zeta["g"] / vorticity_unit))
        data = np.stack(fields, axis=-1)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.from_numpy(np.ascontiguousarray(data)).float()

    if not return_timeseries:
        while solver.sim_time < stop_sim_time - 1e-12:
            dt = min(timestep, stop_sim_time - solver.sim_time)
            solver.step(dt)
        return snapshot().unsqueeze(0)

    save_times = [time * hour for time in save_times]
    history: list[torch.Tensor] = []
    next_save_idx = 0

    if save_times and save_times[0] == 0.0:
        history.append(snapshot())
        next_save_idx = 1

    while solver.sim_time < stop_sim_time - 1e-12:
        target_time = (
            save_times[next_save_idx]
            if next_save_idx < len(save_times)
            else stop_sim_time
        )
        dt = min(
            timestep,
            target_time - solver.sim_time,
            stop_sim_time - solver.sim_time,
        )
        if dt <= 1e-12:
            if next_save_idx < len(save_times):
                history.append(snapshot())
                next_save_idx += 1
                continue
            break

        solver.step(dt)
        if (
            next_save_idx < len(save_times)
            and solver.sim_time >= save_times[next_save_idx] - 1e-10
        ):
            history.append(snapshot())
            next_save_idx += 1

    while len(history) < expected_frames:
        history.append(snapshot())

    return torch.stack(history[skip_nt:expected_frames], dim=0)
