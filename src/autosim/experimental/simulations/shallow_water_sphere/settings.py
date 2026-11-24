r"""Settings of the physical processes of ``SphericalShallowWater``, in user units.

Each process is one argument of the simulator, and ``None`` leaves it out:

- ``DampingSettings``: Newtonian relaxation of the height and linear drag on the wind;
- ``SkebsSettings``: stochastic kinetic-energy backscatter, a vortical random pattern
  (Berner et al. 2009);
- ``GlobalBackscatterSettings`` and ``LocalBackscatterSettings``: the share of the
  dissipated kinetic energy added to the energy rate of the SKEBS pattern, as its global
  mean or as a field on the grid (Berner et al. 2009, eq. 13). The two can be combined;
- ``SpptSettings``: stochastically perturbed physics tendencies of the drag and the
  relaxation (Buizza et al. 1999; Palmer et al. 2009; Leutbecher et al. 2017);
- ``PeriodicForcingSettings``: the daily and yearly height forcing of The Well's
  PlanetSWE (Ohana et al. 2024);
- ``Era5Settings``: the store and the orography of the ``"era5"`` family.

Times are in hours here; the forcing module converts them to seconds. A field named in
``SAMPLED_FIELDS`` can be sampled per simulation: its dotted name, such as
``"skebs.energy_rate"``, is then a key of ``parameters_range`` and the value given here
is not used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from autosim.era5 import ERA5_CLIMATOLOGY_STORE, ERA5_LEVEL, ERA5_STORE

RELAXATION_TARGETS = ("jet", "era5_climatology")

# Fields that parameters_range can sample, per argument of the simulator
SAMPLED_FIELDS: dict[str, tuple[str, ...]] = {
    "damping": ("relaxation_time", "relaxation_jet_speed", "drag_time"),
    "skebs": ("energy_rate", "correlation_time", "spectral_slope"),
    "global_backscatter": ("fraction",),
    "local_backscatter": ("fraction",),
}


@dataclass(frozen=True)
class DampingSettings:
    r"""Newtonian relaxation of the height and linear drag on vorticity and divergence.

    The relaxation pulls the geopotential of the fluid depth towards an equilibrium,
    :math:`\partial_t \Phi = -(\Phi - \Phi_\text{eq})/\tau_\text{relax}`, and the drag
    damps the wind, :math:`\partial_t \zeta = -\zeta/\tau_\text{drag}` (the same for
    the divergence). With both, a jet relaxed against eddies reaches a statistically
    steady state, the forced-dissipative setting of Scott and Polvani (2007, 2008).

    Attributes:
        relaxation_time: Time of the relaxation [hours]. None is no relaxation.
        drag_time: Time of the drag [hours]. None is no drag.
        relaxation_target: ``"jet"``, the balanced height of a zonal jet with the
            profile of Galewsky et al. (2004), or ``"era5_climatology"`` (family
            ``"era5"`` only), the 1990-2019 ERA5 climatology of the free surface at the
            day of year and hour of the start date, held fixed. Both minus the
            orography.
        relaxation_jet_speed: Maximum speed of the target jet [m/s].
        relaxation_jet_latitude: Latitude of the target jet [rad].
    """

    relaxation_time: float | None = None
    drag_time: float | None = None
    relaxation_target: str = "jet"
    relaxation_jet_speed: float = 80.0
    relaxation_jet_latitude: float = math.pi / 4

    def __post_init__(self) -> None:
        """Check the target and the times."""
        if self.relaxation_target not in RELAXATION_TARGETS:
            msg = f"relaxation_target must be one of {RELAXATION_TARGETS}"
            raise ValueError(msg)
        for name in ("relaxation_time", "drag_time"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                msg = f"{name} must be positive or None"
                raise ValueError(msg)


@dataclass(frozen=True)
class SkebsSettings:
    r"""Stochastic kinetic-energy backscatter pattern, Berner et al. (2009, "B09").

    A streamfunction pattern whose spherical-harmonic coefficients follow the AR(1)
    process of B09 eq. (2), with the noise amplitude :math:`g_l = b\,l^p` on the forced
    degrees and :math:`b` set by the energy rate (B09 eq. 8, see module ``forcing``).
    The vorticity gains :math:`\nabla^2\psi'` after each solver step.

    Attributes:
        energy_rate: Kinetic energy per unit mass injected per unit time [m^2/s^3]: the
            growth rate of the global mean of |u|^2 / 2 under a free random walk of the
            increments. The backscatter settings add to it.
        correlation_time: Decorrelation time of the pattern [hours]. Zero is white
            noise.
        spectral_slope: Exponent p of g_l = b l^p. The kinetic-energy spectrum of the
            forcing follows l^(2p+3) (B09 eq. 10).
        degree_min: Lowest forced degree.
        degree_max: Highest forced degree. None is L - 1.
    """

    energy_rate: float = 0.0
    correlation_time: float = 0.0
    spectral_slope: float = -1.27
    degree_min: int = 1
    degree_max: int | None = None

    def __post_init__(self) -> None:
        """Check the rate and the time."""
        if self.energy_rate < 0 or self.correlation_time < 0:
            msg = "skebs energy_rate and correlation_time must be >= 0"
            raise ValueError(msg)


@dataclass(frozen=True)
class GlobalBackscatterSettings:
    """Backscatter of the global mean dissipation into the SKEBS energy rate (B09).

    The rate of the SKEBS pattern gains ``fraction`` times the kinetic energy per unit
    mass lost to the hyperdiffusion in each step, averaged over the sphere: one value
    per simulation.

    Attributes:
        fraction: Share of the diagnosed dissipation added to the energy rate.
        include_drag: Whether the loss to the linear drag counts as dissipation.
    """

    fraction: float
    include_drag: bool = False

    def __post_init__(self) -> None:
        """Check the fraction."""
        if self.fraction < 0:
            msg = "backscatter fraction must be non-negative"
            raise ValueError(msg)


@dataclass(frozen=True)
class LocalBackscatterSettings:
    """Backscatter of the local dissipation into the SKEBS energy rate, B09 eq. (13).

    The rate becomes a field on the grid: ``fraction`` times the kinetic energy lost to
    the hyperdiffusion in each step, smoothed to the degrees below 30 (B09 section 2b)
    and clamped at zero. The pattern is multiplied on the grid by the square root of the
    rate, which makes the forcing depend on the flow. The clamp raises the mean of the
    local rate above the global one (8 to 10 per cent at L = 86).

    Attributes:
        fraction: Share of the diagnosed dissipation added to the energy rate.
        include_drag: Whether the loss to the linear drag counts as dissipation.
    """

    fraction: float
    include_drag: bool = False

    def __post_init__(self) -> None:
        """Check the fraction."""
        if self.fraction < 0:
            msg = "backscatter fraction must be non-negative"
            raise ValueError(msg)


@dataclass(frozen=True)
class SpptSettings:
    """Stochastically perturbed physics tendencies (SPPT).

    The increment of the drag and the relaxation in each step is multiplied by
    :math:`1 + r` (Buizza et al. 1999; Palmer et al. 2009), with :math:`r` a sum of
    independent AR(1) Gaussian fields, one per scale, clipped to [-clip, clip] on the
    grid (Leutbecher et al. 2017, section 2.1). The spectrum of each field follows ECMWF
    Technical Memorandum 598 (appendix 8.1). SPPT needs the drag or the relaxation.

    Attributes:
        amplitudes: Grid-point standard deviation of each field.
        length_scales: Correlation length of each field [m].
        correlation_times: Decorrelation time of each field [hours]; 0 is white noise.
        clip: Bound of the summed pattern.
    """

    amplitudes: tuple[float, ...]
    length_scales: tuple[float, ...]
    correlation_times: tuple[float, ...]
    clip: float = 1.0

    def __post_init__(self) -> None:
        """Store the lists as tuples of floats (configs pass lists) and check them."""
        for name in ("amplitudes", "length_scales", "correlation_times"):
            object.__setattr__(self, name, tuple(float(v) for v in getattr(self, name)))
        object.__setattr__(self, "clip", float(self.clip))
        n_fields = len(self.amplitudes)
        if n_fields == 0 or {len(self.length_scales), len(self.correlation_times)} != {
            n_fields
        }:
            msg = (
                "sppt amplitudes, length_scales and correlation_times must have the "
                "same length, at least 1"
            )
            raise ValueError(msg)
        if min(self.amplitudes) <= 0 or min(self.length_scales) <= 0:
            msg = "sppt amplitudes and length scales must be positive"
            raise ValueError(msg)
        if min(self.correlation_times) < 0 or self.clip <= 0:
            msg = "sppt correlation times must be >= 0 and clip positive"
            raise ValueError(msg)


@dataclass(frozen=True)
class PeriodicForcingSettings:
    r"""Daily and yearly height forcing of The Well's PlanetSWE (Ohana et al. 2024).

    :math:`F = h_{f0}\cos(\lambda - \lambda_c)\exp[-(\varphi - \varphi_c)^2/\sigma^2]`
    with :math:`\lambda_c = 2\pi t/\text{day}` and
    :math:`\varphi_c = \delta\sin(2\pi t/\text{year})`: a source on one half of the
    sphere and a sink on the other. The defaults are those of
    ``gen_SWE_from_ic_file.py`` in PolymathicAI/the_well at commit aac88c0.

    Attributes:
        amplitude: Amplitude h_f0 of the height tendency [m/hour].
        width: Width sigma of the latitude profile [rad].
        declination: Maximum latitude of the centre [rad].
        day: Period of the longitude of the centre [hours].
        year: Period of the latitude of the centre [hours].
    """

    amplitude: float = 2.0
    width: float = 2 * math.pi / 3
    declination: float = 0.4
    day: float = 24.0
    year: float = 1008.0

    def __post_init__(self) -> None:
        """Check the amplitude and the periods."""
        if self.amplitude < 0:
            msg = "periodic amplitude must be non-negative"
            raise ValueError(msg)
        if min(self.width, self.day, self.year) <= 0:
            msg = "periodic width, day and year must be positive"
            raise ValueError(msg)


@dataclass(frozen=True)
class Era5Settings:
    """Store and orography of the ``"era5"`` family (W92 case 7 from ERA5 at 500 hPa).

    Attributes:
        store: WeatherBench2 zarr store of the ERA5 fields (Rasp et al. 2024).
        level: Pressure level [hPa].
        orography_exponent: Exponent p of the orography compression
            sign(h_s)|h_s|^p of The Well's PlanetSWE (15/16 there); 1 is no compression.
        orography_shift: Shift the compressed orography so that its minimum is 0, as
            The Well.
        climatology_store: WeatherBench2 climatology store, for the relaxation target
            ``"era5_climatology"``.
    """

    store: str = ERA5_STORE
    level: int = ERA5_LEVEL
    orography_exponent: float = 1.0
    orography_shift: bool = False
    climatology_store: str = ERA5_CLIMATOLOGY_STORE
