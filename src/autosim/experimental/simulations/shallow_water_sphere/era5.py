r"""Initial states from ERA5 at 500 hPa, the analysed fields of W92 case 7.

W92 case 7 starts the shallow water equations from the analysed 500 mb height and
wind. This module reads them from the WeatherBench2 copy of ERA5 (Rasp et al. 2024,
store ``ERA5_STORE``: 0.25 degree, 6-hourly from 1959-01-01T00 to 2023-01-10T18,
latitudes from 90 to -90 degrees, longitudes from 0), with the surface geopotential:

- the geopotential ``z`` at 500 hPa is the geopotential of the free surface,
  :math:`\Phi + g h_s`, with :math:`\Phi` the geopotential of the fluid depth;
- ``u`` and ``v`` are the eastward and northward wind;
- ``geopotential_at_surface`` is :math:`g h_s`, the Earth's orography.

``era5_initial_condition`` analyses these fields on the equiangular grid of the store
(poles included), keeps the degrees below the band limit of the solver, and applies
the balance step of ``balance_state``. W92 case 7 has no orography. With the Earth's
orography the fluid depth :math:`z/g - h_s` must stay positive, while the ground of
Tibet (5859 m on the 0.25 degree grid) nearly reaches the 500 hPa surface. As in the
PlanetSWE dataset of The Well (Ohana et al. 2024; ``gen_ics.ipynb`` at commit aac88c0
of PolymathicAI/the_well), the orography can be compressed,
:math:`h_s \to \operatorname{sign}(h_s)|h_s|^p` with :math:`p = 15/16`, and shifted so
that its minimum is 0 (``compress_orography``).

``load_era5_climatology`` reads the 1990-2019 climatology of WeatherBench2 (store
``ERA5_CLIMATOLOGY_STORE``, dimensions hour, dayofyear, level, latitude, longitude,
on the grid of ``ERA5_STORE``), the relaxation target of the ERA5 weather dataset.

Reading the store needs the optional ``era5`` extra (xarray, zarr, gcsfs) and network
access. The rest of the module works on arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from autosim.types import TensorLike

from .dynamics import SphereShallowWaterDynamics, make_grid
from .initial_conditions import balance_state

ERA5_STORE = "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721.zarr"
ERA5_CLIMATOLOGY_STORE = (
    "gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_1440x721.zarr"
)
# Exponent of the orography compression of The Well (PlanetSWE, gen_ics.ipynb)
ERA5_OROGRAPHY_EXPONENT = 15 / 16
ERA5_LEVEL = 500  # hPa
ERA5_START = np.datetime64("1959-01-01T00", "h")
ERA5_STEP_HOURS = 6
ERA5_TIMES = 93544  # number of times of ERA5_STORE

# era5_time: hours since ERA5_START, rounded to the 6-hourly axis. W92 case 7 dates:
# 1978-12-21T00, 1979-01-09T00 and 1979-01-16T00 (see era5_hours).
ERA5_RANGES: dict[str, tuple[float, float]] = {
    "era5_time": (0.0, float(ERA5_STEP_HOURS * (ERA5_TIMES - 1))),
}

ERA5_MISSING = "The era5 family requires xarray, zarr and gcsfs (extra 'era5')."


@dataclass
class Era5Fields:
    """ERA5 fields of B times on the equiangular grid of the store.

    Attributes:
        geopotential: Geopotential z at the pressure level [m^2/s^2], (B, nlat, nlon).
        eastward: Eastward wind u [m/s], (B, nlat, nlon).
        northward: Northward wind v [m/s], (B, nlat, nlon).
        surface_geopotential: Surface geopotential g h_s [m^2/s^2], (nlat, nlon).
        latitude: Latitudes of the rows [rad], from pi/2 to -pi/2, (nlat,).
        longitude: Longitudes of the columns [rad], from 0, (nlon,).
    """

    geopotential: TensorLike
    eastward: TensorLike
    northward: TensorLike
    surface_geopotential: TensorLike
    latitude: TensorLike
    longitude: TensorLike


def era5_time_index(hours: TensorLike) -> TensorLike:
    """Index of the time of the store nearest to ``hours`` since 1959-01-01T00.

    Args:
        hours: Hours since ``ERA5_START``, shape (B,).

    Returns:
        Indices on the 6-hourly axis, clipped to the store, shape (B,), int64.
    """
    index = torch.round(torch.as_tensor(hours, dtype=torch.float64) / ERA5_STEP_HOURS)
    return index.long().clamp(0, ERA5_TIMES - 1)


def _open_store(store: str):
    """Open a zarr store with xarray; a ``gs://`` path is read anonymously."""
    try:
        import xarray as xr  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise ImportError(ERA5_MISSING) from error

    storage_options = {"token": "anon"} if store.startswith("gs://") else None
    return xr.open_zarr(store, storage_options=storage_options, chunks=None)


def load_era5_fields(
    hours: TensorLike, store: str = ERA5_STORE, level: int = ERA5_LEVEL
) -> Era5Fields:
    """Read the fields of the times nearest to ``hours`` from a WeatherBench2 store.

    Args:
        hours: Hours since ``ERA5_START``, shape (B,).
        store: Path of the zarr store (WeatherBench2 layout: dimensions time, level,
            latitude, longitude). A ``gs://`` path is read anonymously.
        level: Pressure level [hPa].

    Returns:
        The fields, float64 on the CPU.
    """
    dataset = _open_store(store)
    index = era5_time_index(hours).tolist()
    expected = ERA5_START + np.asarray(index) * np.timedelta64(ERA5_STEP_HOURS, "h")
    times = dataset.time.values[index].astype("datetime64[h]")
    if not (times == expected).all():
        msg = f"the store {store} is not 6-hourly from {ERA5_START}"
        raise ValueError(msg)

    names = ["geopotential", "u_component_of_wind", "v_component_of_wind"]
    selected = dataset[names].isel(time=index).sel(level=level)
    fields = [
        torch.from_numpy(np.asarray(selected[name].values, dtype=np.float64))
        for name in names
    ]
    surface = np.asarray(dataset["geopotential_at_surface"].values, dtype=np.float64)
    latitude = np.deg2rad(np.asarray(dataset.latitude.values, dtype=np.float64))
    longitude = np.deg2rad(np.asarray(dataset.longitude.values, dtype=np.float64))
    return Era5Fields(
        geopotential=fields[0],
        eastward=fields[1],
        northward=fields[2],
        surface_geopotential=torch.from_numpy(surface),
        latitude=torch.from_numpy(latitude),
        longitude=torch.from_numpy(longitude),
    )


def load_era5_climatology(
    hours: TensorLike,
    store: str = ERA5_CLIMATOLOGY_STORE,
    level: int = ERA5_LEVEL,
) -> TensorLike:
    """Climatological geopotential at the day of year and hour of each time.

    The times are rounded to the 6-hourly axis of ``ERA5_STORE``. The day of year
    counts from 1 on 1 January, as in the store.

    Args:
        hours: Hours since ``ERA5_START``, shape (B,).
        store: Path of the climatology store (dimensions hour, dayofyear, level,
            latitude, longitude).
        level: Pressure level [hPa].

    Returns:
        Geopotential [m^2/s^2], (B, nlat, nlon), float64 on the CPU, on the grid of
        the store (latitudes from 90 to -90 degrees).
    """
    dataset = _open_store(store)
    index = era5_time_index(hours).numpy()
    times = ERA5_START + index * np.timedelta64(ERA5_STEP_HOURS, "h")
    days = (times.astype("datetime64[D]") - times.astype("datetime64[Y]")).astype(
        int
    ) + 1
    hours_of_day = (times - times.astype("datetime64[D]")).astype(int)
    fields = [
        np.asarray(
            dataset["geopotential"]
            .sel(level=level, hour=int(hour), dayofyear=int(day))
            .values,
            dtype=np.float64,
        )
        for hour, day in zip(hours_of_day, days, strict=True)
    ]
    return torch.from_numpy(np.stack(fields))


def era5_source_dynamics(
    dynamics: SphereShallowWaterDynamics, fields: Era5Fields
) -> SphereShallowWaterDynamics:
    """Transforms on the grid of the ERA5 fields, at the band limit of ``dynamics``.

    The grid is torch-harmonics' equiangular grid with poles; its points must be those
    of the fields, which is checked.

    Args:
        dynamics: Operators of the solver.
        fields: ERA5 fields.

    Returns:
        Operators on the grid of the fields.
    """
    nlat, nlon = fields.geopotential.shape[-2:]
    grid = make_grid("equiangular", nlat, nlon, None)
    source = SphereShallowWaterDynamics(
        grid,
        dynamics.lmax,
        dynamics.radius,
        dynamics.omega,
        dynamics.gravity,
        dynamics.mean_depth,
        dynamics.dt,
        str(dynamics.laplacian.device),
    )
    grid_latitude = source.latitude[:, 0].cpu()
    grid_longitude = source.longitude[0].cpu()
    latitude_error = (grid_latitude - fields.latitude).abs().max().item()
    longitude_error = (grid_longitude - fields.longitude).abs().max().item()
    if latitude_error > 1e-6 or longitude_error > 1e-6:
        msg = (
            "the ERA5 fields are not on torch-harmonics' equiangular grid with poles "
            f"(latitude error {latitude_error:.2e}, longitude error "
            f"{longitude_error:.2e} rad)"
        )
        raise ValueError(msg)
    return source


def compress_orography(
    surface_geopotential: TensorLike,
    gravity: float,
    exponent: float = 1.0,
    shift: bool = False,
) -> TensorLike:
    r"""Compressed orography of The Well's PlanetSWE, on the grid of the fields.

    The height :math:`h_s = \Phi_s / g` [m] becomes
    :math:`\operatorname{sign}(h_s)|h_s|^p`, and with ``shift`` the result is shifted
    so that its minimum is 0 (``gen_ics.ipynb`` of PolymathicAI/the_well at aac88c0).
    With :math:`p = 15/16` the 5859 m of Tibet become 3407 m. ``exponent=1`` and
    ``shift=False`` leave the orography unchanged.

    Args:
        surface_geopotential: g h_s on the grid of the fields [m^2/s^2].
        gravity: g [m/s^2].
        exponent: p.
        shift: Shift the compressed orography so that its minimum is 0.

    Returns:
        The surface geopotential of the compressed orography, same shape.
    """
    height = surface_geopotential / gravity
    height = torch.sign(height) * height.abs() ** exponent
    if shift:
        height = height - height.min()
    return gravity * height


def era5_initial_condition(
    dynamics: SphereShallowWaterDynamics,
    source: SphereShallowWaterDynamics,
    fields: Era5Fields,
    orography_exponent: float = 1.0,
    orography_shift: bool = False,
) -> tuple[TensorLike, TensorLike]:
    """Balanced spectral state and orography from ERA5 fields, W92 case 7.

    The fields are analysed on their grid (``source``) and the degrees below the band
    limit kept. The orography is first compressed with ``compress_orography``. The
    fluid depth is the free surface minus the orography. The balance
    step removes the divergence and puts the free surface in nonlinear balance with
    the vorticity, with the global mean of the analysed free surface.

    Args:
        dynamics: Operators of the solver.
        source: Operators on the grid of the fields (``era5_source_dynamics``).
        fields: ERA5 fields of B times.
        orography_exponent: Exponent p of ``compress_orography``.
        orography_shift: Shift of ``compress_orography``.

    Returns:
        The spectral state (B, 3, L, L) and the coefficients of the surface
        geopotential g h_s, shape (B, L, L).
    """
    device = source.laplacian.device
    batch = fields.geopotential.shape[0]
    geopotential = fields.geopotential.to(device, torch.float64)
    surface = compress_orography(
        fields.surface_geopotential.to(device, torch.float64),
        dynamics.gravity,
        orography_exponent,
        orography_shift,
    )
    wind = torch.stack(
        [
            fields.eastward.to(device, torch.float64),
            fields.northward.to(device, torch.float64),
        ],
        dim=1,
    )

    free_surface = source.to_coefficients(geopotential)
    surface_geopotential = source.to_coefficients(surface.expand(batch, *surface.shape))
    surface_geopotential = torch.tril(surface_geopotential)
    vorticity, divergence = source.vorticity_divergence(wind)
    state = torch.stack(
        [free_surface - surface_geopotential, vorticity, divergence], dim=1
    )
    state = balance_state(dynamics, torch.tril(state), None, surface_geopotential)
    return state, surface_geopotential


def era5_hours(date: str) -> float:
    """Hours since ``ERA5_START`` of a date, for ``parameters_range``.

    Args:
        date: ISO date and hour, for example ``"1978-12-21T00"``.

    Returns:
        Hours since 1959-01-01T00.
    """
    delta = np.datetime64(date, "h") - ERA5_START
    return float(delta / np.timedelta64(1, "h"))
