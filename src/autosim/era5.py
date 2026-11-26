r"""ERA5 fields from the WeatherBench2 stores (Rasp et al. 2024).

``ERA5_STORE`` holds ERA5 at 0.25 degree, 6-hourly from 1959-01-01T00 to
2023-01-10T18, on an equiangular grid with both poles (latitudes from 90 to -90
degrees, longitudes from 0). ``load_era5_fields`` reads at one pressure level:

- the geopotential ``z``;
- ``u`` and ``v``, the eastward and northward wind;
- ``geopotential_at_surface``, :math:`g h_s`, the Earth's orography.

``load_era5_climatology`` reads the 1990-2019 climatology of WeatherBench2 (store
``ERA5_CLIMATOLOGY_STORE``, dimensions hour, dayofyear, level, latitude, longitude,
on the grid of ``ERA5_STORE``).

Times are hours since ``ERA5_START``, rounded to the 6-hourly axis of the store
(``era5_time_index``). The fields come back as float64 tensors on the CPU, on the grid
of the store.

Reading a store needs the optional ``era5`` extra (xarray, zarr, gcsfs) and network
access. The rest of the module works on arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from autosim.types import TensorLike

ERA5_STORE = "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721.zarr"
ERA5_CLIMATOLOGY_STORE = (
    "gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_1440x721.zarr"
)
ERA5_LEVEL = 500  # hPa
ERA5_START = np.datetime64("1959-01-01T00", "h")
ERA5_STEP_HOURS = 6
ERA5_TIMES = 93544  # number of times of ERA5_STORE

ERA5_MISSING = "Reading ERA5 requires xarray, zarr and gcsfs (extra 'era5')."


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


def era5_hours(date: str) -> float:
    """Hours since ``ERA5_START`` of a date, for ``parameters_range``.

    Args:
        date: ISO date and hour, for example ``"1978-12-21T00"``.

    Returns:
        Hours since 1959-01-01T00.
    """
    delta = np.datetime64(date, "h") - ERA5_START
    return float(delta / np.timedelta64(1, "h"))
