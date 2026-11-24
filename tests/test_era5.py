"""Tests of the ERA5 reader of WeatherBench2 (``autosim.era5``)."""

import torch

from autosim.era5 import era5_hours, era5_time_index


def test_era5_hours_and_time_index():
    assert era5_hours("1959-01-01T00") == 0.0
    assert era5_hours("1978-12-21T00") == 175056.0  # W92 case 7
    index = era5_time_index(torch.tensor([-50.0, 2.9, 3.1, 175056.0, 1e9]))
    assert index.tolist() == [0, 0, 1, 29176, 93543]
