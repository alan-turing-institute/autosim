from functools import partial

import numpy as np
import pytest
import torch

from autosim.experimental.simulations.simple_precip import (
    DEFAULT_PARAMETERS,
    SimplePrecipitation,
    simulate_simple_precipitation,
)


def test_fractional_grid_advection_preserves_latent_variance():
    # Use a short correlation length to put appreciable power at Nyquist.
    params = dict(DEFAULT_PARAMETERS, ell=0.1, tau_g=0.4, cx=0.5, cy=-0.5)
    grid = {"Nx": 4, "Ny": 6, "Lx": 4, "Ly": 6, "T": 1024, "dt_out": 1, "dt_sub": 1}
    _, _, phi, _ = simulate_simple_precipitation(
        params, np.random.default_rng(17), **grid
    )
    for offset in (0, 1):  # half-cell and whole-cell shifts
        assert np.mean(phi[offset::2] ** 2) == pytest.approx(params["s"] ** 2, rel=0.04)


def test_rain_and_accumulation_on_rectangular_grid():
    # With no latent noise, the peak follows the envelope one cell per axis per step.
    params = dict(DEFAULT_PARAMETERS, s=0, theta0=0, dtheta=1, L_env=1, x0=1, y0=0)
    sim = SimplePrecipitation(
        parameters_range={k: (v, v) for k, v in params.items()},
        Nx=5,
        Ny=3,
        Lx=5,
        Ly=3,
        T=2,
        dt_out=1,
        dt_sub=1,
    )
    data = sim.forward_samples_spatiotemporal(1, random_seed=7)["data"]
    assert data.shape == (1, 2, 5, 3, 4)
    accumulation, rain, phi, envelope = data[0].unbind(dim=-1)
    assert torch.count_nonzero(phi) == 0
    torch.testing.assert_close(rain, envelope)
    assert rain[0, 2, 1] == rain[1, 3, 2] == 1
    torch.testing.assert_close(accumulation[1], (rain[0] + rain[1]) / 2)


@pytest.mark.parametrize(
    ("selection", "indices"),
    [
        ({"output_indices": [1]}, [1]),
        ({"output_names": ["a", "b"]}, [0, 1]),
        ({"output_indices": [3, 1, 2], "output_names": ["e", "r", "p"]}, [3, 1, 2]),
    ],
)
def test_selected_final_frame_matches_default_outputs(selection, indices):
    simulator = partial(
        SimplePrecipitation, Nx=5, Ny=3, Lx=5, Ly=3, T=2, dt_out=1, dt_sub=0.5
    )
    full = simulator()
    selected = simulator(**selection, return_timeseries=False)
    all_fields = full.forward_samples_spatiotemporal(1, random_seed=2)["data"]
    fields = selected.forward_samples_spatiotemporal(1, random_seed=2)["data"]
    assert selected.output_names == selection.get(
        "output_names", [full.output_names[i] for i in indices]
    )
    assert torch.equal(fields, all_fields[:, -1:, ..., indices])


@pytest.mark.parametrize("indices", [[], [1, 1], [-1], [4], [True], [1.5]])
def test_invalid_output_selection(indices):
    with pytest.raises(ValueError, match="output_indices"):
        SimplePrecipitation(output_indices=indices)


@pytest.mark.parametrize(
    "settings", [{"dt_out": 0}, {"dt_sub": -1}, {"Nx": 0}, {"m": None}]
)
def test_invalid_generation_settings(settings):
    name = next(iter(settings))
    with pytest.raises(ValueError, match=name):
        SimplePrecipitation(**settings)
    with pytest.raises(ValueError, match=name):
        simulate_simple_precipitation(
            DEFAULT_PARAMETERS, np.random.default_rng(1), **settings
        )
