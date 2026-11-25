from functools import partial

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from omegaconf import OmegaConf

from autosim.experimental.precipitation_redraws import export_redraws
from autosim.experimental.simulations.simple_precip import (
    DEFAULT_PARAMETERS,
    simulate_simple_precipitation,
)

simulate_ou = partial(
    simulate_simple_precipitation,
    temporal_model="ou",
    Nx=8,
    Ny=6,
    Lx=8,
    Ly=6,
    dt_out=1,
    dt_sub=0.25,
)


@pytest.mark.parametrize("broad_front", [False, True])
def test_ou_restart_matches_uninterrupted_trajectory(broad_front):
    params = dict(DEFAULT_PARAMETERS, ell=0.2, cx=0.5, cy=-0.25, theta0=0.5)
    simulate = partial(simulate_ou, params, broad_front=broad_front)
    whole = simulate(np.random.default_rng(9), T=4)
    rng = np.random.default_rng(9)
    _, _, phi, _ = simulate(rng, T=2)
    restarted = simulate(rng, T=2, initial_phi=phi[-1], start_time=2)
    for field, reference in zip(restarted, whole, strict=True):
        np.testing.assert_allclose(field, reference[2:], atol=1e-12)


@pytest.mark.parametrize(
    "config_name",
    [
        "generate_data_simple_precip_ou_64",
        "generate_data_simple_precip_variable_64",
        "generate_data_simple_precip_broad_front_64",
    ],
)
def test_saved_frame_restart_requires_its_time(config_name):
    with initialize_config_module(config_module="autosim.configs", version_base=None):
        cfg = compose(config_name=config_name)
    sim = instantiate(cfg.simulator, Nx=5, Ny=3, T=2 * cfg.simulator.dt_out)
    split = sim.forward_samples_spatiotemporal(1, random_seed=7)
    frames = split["data"][0]
    assert frames.shape == (2, 5, 3, 4)
    draw = partial(
        sim.simulate_trajectory,
        split["constant_scalars"][0],
        np.random.default_rng(8),
        n_frames=1,
        initial_phi=frames[0, ..., 2],
    )
    with pytest.raises(ValueError, match="start_time"):
        draw()
    np.testing.assert_allclose(draw(start_time=sim.dt_out)[3][0], frames[1, ..., 3])


@pytest.mark.parametrize("indices", [[0, 1, 2, 3], [2, 1, 0]])
def test_export_redraws_from_saved_frame(tmp_path, indices):
    with initialize_config_module(config_module="autosim.configs", version_base=None):
        cfg = compose(
            config_name="generate_data_simple_precip_ou_64",
            overrides=[
                "simulator.Nx=8",
                "simulator.Ny=6",
                "simulator.T=20",
                "simulator.parameters_range.theta0=[0,0]",
                f"simulator.output_indices={indices}",
                "dataset.output_dir=.",
            ],
        )
    split = instantiate(cfg.simulator).forward_samples_spatiotemporal(1, random_seed=7)
    (tmp_path / "test").mkdir()
    torch.save(split, tmp_path / "test/data.pt")
    OmegaConf.save(cfg, tmp_path / "resolved_config.yaml", resolve=True)

    output = tmp_path / "redraws"
    export_redraws(tmp_path, output, [(0, 1)], members=2, leads=2, seed=3)
    anchors = torch.load(output / "anchors.pt")
    states = torch.load(output / "reference/anchor_000.pt")["states"]
    saved = split["data"][0]
    assert anchors["output_indices"] == indices
    assert torch.equal(anchors["states"][0], saved[1])
    assert torch.equal(anchors["observed_futures"][0], saved[2:])
    for member in states:
        assert torch.equal(member[0], saved[1])
        if 3 in indices:
            idx = indices.index(3)
            torch.testing.assert_close(member[1:, ..., idx], saved[2:, ..., idx])
    for channel in (indices.index(0), indices.index(1)):
        assert not torch.equal(states[0, 1:, ..., channel], states[1, 1:, ..., channel])


@pytest.mark.parametrize(
    ("config_name", "width_bounds", "heading_deg"),
    [
        ("generate_data_simple_precip_variable_64", (16.0, 24.0), None),
        ("generate_data_simple_precip_broad_front_64", (18.0, 22.0), 148.0),
    ],
)
def test_shared_velocity_and_envelope_width(config_name, width_bounds, heading_deg):
    with initialize_config_module(config_module="autosim.configs", version_base=None):
        cfg = compose(config_name=config_name)
    sim = instantiate(cfg.simulator)
    n = 32
    rows = sim.sample_inputs(n, random_seed=7).numpy()
    np.testing.assert_array_equal(rows, sim.sample_inputs(n, random_seed=7).numpy())
    cx, cy, cex, cey, width = rows[
        :, [sim.get_parameter_idx(name) for name in ("cx", "cy", "cex", "cey", "L_env")]
    ].T
    np.testing.assert_array_equal(cx, cex)
    np.testing.assert_array_equal(cy, cey)
    stratified = [(np.hypot(cx, cy), (1.0, 1.8)), (width, width_bounds)]
    heading = np.arctan2(cy, cx)
    if heading_deg is None:
        stratified.append((heading, (-np.pi, np.pi)))
    else:
        np.testing.assert_allclose(np.rad2deg(heading), heading_deg, atol=1e-4)
    for values, bounds in stratified:
        counts, _ = np.histogram(values, bins=n, range=bounds)
        assert (counts == 1).all()


def test_ou_stationary_field_and_innovations():
    params = dict(DEFAULT_PARAMETERS, s=1.7, tau_g=2, ell=0.4, cx=2, cy=-2)
    _, _, phi, _ = simulate_ou(params, np.random.default_rng(31), T=384)
    # Four half-cell shifts translate by two cells, including for Nyquist modes.
    advected = np.roll(phi[:-1], (2, -2), axis=(1, 2))
    decay = np.exp(-1 / params["tau_g"])
    innovation = phi[1:] - decay * advected
    variance = params["s"] ** 2
    assert np.mean(phi**2) == pytest.approx(variance, rel=0.08)
    assert np.mean(innovation**2) == pytest.approx(variance * (1 - decay**2), rel=0.06)
    assert abs(np.mean(innovation * advected)) < 0.06 * variance

    kx = 2 * np.pi * np.fft.fftfreq(8)
    power = np.exp(-0.5 * (params["ell"] * kx) ** 2)
    adjacent_correlation = np.average(np.cos(kx), weights=power)
    observed = np.mean(innovation * np.roll(innovation, 1, axis=1))
    assert observed / np.mean(innovation**2) == pytest.approx(
        adjacent_correlation, abs=0.04
    )


def test_ou_restart_accumulates_rain_from_forecast_start():
    params = dict(
        DEFAULT_PARAMETERS,
        s=0,
        tau_g=2,
        x0=0,
        y0=0,
        cex=1,
        cey=-1,
        L_env=1,
        theta0=1,
        dtheta=0.5,
    )
    accumulation, rain, phi, envelope = simulate_ou(
        params,
        np.random.default_rng(4),
        initial_phi=np.full((7, 5), 1.25),
        start_time=2,
        Nx=7,
        Ny=5,
        Lx=7,
        Ly=5,
        T=1,
        dt_sub=0.5,
    )
    # At (3, 2) the periodic envelope centre approaches by half a cell per axis.
    lead = np.array([0, 0.5, 1])
    local_phi = 1.25 * np.exp(-lead / params["tau_g"])
    local_envelope = np.exp(-((1 - lead) ** 2))
    local_rain = np.maximum(local_phi - (1 - 0.5 * local_envelope), 0)
    assert envelope[0, 3, 2] == pytest.approx(local_envelope[-1])
    assert phi[0, 3, 2] == pytest.approx(local_phi[-1])
    assert rain[0, 3, 2] == pytest.approx(local_rain[-1])
    assert accumulation[0, 3, 2] == pytest.approx(
        0.25 * (local_rain[0] + 2 * local_rain[1] + local_rain[2])
    )
