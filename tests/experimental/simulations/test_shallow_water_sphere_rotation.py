"""Tests of the rotation operator of the spherical shallow water simulator."""

from __future__ import annotations

import math

import pytest
import torch

th = pytest.importorskip("torch_harmonics")
if not hasattr(th, "as_grid"):
    pytest.skip("torch-harmonics without grid descriptors", allow_module_level=True)

from autosim.experimental.simulations.shallow_water_sphere import (  # noqa: E402
    SphereShallowWaterDynamics,
    adams_bashforth_step,
    coriolis_parameter,
    mountain_initial_condition,
    mountain_orography,
    rossby_haurwitz_initial_condition,
    rotate_coefficients,
    rotate_fields,
    rotate_point,
    spectral_to_fields,
    wigner_d_matrix,
    williamson2_initial_condition,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RADIUS = 6.37122e6
OMEGA = 7.292e-5
GRAVITY = 9.80616
W2_FLOW_SPEED = 2 * math.pi * RADIUS / (12 * 86400)
W2_MEAN_DEPTH = 2.94e4 / GRAVITY

# Three different angles, so that a sign error in one of them cannot hide
ANGLES = (0.7, 1.1, -0.4)


def _dynamics(lmax, nlat, nlon, mean_depth):
    grid = th.as_grid("legendre-gauss", nlat=nlat, nlon=nlon)
    dt = 0.4 * (RADIUS / lmax) / (math.sqrt(GRAVITY * mean_depth) + 80.0)
    steps_per_hour = math.ceil(3600 / dt)
    dynamics = SphereShallowWaterDynamics(
        grid, lmax, RADIUS, OMEGA, GRAVITY, mean_depth, 3600 / steps_per_hour, DEVICE
    )
    return dynamics, steps_per_hour


def _rotation_matrix(alpha, beta, gamma):
    """R = R_z(alpha) R_y(beta) R_z(gamma) as a 3 x 3 matrix."""
    rz_alpha = torch.tensor(
        [
            [math.cos(alpha), -math.sin(alpha), 0.0],
            [math.sin(alpha), math.cos(alpha), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    ry_beta = torch.tensor(
        [
            [math.cos(beta), 0.0, math.sin(beta)],
            [0.0, 1.0, 0.0],
            [-math.sin(beta), 0.0, math.cos(beta)],
        ],
        dtype=torch.float64,
    )
    rz_gamma = torch.tensor(
        [
            [math.cos(gamma), -math.sin(gamma), 0.0],
            [math.sin(gamma), math.cos(gamma), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    return rz_alpha @ ry_beta @ rz_gamma


def _test_function(x, y, z):
    """A polynomial of degree 6 in the Cartesian coordinates: band-limited."""
    return (
        0.7 * x
        + 0.3 * y * z
        + (0.5 * x - 0.2 * y + 0.4 * z) ** 3
        + (0.3 * x + 0.9 * y - 0.1 * z) ** 6
        - z**2 * x * y
    )


def _grid_points(dynamics):
    latitude = dynamics.latitude
    longitude = dynamics.longitude
    x = torch.cos(latitude) * torch.cos(longitude)
    y = torch.cos(latitude) * torch.sin(longitude)
    z = torch.sin(latitude)
    return x, y, z


def _advance(dynamics, state, steps, coriolis=None, surface_geopotential=None):
    previous = []
    for _ in range(steps):
        state, previous = adams_bashforth_step(
            dynamics, state, previous, coriolis, surface_geopotential
        )
    return state


def _random_state(lmax, cutoff, seed=0):
    generator = torch.Generator().manual_seed(seed)
    state = torch.randn(2, 3, lmax, lmax, dtype=torch.complex128, generator=generator)
    state[:, :, :, 0] = state[:, :, :, 0].real  # m = 0 coefficients are real
    state = torch.tril(state)
    state[:, :, cutoff:] = 0.0
    return state


def test_wigner_d_matrix_of_degree_one():
    beta = 0.6
    d = wigner_d_matrix(1, beta)  # indexed [m + 1, m' + 1]
    cosine, sine = math.cos(beta), math.sin(beta)
    expected = torch.tensor(
        [
            [(1 + cosine) / 2, sine / math.sqrt(2), (1 - cosine) / 2],
            [-sine / math.sqrt(2), cosine, sine / math.sqrt(2)],
            [(1 - cosine) / 2, -sine / math.sqrt(2), (1 + cosine) / 2],
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(d, expected, rtol=0.0, atol=1e-14)


@pytest.mark.parametrize("degree", [0, 1, 5, 42, 85])
def test_wigner_d_matrix_is_orthogonal(degree):
    d = wigner_d_matrix(degree, 2.3)
    identity = torch.eye(2 * degree + 1, dtype=torch.float64)
    assert (d @ d.T - identity).abs().max() < 1e-12
    # the inverse rotation is the transpose
    back = wigner_d_matrix(degree, -2.3)
    assert (back - d.T).abs().max() < 1e-12


def test_rotation_of_a_scalar_field_matches_point_evaluation():
    """The rotated field is g(x) = f(R^-1 x), evaluated here directly on the grid."""
    dynamics, _ = _dynamics(32, 48, 96, 1.0e4)
    alpha, beta, gamma = ANGLES
    x, y, z = _grid_points(dynamics)
    coefficients = dynamics.to_coefficients(_test_function(x, y, z).unsqueeze(0))

    points = torch.stack([x, y, z], dim=-1).cpu()  # (nlat, nlon, 3)
    rotation = _rotation_matrix(alpha, beta, gamma)
    back = points @ rotation  # rows are R^T x = R^-1 x
    rotated_field = _test_function(back[..., 0], back[..., 1], back[..., 2])
    expected = dynamics.to_coefficients(rotated_field.unsqueeze(0).to(DEVICE))

    rotated = rotate_coefficients(coefficients, alpha, beta, gamma)
    assert (rotated - expected).abs().max() < 1e-12 * expected.abs().max()


def test_rotation_about_the_z_axis_shifts_the_longitude():
    dynamics, _ = _dynamics(32, 48, 96, 1.0e4)
    shift = 2 * math.pi * 5 / 96  # five grid columns
    field = dynamics.to_grid(_random_state(32, 12).to(DEVICE)[:, 0])
    rotated = rotate_coefficients(dynamics.to_coefficients(field), shift, 0.0, 0.0)
    rotated_field = dynamics.to_grid(rotated)
    expected = torch.roll(field, shifts=5, dims=-1)
    assert (rotated_field - expected).abs().max() < 1e-12 * field.abs().max()


def test_rotation_and_inverse_rotation_give_back_the_state():
    alpha, beta, gamma = ANGLES
    state = _random_state(43, 43)
    rotated = rotate_coefficients(state, alpha, beta, gamma)
    assert (rotated - state).abs().max() > 0.1 * state.abs().max()
    back = rotate_coefficients(rotated, -gamma, -beta, -alpha)
    assert (back - state).abs().max() < 1e-12 * state.abs().max()


def test_rotation_keeps_the_power_of_each_degree():
    alpha, beta, gamma = ANGLES
    state = _random_state(43, 43)
    rotated = rotate_coefficients(state, alpha, beta, gamma)
    weights = torch.full((43,), 2.0, dtype=torch.float64)
    weights[0] = 1.0
    power = (state.abs() ** 2 * weights).sum(dim=-1)
    rotated_power = (rotated.abs() ** 2 * weights).sum(dim=-1)
    torch.testing.assert_close(rotated_power, power, rtol=1e-11, atol=0.0)


def test_rotate_point():
    # the geographic pole tilted by beta about the y axis goes to longitude 0
    longitude, latitude = rotate_point(0.3, math.pi / 2, 0.0, 0.4, 0.0)
    assert longitude == pytest.approx(0.0, abs=1e-12)
    assert latitude == pytest.approx(math.pi / 2 - 0.4, abs=1e-12)
    # the rotation pole of W92 case 2 with tilt alpha: beta = -alpha
    longitude, latitude = rotate_point(0.0, math.pi / 2, 0.0, -0.4, 0.0)
    assert longitude == pytest.approx(math.pi, abs=1e-12)
    assert latitude == pytest.approx(math.pi / 2 - 0.4, abs=1e-12)
    # a point of the equator turned about the z axis
    longitude, latitude = rotate_point(1.0, 0.0, 0.5, 0.0, 0.25)
    assert longitude == pytest.approx(1.75, abs=1e-12)
    assert latitude == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("tilt", [0.05, 0.7, math.pi / 2])
def test_rotating_williamson2_gives_the_tilted_state(tilt):
    """W92 case 2 at tilt 0, rotated by the tilt, is the analytic state at that tilt.

    The rotation pole of the tilted case is the geographic pole turned by
    beta = -tilt about the y axis (W92 eq. 96).
    """
    dynamics, _ = _dynamics(32, 48, 96, W2_MEAN_DEPTH)
    flow_speed = torch.tensor([W2_FLOW_SPEED], dtype=torch.float64)
    no_tilt = torch.tensor([0.0], dtype=torch.float64)
    tilted = torch.tensor([tilt], dtype=torch.float64)
    state = williamson2_initial_condition(dynamics, flow_speed, no_tilt)
    expected = williamson2_initial_condition(dynamics, flow_speed, tilted)

    rotated = rotate_coefficients(state, 0.0, -tilt, 0.0)
    geopotential_scale = expected[:, 0].abs().max()
    vorticity_scale = expected[:, 1].abs().max()
    assert (rotated[:, 0] - expected[:, 0]).abs().max() < 1e-12 * geopotential_scale
    assert (rotated[:, 1] - expected[:, 1]).abs().max() < 1e-12 * vorticity_scale
    # the flow has no divergence: compare it on the scale of the vorticity
    assert (rotated[:, 2] - expected[:, 2]).abs().max() < 1e-12 * vorticity_scale

    # the same on saved frames: h and the two wind components (W92 eqs. 90, 91, 95)
    frames = spectral_to_fields(dynamics, state)
    rotated_frames = rotate_fields(dynamics, frames, 0.0, -tilt, 0.0)
    expected_frames = spectral_to_fields(dynamics, expected)
    for channel in range(3):
        scale = expected_frames[..., channel].abs().max()
        error = (rotated_frames[..., channel] - expected_frames[..., channel]).abs()
        assert error.max() < 1e-10 * scale


def test_wind_components_do_not_rotate_as_scalars():
    """Rotating the maps of u and v as two scalars gives a wrong wind."""
    dynamics, _ = _dynamics(32, 48, 96, W2_MEAN_DEPTH)
    tilt = 0.7
    flow_speed = torch.tensor([W2_FLOW_SPEED], dtype=torch.float64)
    state = williamson2_initial_condition(
        dynamics, flow_speed, torch.tensor([0.0], dtype=torch.float64)
    )
    expected = williamson2_initial_condition(
        dynamics, flow_speed, torch.tensor([tilt], dtype=torch.float64)
    )
    frames = spectral_to_fields(dynamics, state)
    expected_frames = spectral_to_fields(dynamics, expected)

    eastward = dynamics.to_coefficients(frames[..., 1])
    as_scalar = dynamics.to_grid(rotate_coefficients(eastward, 0.0, -tilt, 0.0))
    error = (as_scalar - expected_frames[..., 1]).abs().max()
    assert error > 0.1 * W2_FLOW_SPEED

    as_spin_1 = rotate_fields(dynamics, frames, 0.0, -tilt, 0.0)
    assert (as_spin_1[..., 1] - expected_frames[..., 1]).abs().max() < 1e-9


def test_rotated_problem_matches_rotated_simulation():
    """Simulating a rotated problem gives the rotated simulation.

    The state and the rotation pole are rotated together. On a grid with exact
    products the solver has no preferred direction, so the two results agree to
    round-off amplified by 12 h of integration.
    """
    alpha, beta, gamma = ANGLES
    dynamics, steps_per_hour = _dynamics(43, 64, 128, 8000.0)
    assert dynamics.exact_products
    state = rossby_haurwitz_initial_condition(
        dynamics,
        wavenumber=torch.tensor([4.0]),
        wave_amplitude=torch.tensor([7.848e-6], dtype=torch.float64),
        mean_depth=torch.tensor([8000.0]),
    )
    steps = 12 * steps_per_hour

    simulated = _advance(dynamics, state, steps)
    rotated_simulation = rotate_coefficients(simulated, alpha, beta, gamma)

    rotated_state = rotate_coefficients(state, alpha, beta, gamma)
    pole_lon, pole_lat = rotate_point(0.0, math.pi / 2, alpha, beta, gamma)
    coriolis = coriolis_parameter(
        dynamics,
        torch.tensor([pole_lon], dtype=torch.float64),
        torch.tensor([pole_lat], dtype=torch.float64),
    )
    simulated_rotation = _advance(dynamics, rotated_state, steps, coriolis)

    # the wave has moved, so the comparison is not trivially satisfied
    assert (simulated - state)[:, 0].abs().max() > 1e-2 * state[:, 0, 1:].abs().max()
    for index in range(3):
        scale = rotated_simulation[:, index].abs().max()
        difference = (simulated_rotation[:, index] - rotated_simulation[:, index]).abs()
        assert difference.max() < 1e-9 * scale


def test_rotated_problem_with_orography_matches_rotated_simulation():
    """The same with a mountain: the orography is rotated with the state.

    The cone of W92 is defined in longitude and latitude, so the rotated orography
    is the rotation of its coefficients and not the cone at the rotated summit.
    """
    alpha, beta, gamma = ANGLES
    dynamics, steps_per_hour = _dynamics(43, 64, 128, 5960.0)
    parameters = [
        torch.tensor([value], dtype=torch.float64)
        for value in (20.0, 1.5 * math.pi, math.pi / 6, 2000.0, math.pi / 9)
    ]
    state, surface = mountain_initial_condition(dynamics, *parameters)
    steps = 6 * steps_per_hour

    simulated = _advance(dynamics, state, steps, None, surface)
    rotated_simulation = rotate_coefficients(simulated, alpha, beta, gamma)

    rotated_state = rotate_coefficients(state, alpha, beta, gamma)
    rotated_surface = rotate_coefficients(surface, alpha, beta, gamma)
    pole_lon, pole_lat = rotate_point(0.0, math.pi / 2, alpha, beta, gamma)
    coriolis = coriolis_parameter(
        dynamics,
        torch.tensor([pole_lon], dtype=torch.float64),
        torch.tensor([pole_lat], dtype=torch.float64),
    )
    simulated_rotation = _advance(
        dynamics, rotated_state, steps, coriolis, rotated_surface
    )
    for index in range(3):
        scale = rotated_simulation[:, index].abs().max()
        difference = (simulated_rotation[:, index] - rotated_simulation[:, index]).abs()
        assert difference.max() < 1e-8 * scale

    # the rotated orography differs from the W92 cone placed at the rotated summit
    summit_lon, summit_lat = rotate_point(
        1.5 * math.pi, math.pi / 6, alpha, beta, gamma
    )
    cone = mountain_orography(
        dynamics,
        torch.tensor([summit_lon], dtype=torch.float64),
        torch.tensor([summit_lat], dtype=torch.float64),
        parameters[3],
        parameters[4],
    )
    assert (cone - rotated_surface).abs().max() > 1e-3 * rotated_surface.abs().max()
