"""Tests of the diagnostics of the spherical shallow water simulator."""

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
    degree_power,
    galewsky_initial_condition,
    global_integral,
    height_error_norms,
    height_power_spectrum,
    invariants,
    kinetic_energy_spectrum,
    non_zonal_energy,
    wind_error_norms,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RADIUS = 6.37122e6
OMEGA = 7.292e-5
GRAVITY = 9.80616
MEAN_DEPTH = 1.0e4


def _dynamics(lmax=32, nlat=48, nlon=96):
    grid = th.as_grid("legendre-gauss", nlat=nlat, nlon=nlon)
    dt = 0.4 * (RADIUS / lmax) / (math.sqrt(GRAVITY * MEAN_DEPTH) + 80.0)
    steps_per_hour = math.ceil(3600 / dt)
    dynamics = SphereShallowWaterDynamics(
        grid, lmax, RADIUS, OMEGA, GRAVITY, MEAN_DEPTH, 3600 / steps_per_hour, DEVICE
    )
    return dynamics, steps_per_hour


def _random_state(lmax, seed=0):
    """A smooth random state: depth variations of 100 m, winds of a few m/s."""
    generator = torch.Generator().manual_seed(seed)
    state = torch.randn(2, 3, lmax, lmax, dtype=torch.complex128, generator=generator)
    state[:, :, :, 0] = state[:, :, :, 0].real  # m = 0 coefficients are real
    state = torch.tril(state)
    state[:, :, 12:] = 0.0
    state[:, 0] = 100.0 * GRAVITY * state[:, 0]
    state[:, 0, 0, 0] = math.sqrt(4 * math.pi) * GRAVITY * MEAN_DEPTH
    state[:, 1:] = 1e-6 * state[:, 1:]
    state[:, 1:, 0, :] = 0.0
    return state.to(DEVICE)


def _galewsky_state(dynamics):
    """The jet of the paper with its bump and without noise."""
    return galewsky_initial_condition(
        dynamics,
        jet_speed=torch.tensor([80.0]),
        height_perturbation=torch.tensor([120.0]),
        perturbation_lon=torch.tensor([0.0]),
        perturbation_lat=torch.tensor([math.pi / 4]),
        noise_amplitude=0.0,
        noise_lmax=120,
        device=DEVICE,
    )


def test_degree_power_sums_to_the_integral_of_the_square():
    dynamics, _ = _dynamics()
    state = _random_state(32)
    field = dynamics.to_grid(state[:, 0])
    integral = global_integral(field**2, dynamics.quad_weights)
    torch.testing.assert_close(degree_power(state[:, 0]).sum(dim=-1), integral)


def test_height_power_spectrum_sums_to_the_mean_square_depth():
    dynamics, _ = _dynamics()
    state = _random_state(32)
    depth = dynamics.to_grid(state[:, 0]) / GRAVITY
    mean_square = global_integral(depth**2, dynamics.quad_weights) / (4 * math.pi)
    spectrum = height_power_spectrum(state, GRAVITY)
    assert spectrum.shape == (2, 32)
    torch.testing.assert_close(spectrum.sum(dim=-1), mean_square)
    # degree 0 holds the squared mean depth
    torch.testing.assert_close(
        spectrum[:, 0], torch.full_like(spectrum[:, 0], MEAN_DEPTH**2)
    )


def test_kinetic_energy_spectrum_sums_to_the_mean_kinetic_energy():
    dynamics, _ = _dynamics()
    state = _random_state(32)
    wind = dynamics.wind(state[:, 1], state[:, 2])
    kinetic_energy = 0.5 * (wind[:, 0] ** 2 + wind[:, 1] ** 2)
    mean = global_integral(kinetic_energy, dynamics.quad_weights) / (4 * math.pi)
    spectrum = kinetic_energy_spectrum(state, RADIUS)
    assert spectrum.shape == (2, 2, 32)
    torch.testing.assert_close(spectrum.sum(dim=(1, 2)), mean)

    # a wind without divergence has no divergent part
    rotational_only = state.clone()
    rotational_only[:, 2] = 0.0
    assert kinetic_energy_spectrum(rotational_only, RADIUS)[:, 1].abs().max() == 0.0


def test_non_zonal_energy_separates_zonal_and_non_zonal_states():
    state = _random_state(32)
    assert (non_zonal_energy(state, RADIUS, GRAVITY, MEAN_DEPTH) > 0).all()
    zonal = state.clone()
    zonal[:, :, :, 1:] = 0.0
    assert non_zonal_energy(zonal, RADIUS, GRAVITY, MEAN_DEPTH).abs().max() == 0.0


def test_non_zonal_energy_is_the_linearised_energy_of_the_non_zonal_part():
    dynamics, _ = _dynamics()
    state = _random_state(32)
    non_zonal = state.clone()
    non_zonal[:, :, :, 0] = 0.0
    wind = dynamics.wind(non_zonal[:, 1], non_zonal[:, 2])
    height = dynamics.to_grid(non_zonal[:, 0]) / GRAVITY
    energy = (
        0.5 * (wind[:, 0] ** 2 + wind[:, 1] ** 2)
        + 0.5 * (GRAVITY / MEAN_DEPTH) * height**2
    )
    mean = global_integral(energy, dynamics.quad_weights) / (4 * math.pi)
    torch.testing.assert_close(
        non_zonal_energy(state, RADIUS, GRAVITY, MEAN_DEPTH), mean
    )


def test_height_error_norms_of_a_scaled_field():
    dynamics, _ = _dynamics()
    height_true = dynamics.to_grid(_random_state(32)[:, 0]) / GRAVITY
    height = 1.001 * height_true
    norms = height_error_norms(height, height_true, dynamics.quad_weights)
    for norm in norms:
        assert norm.shape == (2,)
        torch.testing.assert_close(
            norm, torch.full_like(norm, 0.001), rtol=1e-9, atol=0.0
        )


def test_wind_error_norms_are_the_same_in_a_rotated_frame():
    dynamics, _ = _dynamics()
    state = _random_state(32)
    wind_true = dynamics.wind(state[:, 1], state[:, 2])
    other = _random_state(32, seed=1)
    wind = wind_true + 0.01 * dynamics.wind(other[:, 1], other[:, 2])
    norms = wind_error_norms(wind, wind_true, dynamics.quad_weights)
    for norm in norms:
        assert norm.shape == (2,)
        assert (norm > 0).all()

    # scaling the wind by 1 + epsilon gives epsilon for the three norms
    scaled = wind_error_norms(1.001 * wind_true, wind_true, dynamics.quad_weights)
    for norm in scaled:
        torch.testing.assert_close(
            norm, torch.full_like(norm, 0.001), rtol=1e-9, atol=0.0
        )

    # the same vectors in a local basis turned by an angle that varies in space
    angle = dynamics.longitude + 2 * dynamics.latitude
    cosine, sine = torch.cos(angle), torch.sin(angle)
    turned = torch.stack(
        [
            cosine * wind[:, 0] - sine * wind[:, 1],
            sine * wind[:, 0] + cosine * wind[:, 1],
        ],
        dim=1,
    )
    turned_true = torch.stack(
        [
            cosine * wind_true[:, 0] - sine * wind_true[:, 1],
            sine * wind_true[:, 0] + cosine * wind_true[:, 1],
        ],
        dim=1,
    )
    turned_norms = wind_error_norms(turned, turned_true, dynamics.quad_weights)
    for norm, turned_norm in zip(norms, turned_norms, strict=True):
        torch.testing.assert_close(norm, turned_norm)


def test_invariants_of_a_fluid_at_rest():
    dynamics, _ = _dynamics()
    state = torch.zeros(1, 3, 32, 32, dtype=torch.complex128, device=DEVICE)
    state[:, 0, 0, 0] = math.sqrt(4 * math.pi) * GRAVITY * MEAN_DEPTH
    values = invariants(dynamics, state)[0]
    area = 4 * math.pi
    assert values[0].item() == pytest.approx(area * MEAN_DEPTH, rel=1e-12)
    assert values[1].item() == pytest.approx(
        area * 0.5 * GRAVITY * MEAN_DEPTH**2, rel=1e-12
    )
    # integral of f^2 = (2 Omega)^2 sin^2(lat) over the sphere is (2 Omega)^2 4 pi / 3
    assert values[2].item() == pytest.approx(
        (2 * OMEGA) ** 2 * area / 3 / (2 * MEAN_DEPTH), rel=1e-12
    )


def test_invariants_of_the_galewsky_jet_over_one_day():
    """Mass at round-off; energy and potential enstrophy only lost to hyperdiffusion.

    At lmax = 32 the jet is poorly resolved and the hyperdiffusion (the only sink)
    removes 1.4e-4 of the energy and 2e-2 of the potential enstrophy in one day.
    """
    dynamics, steps_per_hour = _dynamics()
    state = _galewsky_state(dynamics)
    first = invariants(dynamics, state)
    previous = []
    for _ in range(24 * steps_per_hour):
        state, previous = adams_bashforth_step(dynamics, state, previous)
    change = ((invariants(dynamics, state) - first) / first)[0]
    assert change[0].abs() < 1e-13
    assert -1e-3 < change[1] <= 0.0
    assert -0.1 < change[2] <= 0.0


@pytest.mark.slow
def test_invariants_of_the_galewsky_jet_at_the_development_resolution():
    """Validation at lmax = 86 over 48 h, before the jet breaks down.

    Mass is conserved to round-off. With hyperdiffusion as the only sink the energy
    decreases by 2.8e-7 and the potential enstrophy by 6.2e-5 (measured), within the
    tolerances 1e-6 and 1e-3.
    """
    dynamics, steps_per_hour = _dynamics(86, 128, 256)
    state = _galewsky_state(dynamics)
    first = invariants(dynamics, state)
    previous = []
    for _ in range(48 * steps_per_hour):
        state, previous = adams_bashforth_step(dynamics, state, previous)
    change = ((invariants(dynamics, state) - first) / first)[0]
    assert change[0].abs() < 1e-13
    assert -1e-6 < change[1] <= 0.0
    assert -1e-3 < change[2] <= 0.0
