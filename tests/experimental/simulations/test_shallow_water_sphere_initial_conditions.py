"""Tests of the initial-condition families of the spherical shallow water simulator."""

from __future__ import annotations

import math

import pytest
import torch

th = pytest.importorskip("torch_harmonics")
if not hasattr(th, "as_grid"):
    pytest.skip("torch-harmonics without grid descriptors", allow_module_level=True)

from autosim.experimental.simulations.shallow_water_sphere import (  # noqa: E402
    INITIAL_CONDITIONS,
    SphereShallowWaterDynamics,
    SphericalShallowWater,
    adams_bashforth_step,
    coriolis_parameter,
    fields_to_spectral,
    height_error_norms,
    mountain_initial_condition,
    rossby_haurwitz_initial_condition,
    shallow_water_tendencies,
    spectral_to_fields,
    williamson2_initial_condition,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RADIUS = 6.37122e6
OMEGA = 7.292e-5
GRAVITY = 9.80616

# Published values of W92 case 2
W2_FLOW_SPEED = 2 * math.pi * RADIUS / (12 * 86400)
W2_MEAN_DEPTH = 2.94e4 / GRAVITY
W2_TILTS = [0.0, 0.05, math.pi / 2 - 0.05, math.pi / 2]

# Parameters of each family at the published values (equal bounds)
PUBLISHED_RANGES = {
    "galewsky": {
        "jet_speed": (80.0, 80.0),
        "height_perturbation_m": (120.0, 120.0),
        "perturbation_lon": (0.0, 0.0),
        "perturbation_lat": (math.pi / 4, math.pi / 4),
    },
    "williamson2": {"flow_speed": (W2_FLOW_SPEED, W2_FLOW_SPEED), "tilt": (0.05, 0.05)},
    "rossby_haurwitz": {
        "wavenumber": (4.0, 4.0),
        "wave_amplitude": (7.848e-6, 7.848e-6),
        "mean_depth": (8000.0, 8000.0),
    },
    "mountain": {
        "flow_speed": (20.0, 20.0),
        "mountain_height": (2000.0, 2000.0),
        "mountain_radius": (math.pi / 9, math.pi / 9),
        "mountain_lon": (1.5 * math.pi, 1.5 * math.pi),
        "mountain_lat": (math.pi / 6, math.pi / 6),
    },
}
# Families that build their own initial state ("restart" and "from_fields" take one)
GENERATED_FAMILIES = list(PUBLISHED_RANGES)

PUBLISHED_MEAN_DEPTH = {
    "galewsky": 1.0e4,
    "williamson2": W2_MEAN_DEPTH,
    "rossby_haurwitz": 8000.0,
    "mountain": 5960.0,
}


def _small_simulator(**overrides) -> SphericalShallowWater:
    options = {
        "lmax": 32,
        "grid": "legendre-gauss",
        "nlat": 48,
        "nlon": 96,
        "T": 6.0,
        "dt_save": 3.0,
        "skip_nt": 0,
        "batch_size": 2,
        "device": DEVICE,
        "log_level": "warning",
    }
    return SphericalShallowWater(**(options | overrides))


def _dynamics(lmax, nlat, nlon, mean_depth):
    """Operators on a Gauss-Legendre grid, with a whole number of steps per hour."""
    grid = th.as_grid("legendre-gauss", nlat=nlat, nlon=nlon)
    dt = 0.4 * (RADIUS / lmax) / (math.sqrt(GRAVITY * mean_depth) + 80.0)
    steps_per_hour = math.ceil(3600 / dt)
    dynamics = SphereShallowWaterDynamics(
        grid, lmax, RADIUS, OMEGA, GRAVITY, mean_depth, 3600 / steps_per_hour, DEVICE
    )
    return dynamics, steps_per_hour


def _advance(dynamics, state, steps, coriolis=None, surface_geopotential=None):
    previous = []
    for _ in range(steps):
        state, previous = adams_bashforth_step(
            dynamics, state, previous, coriolis, surface_geopotential
        )
    return state


def _williamson2(dynamics, tilt):
    """State and Coriolis parameter of W92 case 2 for one tilt, in float64."""
    flow_speed = torch.tensor([W2_FLOW_SPEED], dtype=torch.float64)
    alpha = torch.tensor([tilt], dtype=torch.float64)
    state = williamson2_initial_condition(dynamics, flow_speed, alpha)
    axis_lon = torch.tensor([math.pi], dtype=torch.float64)
    coriolis = coriolis_parameter(dynamics, axis_lon, math.pi / 2 - alpha)
    return state, coriolis


# ---------------------------------------------------------------------------
# Families through the simulator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", GENERATED_FAMILIES)
def test_family_runs_with_default_and_published_parameters(family):
    default = _small_simulator(initial_condition=family)
    payload = default.forward_samples_spatiotemporal(n=3, random_seed=0)
    n_parameters = len(default.param_names)
    assert payload["data"].shape == (3, 3, 48, 96, 3)
    assert payload["constant_scalars"].shape == (3, n_parameters)
    assert torch.isfinite(payload["data"]).all()

    published = _small_simulator(
        initial_condition=family,
        parameters_range=PUBLISHED_RANGES[family],
        mean_depth=PUBLISHED_MEAN_DEPTH[family],
    )
    payload = published.forward_samples_spatiotemporal(n=2, random_seed=0)
    assert torch.isfinite(payload["data"]).all()
    # equal bounds: every simulation has the published parameters
    torch.testing.assert_close(
        payload["constant_scalars"][0], payload["constant_scalars"][1]
    )


def test_every_generated_family_is_tested():
    assert set(GENERATED_FAMILIES) | {"restart", "from_fields"} == set(
        INITIAL_CONDITIONS
    )


def test_default_family_is_galewsky():
    sim = _small_simulator()
    assert sim.initial_condition == "galewsky"
    assert sim.param_names == [
        "jet_speed",
        "height_perturbation_m",
        "perturbation_lon",
        "perturbation_lat",
    ]


def test_invalid_initial_condition_raises():
    with pytest.raises(ValueError, match="initial_condition must be one of"):
        _small_simulator(initial_condition="unknown")


def test_missing_family_parameter_raises():
    with pytest.raises(ValueError, match="tilt"):
        _small_simulator(
            initial_condition="williamson2",
            parameters_range={"flow_speed": (20.0, 40.0)},
        )


def test_williamson2_rejects_a_separate_rotation_axis():
    with pytest.raises(ValueError, match="rotation axis"):
        _small_simulator(initial_condition="williamson2", rotation_axis=(0.0, 1.0))


def test_noise_of_a_simulation_does_not_depend_on_the_batch_size():
    options = {"T": 0.0, "noise_amplitude": 10.0}
    one_by_one = _small_simulator(batch_size=1, **options)
    together = _small_simulator(batch_size=3, **options)
    first = one_by_one.forward_samples_spatiotemporal(n=3, random_seed=5)["data"]
    second = together.forward_samples_spatiotemporal(n=3, random_seed=5)["data"]
    torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-5)
    # the noise differs between the simulations of one call
    other = _small_simulator(
        batch_size=3, parameters_range=PUBLISHED_RANGES["galewsky"], **options
    )
    data = other.forward_samples_spatiotemporal(n=2, random_seed=5)["data"]
    assert (data[0, 0, ..., 0] - data[1, 0, ..., 0]).abs().max() > 1.0


def test_noise_has_real_zonal_coefficients():
    sim = _small_simulator(noise_amplitude=10.0)
    rows = sim.sample_inputs(2, random_seed=0)
    generators = [torch.Generator(device=DEVICE).manual_seed(seed) for seed in (1, 2)]
    state, _, _ = sim.initial_problem(rows, generators)
    assert state[:, :, :, 0].imag.abs().max() == 0.0


def test_rossby_haurwitz_wavenumber_is_rounded_and_returned():
    sim = _small_simulator(
        initial_condition="rossby_haurwitz", mean_depth=8000.0, T=0.0
    )
    payload = sim.forward_samples_spatiotemporal(n=6, random_seed=1)
    wavenumber = payload["constant_scalars"][:, sim.get_parameter_idx("wavenumber")]
    torch.testing.assert_close(wavenumber, wavenumber.round())
    assert wavenumber.min() >= 3
    assert wavenumber.max() <= 6


# ---------------------------------------------------------------------------
# Rotation axis, orography, constant fields
# ---------------------------------------------------------------------------


def test_coriolis_parameter_of_the_geographic_pole_is_the_default():
    dynamics, _ = _dynamics(32, 48, 96, 1.0e4)
    axis_lon = torch.tensor([0.3], dtype=torch.float64)
    axis_lat = torch.tensor([math.pi / 2], dtype=torch.float64)
    coriolis = coriolis_parameter(dynamics, axis_lon, axis_lat)
    torch.testing.assert_close(coriolis[0], dynamics.coriolis, rtol=0.0, atol=1e-18)


def test_coriolis_parameter_of_a_tilted_pole_is_w92_equation_96():
    dynamics, _ = _dynamics(32, 48, 96, 1.0e4)
    alpha = 0.7
    axis_lon = torch.tensor([math.pi], dtype=torch.float64)
    axis_lat = torch.tensor([math.pi / 2 - alpha], dtype=torch.float64)
    coriolis = coriolis_parameter(dynamics, axis_lon, axis_lat)
    latitude = dynamics.latitude
    longitude = dynamics.longitude
    expected = (
        2
        * OMEGA
        * (
            -torch.cos(longitude) * torch.cos(latitude) * math.sin(alpha)
            + torch.sin(latitude) * math.cos(alpha)
        )
    )
    torch.testing.assert_close(coriolis[0], expected, rtol=0.0, atol=1e-18)


def test_constant_fields_hold_coriolis_and_orography():
    sim = _small_simulator(
        initial_condition="mountain",
        parameters_range=PUBLISHED_RANGES["mountain"],
        mean_depth=5960.0,
        return_constant_fields=True,
    )
    payload = sim.forward_samples_spatiotemporal(n=2, random_seed=0)
    fields = payload["constant_fields"]
    assert fields.shape == (2, 48, 96, 2)
    coriolis = sim.dynamics.coriolis.float().cpu()
    torch.testing.assert_close(fields[0, ..., 0], coriolis)
    # The cone of 2000 m is truncated at degree 31: lower summit, small ringing
    orography = fields[0, ..., 1]
    assert 1600.0 < orography.max() < 2000.0
    assert orography.min() > -100.0
    # the summit is at longitude 3 pi / 2 and latitude pi / 6
    ilat, ilon = divmod(int(orography.argmax()), 96)
    assert abs(sim.dynamics.latitude[ilat, ilon].item() - math.pi / 6) < 0.08
    assert abs(sim.dynamics.longitude[ilat, ilon].item() - 1.5 * math.pi) < 0.08

    flat = _small_simulator(return_constant_fields=True)
    fields = flat.forward_samples_spatiotemporal(n=1, random_seed=0)["constant_fields"]
    assert fields[..., 1].abs().max() == 0.0


def test_constant_fields_are_absent_by_default():
    payload = _small_simulator().forward_samples_spatiotemporal(n=1, random_seed=0)
    assert payload["constant_fields"] is None


def test_rotation_axis_changes_the_coriolis_parameter():
    sim = _small_simulator(rotation_axis=(0.0, 0.0), return_constant_fields=True, T=0.0)
    fields = sim.forward_samples_spatiotemporal(n=1, random_seed=0)["constant_fields"]
    # pole on the equator at longitude 0: f = 2 Omega cos(lat) cos(lon)
    latitude = sim.dynamics.latitude
    longitude = sim.dynamics.longitude
    expected = 2 * OMEGA * torch.cos(latitude) * torch.cos(longitude)
    torch.testing.assert_close(
        fields[0, ..., 0], expected.float().cpu(), rtol=0.0, atol=1e-10
    )


def test_fluid_at_rest_over_a_mountain_stays_at_rest():
    dynamics, _ = _dynamics(32, 48, 96, 5960.0)
    parameters = [
        torch.tensor([value], dtype=torch.float64)
        for value in (20.0, 1.5 * math.pi, math.pi / 6, 2000.0, math.pi / 9)
    ]
    _, surface_geopotential = mountain_initial_condition(dynamics, *parameters)
    # flat free surface: the fluid depth is the mean depth minus the mountain
    state = torch.zeros(1, 3, 32, 32, dtype=torch.complex128, device=DEVICE)
    state[:, 0] = -surface_geopotential
    state[:, 0, 0, 0] += math.sqrt(4 * math.pi) * GRAVITY * 5960.0
    tendency = shallow_water_tendencies(dynamics, state, None, surface_geopotential)
    assert tendency.abs().max() < 1e-15


def test_mass_is_conserved_over_a_mountain():
    dynamics, _ = _dynamics(32, 48, 96, 5960.0)
    parameters = [
        torch.tensor([value], dtype=torch.float64)
        for value in (20.0, 1.5 * math.pi, math.pi / 6, 2000.0, math.pi / 9)
    ]
    state, surface_geopotential = mountain_initial_condition(dynamics, *parameters)
    mass = state[0, 0, 0, 0].clone()
    state = _advance(dynamics, state, 200, None, surface_geopotential)
    torch.testing.assert_close(state[0, 0, 0, 0], mass, rtol=1e-13, atol=0.0)


def test_fields_to_spectral_inverts_spectral_to_fields():
    dynamics, _ = _dynamics(32, 48, 96, 5960.0)
    parameters = [
        torch.tensor([value], dtype=torch.float64)
        for value in (20.0, 1.5 * math.pi, math.pi / 6, 2000.0, math.pi / 9)
    ]
    state, surface_geopotential = mountain_initial_condition(dynamics, *parameters)
    fields = spectral_to_fields(dynamics, state, surface_geopotential)
    back = fields_to_spectral(dynamics, fields, surface_geopotential)
    assert (back - state).abs().max() <= 1e-12 * state.abs().max()


# ---------------------------------------------------------------------------
# Published cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tilt", W2_TILTS)
def test_williamson2_is_steady_for_every_tilt(tilt):
    """W92 case 2 at lmax = 43: the height stays within 1e-10 for one day.

    The only error is the hyperdiffusion acting on the degree-1 wind, 2e-11 per day
    at this truncation (see the convergence test).
    """
    dynamics, steps_per_hour = _dynamics(43, 64, 128, W2_MEAN_DEPTH)
    state, coriolis = _williamson2(dynamics, tilt)
    height = spectral_to_fields(dynamics, state)[..., 0] + W2_MEAN_DEPTH
    state = _advance(dynamics, state, 24 * steps_per_hour, coriolis)
    height_later = spectral_to_fields(dynamics, state)[..., 0] + W2_MEAN_DEPTH
    _, l2, linf = height_error_norms(height_later, height, dynamics.quad_weights)
    assert l2 < 1e-10
    assert linf < 1e-10


@pytest.mark.slow
@pytest.mark.parametrize("tilt", [0.0, math.pi / 2])
def test_williamson2_error_decreases_at_the_rate_of_the_hyperdiffusion(tilt):
    """Convergence study of W92 case 2: l2(h) at 5 days over three resolutions.

    The truncated basis represents this low-degree state exactly, so the spatial
    discretisation has no error. The remaining error is the hyperdiffusion on the
    wind, which has degree 1: over a time t its relative damping is
    (t / 2 h) (2 / ((L - 1) L))^4. The error therefore decreases as ((L - 1) L)^-4,
    about lmax^-8, until it reaches round-off (below 1e-11 at lmax = 86).
    """
    errors = []
    for lmax, nlat, nlon in [(11, 16, 32), (22, 32, 64), (43, 64, 128), (86, 128, 256)]:
        dynamics, steps_per_hour = _dynamics(lmax, nlat, nlon, W2_MEAN_DEPTH)
        state, coriolis = _williamson2(dynamics, tilt)
        height = spectral_to_fields(dynamics, state)[..., 0] + W2_MEAN_DEPTH
        state = _advance(dynamics, state, 5 * 24 * steps_per_hour, coriolis)
        height_later = spectral_to_fields(dynamics, state)[..., 0] + W2_MEAN_DEPTH
        _, l2, _ = height_error_norms(height_later, height, dynamics.quad_weights)
        errors.append(l2.item())

    # three resolutions above round-off: measured ratio against the predicted one
    for coarse, fine, lmax_coarse, lmax_fine in [(0, 1, 11, 22), (1, 2, 22, 43)]:
        predicted_ratio = (
            ((lmax_fine - 1) * lmax_fine) / ((lmax_coarse - 1) * lmax_coarse)
        ) ** 4
        measured_ratio = errors[coarse] / errors[fine]
        assert measured_ratio == pytest.approx(predicted_ratio, rel=0.1)
    assert errors[0] < 1e-6
    assert errors[3] < 1e-11


def test_rossby_haurwitz_published_case_is_balanced():
    """W92 case 6: zero initial divergence tendency, orders 0, R and 2R only."""
    dynamics, _ = _dynamics(43, 64, 128, 8000.0)
    state = rossby_haurwitz_initial_condition(
        dynamics,
        wavenumber=torch.tensor([4.0]),
        wave_amplitude=torch.tensor([7.848e-6], dtype=torch.float64),
        mean_depth=torch.tensor([8000.0]),
    )
    tendency = shallow_water_tendencies(dynamics, state)
    scale = (dynamics.laplacian * state[0, 0]).abs().max()
    assert tendency[0, 2].abs().max() < 1e-8 * scale
    assert state[0, 2].abs().max() < 1e-15  # the wind has no divergence

    geopotential_power = state[0, 0].abs().amax(dim=0)  # per order m
    for order in range(43):
        if order not in (0, 4, 8):
            assert geopotential_power[order] < 1e-9 * geopotential_power[4]
    wind = dynamics.wind(state[:, 1], state[:, 2])
    assert 95.0 < wind.abs().max() < 105.0  # about 100 m/s


@pytest.mark.slow
def test_rossby_haurwitz_converges_to_a_high_resolution_run_at_day_7():
    """Self-convergence of W92 case 6 at day 7 (W92 reports days 0, 7 and 14).

    The reference is a run at lmax = 171, truncated to each lower band limit. The
    measured l2(h) errors are 5.9e-3, 1.3e-3 and 3.6e-4 at lmax = 22, 43 and 86: a
    factor of about 4 for each doubling of the resolution.
    """
    parameters = {
        "wavenumber": torch.tensor([4.0]),
        "wave_amplitude": torch.tensor([7.848e-6], dtype=torch.float64),
        "mean_depth": torch.tensor([8000.0]),
    }
    reference_dynamics, steps_per_hour = _dynamics(171, 256, 512, 8000.0)
    reference = rossby_haurwitz_initial_condition(reference_dynamics, **parameters)
    reference = _advance(reference_dynamics, reference, 7 * 24 * steps_per_hour)

    errors = []
    for lmax, nlat, nlon in [(22, 32, 64), (43, 64, 128), (86, 128, 256)]:
        dynamics, steps_per_hour = _dynamics(lmax, nlat, nlon, 8000.0)
        state = rossby_haurwitz_initial_condition(dynamics, **parameters)
        state = _advance(dynamics, state, 7 * 24 * steps_per_hour)
        height = spectral_to_fields(dynamics, state)[..., 0] + 8000.0
        truncated_reference = reference[:, :, :lmax, :lmax]
        height_reference = (
            spectral_to_fields(dynamics, truncated_reference)[..., 0] + 8000.0
        )
        _, l2, _ = height_error_norms(height, height_reference, dynamics.quad_weights)
        errors.append(l2.item())
    assert errors[0] < 1e-2
    assert errors[1] < errors[0] / 2
    assert errors[2] < errors[1] / 2
    assert errors[2] < 1e-3


# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------

GRIDS = {
    "legendre-gauss": {"grid": "legendre-gauss", "nlat": 36, "nlon": 72},
    "equiangular": {"grid": "equiangular", "nlat": 49, "nlon": 96},
    "healpix": {"grid": "healpix", "nlat": None, "nlon": None, "nside": 16},
}


@pytest.mark.parametrize("grid_name", list(GRIDS))
@pytest.mark.parametrize("family", GENERATED_FAMILIES)
def test_family_runs_on_every_grid(family, grid_name):
    sim = _small_simulator(
        lmax=24,
        initial_condition=family,
        parameters_range=PUBLISHED_RANGES[family],
        mean_depth=PUBLISHED_MEAN_DEPTH[family],
        return_constant_fields=True,
        **GRIDS[grid_name],
    )
    payload = sim.forward_samples_spatiotemporal(n=2, random_seed=0)
    assert payload["data"].shape == (2, 3, *sim.grid.shape, 3)
    assert payload["constant_fields"].shape == (2, *sim.grid.shape, 2)
    assert torch.isfinite(payload["data"]).all()


@pytest.mark.parametrize(
    ("grid_name", "sht_iterations", "tolerance"),
    [
        # Exact quadrature: the error is the float32 rounding of the saved frames,
        # about 1e-8 of the height.
        ("legendre-gauss", 0, 1e-7),
        ("equiangular", 0, 1e-7),
        # HEALPix has no exact quadrature. The analysis of torch-harmonics is
        # approximate there (measured error 1.5e-4), and 3 Jacobi iterations bring
        # it back to the error of the exact grids.
        ("healpix", 0, 1e-3),
        ("healpix", 3, 1e-7),
    ],
)
def test_williamson2_error_on_every_grid(grid_name, sht_iterations, tolerance):
    """W92 case 2 (tilt 0.05) through the simulator: l2(h) after 6 h, per grid."""
    sim = _small_simulator(
        lmax=24,
        initial_condition="williamson2",
        parameters_range=PUBLISHED_RANGES["williamson2"],
        mean_depth=W2_MEAN_DEPTH,
        sht_iterations=sht_iterations,
        batch_size=1,
        **GRIDS[grid_name],
    )
    data = sim.forward_samples_spatiotemporal(n=1, random_seed=0)["data"]
    height = data[:, 0, ..., 0].double().to(DEVICE) + W2_MEAN_DEPTH
    height_later = data[:, -1, ..., 0].double().to(DEVICE) + W2_MEAN_DEPTH
    _, l2, _ = height_error_norms(height_later, height, sim.dynamics.quad_weights)
    assert l2 < tolerance
