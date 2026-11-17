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
    rotate_coefficients,
    shallow_water_tendencies,
    spectral_to_fields,
    tilted_pole_coriolis,
    williamson3_initial_condition,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RADIUS = 6.37122e6
OMEGA = 7.292e-5
GRAVITY = 9.80616

# Published values of W92 case 3; the tilts are those of the T1 dataset, with pi/3 of
# W92's convergence study
W3_FLOW_SPEED = 2 * math.pi * RADIUS / (12 * 86400)
W3_MEAN_DEPTH = 2.94e4 / GRAVITY
W3_TILTS = [0.0, 0.05, math.pi / 3, math.pi / 2 - 0.05, math.pi / 2]
# Gaussian mountain of DCMIP-2008: half-width 1500 km
MOUNTAIN_WIDTH = 1.5e6 / RADIUS

# Tolerances of the W92 case 3 tests, each about ten times the measured value given in
# the docstring of its test
CAP_TOLERANCE = 5e-8
STEADY_TOLERANCE = 1e-10
ROTATION_TOLERANCE = 5e-10
CONVERGENCE_FACTOR = 50.0

# Parameters of each family at the published values (equal bounds)
PUBLISHED_RANGES = {
    "galewsky": {
        "jet_speed": (80.0, 80.0),
        "height_perturbation_m": (120.0, 120.0),
        "perturbation_lon": (0.0, 0.0),
        "perturbation_lat": (math.pi / 4, math.pi / 4),
    },
    "williamson3": {
        "flow_speed": (W3_FLOW_SPEED, W3_FLOW_SPEED),
        "tilt": (math.pi / 3, math.pi / 3),
    },
    "shamir_paldor": {
        "wave_type": (1.0, 1.0),
        "zonal_wavenumber": (10.0, 10.0),
        "mode_number": (5.0, 5.0),
        "wave_amplitude": (5.0, 5.0),
    },
    "mountain": {
        "flow_speed": (20.0, 20.0),
        "mountain_height": (2000.0, 2000.0),
        "mountain_width": (MOUNTAIN_WIDTH, MOUNTAIN_WIDTH),
        "mountain_lon": (1.5 * math.pi, 1.5 * math.pi),
        "mountain_lat": (math.pi / 6, math.pi / 6),
    },
}
# Families built from parameters alone; "era5" reads a store over the network and is
# tested in test_shallow_water_sphere_era5.py
GENERATED_FAMILIES = list(PUBLISHED_RANGES)

PUBLISHED_MEAN_DEPTH = {
    "galewsky": 1.0e4,
    "williamson3": W3_MEAN_DEPTH,
    "shamir_paldor": 5.0e4 / GRAVITY,
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


def _williamson3(dynamics, tilt):
    """State and Coriolis parameter of W92 case 3 for one tilt, in float64."""
    flow_speed = torch.tensor([W3_FLOW_SPEED], dtype=torch.float64)
    alpha = torch.tensor([tilt], dtype=torch.float64)
    state = williamson3_initial_condition(dynamics, flow_speed, alpha)
    return state, tilted_pole_coriolis(dynamics, alpha)


def _height(dynamics, state):
    """Height of the free surface [m] on the grid of a W92 case 3 state."""
    return spectral_to_fields(dynamics, state)[..., 0] + W3_MEAN_DEPTH


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
    assert set(GENERATED_FAMILIES) | {"era5"} == set(INITIAL_CONDITIONS)


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
            initial_condition="williamson3",
            parameters_range={"flow_speed": (20.0, 40.0)},
        )


@pytest.mark.parametrize("family", ["williamson3", "era5"])
def test_family_with_its_own_axis_rejects_a_rotation_axis(family):
    with pytest.raises(ValueError, match="rotation axis"):
        _small_simulator(initial_condition=family, rotation_axis=(0.0, 1.0))


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
    # The Gaussian of 2000 m has little power above degree 31: almost no ringing, and
    # the summit is seen by the grid point next to it (measured: maximum 1974 m,
    # minimum -5e-4 m)
    orography = fields[0, ..., 1]
    assert 1900.0 < orography.max() <= 2000.0
    assert orography.min() > -5e-3
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
        for value in (20.0, 1.5 * math.pi, math.pi / 6, 2000.0, MOUNTAIN_WIDTH)
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
        for value in (20.0, 1.5 * math.pi, math.pi / 6, 2000.0, MOUNTAIN_WIDTH)
    ]
    state, surface_geopotential = mountain_initial_condition(dynamics, *parameters)
    mass = state[0, 0, 0, 0].clone()
    state = _advance(dynamics, state, 200, None, surface_geopotential)
    torch.testing.assert_close(state[0, 0, 0, 0], mass, rtol=1e-13, atol=0.0)


def test_fields_to_spectral_inverts_spectral_to_fields():
    dynamics, _ = _dynamics(32, 48, 96, 5960.0)
    parameters = [
        torch.tensor([value], dtype=torch.float64)
        for value in (20.0, 1.5 * math.pi, math.pi / 6, 2000.0, MOUNTAIN_WIDTH)
    ]
    state, surface_geopotential = mountain_initial_condition(dynamics, *parameters)
    fields = spectral_to_fields(dynamics, state, surface_geopotential)
    back = fields_to_spectral(dynamics, fields, surface_geopotential)
    assert (back - state).abs().max() <= 1e-12 * state.abs().max()


# ---------------------------------------------------------------------------
# Published cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tilt", W3_TILTS)
def test_williamson3_has_the_published_jet_and_height(tilt):
    """W92 case 3: no divergence, a maximum speed u_0, and h = h_0 south of the jet.

    The represented height has mean h_0 over the cap theta' < -pi/6, where the wind
    of W92 is zero; it departs from h_0 there only by the ringing of the truncated jet
    (measured: at most 4.8e-9 of h_0 at lmax = 43).
    """
    dynamics, _ = _dynamics(43, 64, 128, W3_MEAN_DEPTH)
    state, _ = _williamson3(dynamics, tilt)
    # the sampled tilted jet aliases into a divergence of 2.7e-12 of the vorticity
    assert state[:, 2].abs().max() < 1e-10 * state[:, 1].abs().max()
    speed = dynamics.wind(state[:, 1], state[:, 2]).norm(dim=1)
    assert speed.max().item() == pytest.approx(W3_FLOW_SPEED, rel=1e-2)

    latitude = dynamics.latitude
    longitude = dynamics.longitude
    tilted_sine = torch.sin(latitude) * math.cos(tilt) - torch.cos(
        latitude
    ) * torch.cos(longitude) * math.sin(tilt)
    cap = tilted_sine < math.sin(-math.pi / 6)
    height = _height(dynamics, state)[0]
    weights = dynamics.quad_weights * cap
    cap_mean = (height * weights).sum() / weights.sum()
    assert cap_mean.item() == pytest.approx(W3_MEAN_DEPTH, rel=1e-12)
    assert (height[cap] - W3_MEAN_DEPTH).abs().max() < CAP_TOLERANCE * W3_MEAN_DEPTH


@pytest.mark.parametrize("tilt", W3_TILTS)
def test_williamson3_is_steady_for_every_tilt(tilt):
    """W92 case 3 at lmax = 43 without hyperdiffusion: steady for one day.

    The height is in discrete balance with the represented jet, and the jet is zonal
    about the axis of the Coriolis parameter. The change after one day is at the level
    of round-off and of the aliasing of the sampled jet (measured over the five tilts:
    at most 3e-14 for the height and 5e-12 for the vorticity).
    """
    dynamics, steps_per_hour = _dynamics(43, 64, 128, W3_MEAN_DEPTH)
    dynamics.step_damping = torch.ones_like(dynamics.step_damping)
    state, coriolis = _williamson3(dynamics, tilt)
    later = _advance(dynamics, state, 24 * steps_per_hour, coriolis)
    for index in range(2):
        change = (later[:, index] - state[:, index]).abs().max()
        assert change < STEADY_TOLERANCE * state[:, index].abs().max()


@pytest.mark.parametrize("tilt", [0.05, 0.7, math.pi / 2])
def test_tilted_williamson3_is_the_rotated_untilted_one(tilt):
    """W92 case 3 at tilt alpha is the case at tilt 0 turned by beta = -alpha.

    The two are built independently: the tilted jet from W92's rotated coordinates,
    the other by rotating the coefficients. They agree to the aliasing of the sampled
    jet (measured: at most 1.2e-11 for the height and 3.5e-11 for the vorticity). The
    Coriolis parameter of the tilted axis is the rotated polar one to round-off
    (measured: 4e-14).
    """
    dynamics, _ = _dynamics(43, 64, 128, W3_MEAN_DEPTH)
    untilted, _ = _williamson3(dynamics, 0.0)
    tilted, coriolis = _williamson3(dynamics, tilt)
    rotated = rotate_coefficients(untilted, 0.0, -tilt, 0.0)
    for index in range(2):
        error = (rotated[:, index] - tilted[:, index]).abs().max()
        assert error < ROTATION_TOLERANCE * tilted[:, index].abs().max()

    polar = dynamics.to_coefficients(dynamics.coriolis.unsqueeze(0))
    expected = dynamics.to_coefficients(coriolis)
    error = (rotate_coefficients(polar, 0.0, -tilt, 0.0) - expected).abs().max()
    assert error < 1e-12 * expected.abs().max()


@pytest.mark.slow
def test_williamson3_error_decreases_with_the_resolution():
    """Convergence study of W92 case 3 at alpha = pi/3: l2(h) at 5 days.

    Without damping the state is steady (previous tests), so the error is the damping
    of the jet by the hyperdiffusion, which acts on higher degrees as the resolution
    increases. Measured l2(h): 5.3e-4, 3.2e-6 and 1.2e-8 at lmax = 22, 43 and 86, a
    factor of at least 160 per doubling.
    """
    errors = []
    for lmax, nlat, nlon in [(22, 32, 64), (43, 64, 128), (86, 128, 256)]:
        dynamics, steps_per_hour = _dynamics(lmax, nlat, nlon, W3_MEAN_DEPTH)
        state, coriolis = _williamson3(dynamics, math.pi / 3)
        height = _height(dynamics, state)
        state = _advance(dynamics, state, 5 * 24 * steps_per_hour, coriolis)
        _, l2, _ = height_error_norms(
            _height(dynamics, state), height, dynamics.quad_weights
        )
        errors.append(l2.item())
    assert errors[1] < errors[0] / CONVERGENCE_FACTOR
    assert errors[2] < errors[1] / CONVERGENCE_FACTOR


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


def _williamson3_after_six_hours(grid_name, sht_iterations):
    """Spectral state of W92 case 3 (tilt 0.05) after 6 h through the simulator."""
    sim = _small_simulator(
        lmax=24,
        initial_condition="williamson3",
        parameters_range={
            "flow_speed": (W3_FLOW_SPEED, W3_FLOW_SPEED),
            "tilt": (0.05, 0.05),
        },
        mean_depth=W3_MEAN_DEPTH,
        sht_iterations=sht_iterations,
        batch_size=1,
        **GRIDS[grid_name],
    )
    data = sim.forward_samples_spatiotemporal(n=1, random_seed=0)["data"]
    return fields_to_spectral(sim.dynamics, data[:, -1])


@pytest.mark.parametrize(
    ("grid_name", "sht_iterations", "tolerance"),
    [
        # Exact quadrature: the difference is the float32 rounding of the saved
        # frames (measured: 2e-8).
        ("equiangular", 0, 2e-7),
        # HEALPix has no exact quadrature. The analysis of torch-harmonics is
        # approximate there (measured: 9e-4 on the vorticity), and 3 Jacobi
        # iterations bring it to 2.3e-6.
        ("healpix", 0, 1e-2),
        ("healpix", 3, 3e-5),
    ],
)
def test_williamson3_agrees_on_every_grid(grid_name, sht_iterations, tolerance):
    """W92 case 3 after 6 h on each grid equals the run on the Gauss-Legendre grid.

    The spectral state does not depend on the grid where the products are formed, as
    long as they are exact, so the coefficients of the saved frames are compared.
    """
    reference = _williamson3_after_six_hours("legendre-gauss", 0)
    state = _williamson3_after_six_hours(grid_name, sht_iterations)
    for index in range(2):
        error = (state[:, index] - reference[:, index]).abs().max()
        assert error < tolerance * reference[:, index].abs().max()
