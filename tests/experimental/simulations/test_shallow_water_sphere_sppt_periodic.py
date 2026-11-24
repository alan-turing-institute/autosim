"""Tests of SPPT and of The Well's periodic forcing on the sphere."""

from __future__ import annotations

import math

import pytest
import torch

th = pytest.importorskip("torch_harmonics")
if not hasattr(th, "as_grid"):
    pytest.skip("torch-harmonics without grid descriptors", allow_module_level=True)

from autosim.experimental.simulations.shallow_water_sphere import (  # noqa: E402
    ENERGY_BUDGET_NAMES,
    DampingSettings,
    PeriodicForcingSettings,
    SphereShallowWaterDynamics,
    SphericalShallowWater,
    SpptSettings,
    advance_pattern,
    complex_noise,
    galewsky_initial_condition,
    integrate_trajectories,
    periodic_forcing_increment,
    periodic_height_forcing,
    sppt_increment,
    sppt_noise_amplitudes,
    sppt_pattern_on_grid,
    stationary_pattern,
    total_energy,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RADIUS = 6.37122e6
OMEGA = 7.292e-5
GRAVITY = 9.80616
MEAN_DEPTH = 1.0e4
LMAX = 22
DT = 300.0
HOUR = 3600.0
RATE = ENERGY_BUDGET_NAMES.index("effective_forcing_energy_rate")
CHANGES = slice(1, RATE)
SPPT_COLUMN = ENERGY_BUDGET_NAMES.index("sppt_energy_change")
PERIODIC_COLUMN = ENERGY_BUDGET_NAMES.index("periodic_energy_change")

# The three scales of ECMWF's SPPT (Leutbecher et al. 2017, Table 1), times in hours
ECMWF_SETTINGS = SpptSettings(
    (0.52, 0.18, 0.06),
    (500.0e3, 1000.0e3, 2000.0e3),
    (6.0, 72.0, 720.0),
)
ECMWF_SPPT = {"sppt": ECMWF_SETTINGS}
# One field of 500 km and 6 h
ONE_FIELD = SpptSettings((0.52,), (500.0e3,), (6.0,))
# Relative sampling error of the grid-point standard deviation of eight patterns
# (measured: 0.6 and 1.8 per cent, see the test)
SPPT_STD_TOLERANCE = 0.05
# The forcing of The Well (gen_SWE_from_ic_file.py at commit aac88c0): m/hour, hours
THE_WELL = PeriodicForcingSettings(
    amplitude=2.0,
    width=2 * math.pi / 3,
    declination=0.4,
    day=24.0,
    year=1008.0,
)
FIXED_GALEWSKY = {
    "jet_speed": (80.0, 80.0),
    "height_perturbation_m": (120.0, 120.0),
    "perturbation_lon": (0.0, 0.0),
    "perturbation_lat": (math.pi / 4, math.pi / 4),
}
RELAXED_CLIMATE = {
    "damping": DampingSettings(
        relaxation_time=24.0, relaxation_jet_speed=80.0, drag_time=2400.0
    ),
}


def _dynamics(lmax=LMAX, nlat=32, nlon=64, dt=DT):
    grid = th.as_grid("legendre-gauss", nlat=nlat, nlon=nlon)
    return SphereShallowWaterDynamics(
        grid, lmax, RADIUS, OMEGA, GRAVITY, MEAN_DEPTH, dt, DEVICE
    )


def _generators(batch, first_seed=0):
    return [
        torch.Generator(device=DEVICE).manual_seed(first_seed + index)
        for index in range(batch)
    ]


def _galewsky_state(dynamics, batch):
    ones = torch.ones(batch, dtype=torch.float64)
    return galewsky_initial_condition(
        dynamics,
        80.0 * ones,
        120.0 * ones,
        0.0 * ones,
        math.pi / 4 * ones,
        noise_amplitude=0.0,
        noise_lmax=1,
        device=DEVICE,
    )


def _small_simulator(**overrides) -> SphericalShallowWater:
    options = {
        "lmax": LMAX,
        "grid": "legendre-gauss",
        "nlat": 32,
        "nlon": 64,
        "T": 6.0,
        "dt_save": 3.0,
        "skip_nt": 0,
        "batch_size": 2,
        "device": DEVICE,
        "log_level": "warning",
        "initial_condition": "galewsky",
        "parameters_range": FIXED_GALEWSKY,
    }
    return SphericalShallowWater(**(options | overrides))


def _sppt_patterns(dynamics, sppt, batch, first_seed=0):
    """Stationary SPPT fields of a batch, with their amplitudes and AR(1) parameter."""
    times = HOUR * torch.tensor(sppt.correlation_times, dtype=torch.float64)
    alpha = (dynamics.dt / times).to(DEVICE)
    n_fields = len(sppt.amplitudes)
    amplitudes = sppt_noise_amplitudes(dynamics, sppt, alpha)
    amplitudes = amplitudes.expand(batch, n_fields, dynamics.lmax)
    alpha = alpha.expand(batch, n_fields)
    generators = _generators(batch, first_seed)
    noise = complex_noise(generators, n_fields, dynamics.lmax)
    pattern = stationary_pattern(noise, amplitudes, alpha)
    return pattern, amplitudes, alpha, generators


def _grid_variance(dynamics, field):
    """Area mean of field^2 for each simulation, (B,)."""
    return (field**2 * dynamics.quad_weights).sum(dim=(-2, -1)) / (4 * math.pi)


# ---------------------------------------------------------------------------
# SPPT pattern
# ---------------------------------------------------------------------------


def test_sppt_pattern_has_the_grid_point_variance_of_its_scales():
    """The fields have the grid-point standard deviation sigma_k of TM 598 eq. 18.

    One field of 500 km and the three scales of ECMWF together, at lmax = 86 for
    eight simulations. Measured: 0.523 for a target of 0.520, and 0.543 for 0.554;
    the large scales of the second have few independent samples on the sphere.
    """
    dynamics = _dynamics(86, 128, 256)
    for sppt in (ONE_FIELD, ECMWF_SETTINGS):
        pattern, _, _, _ = _sppt_patterns(dynamics, sppt, batch=8)
        field = dynamics.to_grid(pattern.sum(dim=1))
        std = _grid_variance(dynamics, field).mean().sqrt()
        target = math.sqrt(sum(value**2 for value in sppt.amplitudes))
        assert std.item() == pytest.approx(target, rel=SPPT_STD_TOLERANCE)


def test_sppt_pattern_follows_its_ar1_process():
    """The lag-one correlation of a field is 1 - alpha, the AR(1) step of B09 eq. 2."""
    dynamics = _dynamics(43, 64, 128, dt=600.0)
    sppt = SpptSettings((0.5,), (500.0e3,), (6.0,))
    pattern, amplitudes, alpha, generators = _sppt_patterns(dynamics, sppt, batch=8)
    before = dynamics.to_grid(pattern.sum(dim=1))
    noise = complex_noise(generators, 1, dynamics.lmax)
    advanced = advance_pattern(pattern, noise, amplitudes, alpha)
    after = dynamics.to_grid(advanced.sum(dim=1))
    weights = dynamics.quad_weights
    correlation = (before * after * weights).sum() / (before**2 * weights).sum()
    assert correlation.item() == pytest.approx(1 - 600.0 / (6 * HOUR), abs=0.01)


def test_sppt_pattern_is_clipped():
    dynamics = _dynamics()
    sppt = SpptSettings((5.0,), (500.0e3,), (6.0,), clip=1.0)
    pattern, _, _, _ = _sppt_patterns(dynamics, sppt, batch=2)
    grid = sppt_pattern_on_grid(dynamics, pattern, sppt.clip)
    assert grid.abs().max() == 1.0
    assert (grid.abs() == 1.0).float().mean() > 0.1  # the amplitude 5 is clipped often


def test_sppt_increment_multiplies_the_physics_by_the_pattern():
    """With a uniform pattern r = c the perturbation is c times the physics increment.

    The wind is exact; the height loses its global mean, which keeps the mass.
    """
    dynamics = _dynamics()
    physics = _galewsky_state(dynamics, 2) * 1e-3
    shape = (2, *dynamics.grid.shape)
    constant = torch.full(shape, 0.3, dtype=torch.float64, device=DEVICE)
    increment = sppt_increment(dynamics, constant, physics)
    expected = 0.3 * physics
    expected[:, 0, 0, 0] = 0.0
    for index in range(3):
        scale = physics[:, index].abs().max() + 1e-30
        assert (increment[:, index] - expected[:, index]).abs().max() < 1e-12 * scale


# ---------------------------------------------------------------------------
# SPPT in the time loop
# ---------------------------------------------------------------------------


def _relaxed_run(batch, sppt=None, periodic=None, budget=False, steps=24):
    dynamics = _dynamics()
    state = _galewsky_state(dynamics, batch)
    equilibrium = state[:, 0].clone()
    state[:, 0] = state[:, 0] * 1.001  # away from the equilibrium of the relaxation
    output = integrate_trajectories(
        dynamics,
        state,
        n_saves=2,
        steps_per_save=steps // 2,
        skip_nt=0,
        drag_time=torch.full((batch,), 5 * 86400.0),
        relaxation_time=torch.full((batch,), 86400.0),
        equilibrium_geopotential=equilibrium,
        sppt=sppt,
        sppt_generators=_generators(batch, 100) if sppt is not None else None,
        periodic=periodic,
        return_energy_budget=budget,
        return_additional_input_fields=True,
    )
    return dynamics, output


def test_sppt_requires_drag_or_relaxation():
    dynamics = _dynamics()
    with pytest.raises(ValueError, match="drag or relaxation"):
        integrate_trajectories(
            dynamics,
            _galewsky_state(dynamics, 1),
            n_saves=1,
            steps_per_save=1,
            skip_nt=0,
            sppt=ONE_FIELD,
            sppt_generators=_generators(1),
        )
    with pytest.raises(ValueError, match="relaxation_time or a drag_time"):
        _small_simulator(**ECMWF_SPPT)


def test_sppt_keeps_the_mass_and_changes_the_flow():
    """SPPT increments have no global mean height; the relaxation sets the mass."""
    _, plain = _relaxed_run(2)
    _, perturbed = _relaxed_run(2, sppt=ECMWF_SETTINGS)
    mass = plain["final_state"][:, 0, 0, 0]
    torch.testing.assert_close(
        perturbed["final_state"][:, 0, 0, 0], mass, rtol=1e-13, atol=0.0
    )
    difference = (perturbed["final_state"] - plain["final_state"])[:, 0]
    assert difference.abs().max() > 1e-6 * plain["final_state"][:, 0, 1:].abs().max()
    assert perturbed["additional_input_fields"][:, 0, ..., 0].abs().max() > 0


def test_energy_budget_closes_with_sppt_and_the_periodic_forcing():
    dynamics, output = _relaxed_run(2, sppt=ONE_FIELD, periodic=THE_WELL, budget=True)
    budget = output["energy_budget"]
    assert budget.shape == (2, 3, len(ENERGY_BUDGET_NAMES))
    change = budget[:, 1:, 0] - budget[:, :-1, 0]
    terms = budget[:, :-1, CHANGES].sum(dim=-1)
    assert (change - terms).abs().max() < 1e-12 * budget[:, :, 0].abs().max()
    assert (budget[:, :-1, SPPT_COLUMN] != 0).all()
    assert (budget[:, :-1, PERIODIC_COLUMN] != 0).all()
    final_energy = total_energy(dynamics, output["final_state"])
    torch.testing.assert_close(budget[:, -1, 0], final_energy.cpu())


def test_sppt_through_the_simulator_is_reproducible_and_seeded():
    options = {"T": 6.0, "batch_size": 2} | RELAXED_CLIMATE | ECMWF_SPPT

    def run(seed, **changes):
        sim = _small_simulator(**(options | changes))
        return sim.forward_samples_spatiotemporal(n=2, random_seed=seed)

    first = run(4)
    again = run(4)
    other = run(5)
    single = run(4, batch_size=1)
    plain = _small_simulator(T=6.0, **RELAXED_CLIMATE).forward_samples_spatiotemporal(
        n=2, random_seed=4
    )
    torch.testing.assert_close(first["data"], again["data"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(first["data"], single["data"], rtol=1e-5, atol=1e-5)
    assert (first["data"] - other["data"]).abs().max() > 0
    assert (first["data"] - plain["data"]).abs().max() > 0


# ---------------------------------------------------------------------------
# The Well's periodic forcing
# ---------------------------------------------------------------------------


def _the_well_forcing(longitude, latitude, time):
    """season_day_forcing of gen_SWE_from_ic_file.py (The Well, commit aac88c0)."""
    time_of_day = time / (24 * HOUR)
    time_of_year = time / (1008 * HOUR)
    lon_center = time_of_day * 2 * math.pi
    lat_center = math.sin(time_of_year * 2 * math.pi) * 0.4
    sigma = 2 * math.pi / 3
    coefficients = torch.cos(longitude - lon_center) * torch.exp(
        -((latitude - lat_center) ** 2) / sigma**2
    )
    return (2.0 / HOUR) * coefficients


@pytest.mark.parametrize("time", [0.0, 5 * HOUR, 300 * HOUR, 777 * HOUR])
def test_periodic_forcing_is_the_forcing_of_the_well(time):
    dynamics = _dynamics()
    forcing = periodic_height_forcing(dynamics, THE_WELL, time)
    expected = _the_well_forcing(dynamics.longitude, dynamics.latitude, time)
    # absolute tolerance: the two evaluations differ by round-off where the cosine is 0
    atol = 1e-12 * THE_WELL.amplitude / HOUR
    torch.testing.assert_close(forcing, expected, rtol=1e-12, atol=atol)


def test_periodic_forcing_moves_eastward_once_a_day_and_adds_no_mass():
    dynamics = _dynamics(43, 64, 128)
    longitude = dynamics.longitude
    for hours, centre in [(0.0, 0.0), (6.0, math.pi / 2), (12.0, math.pi)]:
        forcing = periodic_height_forcing(dynamics, THE_WELL, hours * HOUR)
        index = int(forcing.argmax())
        assert longitude.flatten()[index].item() == pytest.approx(centre, abs=0.06)
    increment = periodic_forcing_increment(dynamics, THE_WELL, 1234.0, 2)
    assert increment[:, 0, 0, 0].abs().max() < 1e-12 * increment[:, 0].abs().max()
    assert increment[:, 1:].abs().max() == 0.0  # the wind is not forced


def test_periodic_forcing_through_the_simulator():
    """The height gains the forcing between frames, and the energy accounts for it."""
    payload = _small_simulator(
        T=9.0,
        skip_nt=1,
        periodic=PeriodicForcingSettings(),
        return_additional_input_fields=True,
        return_energy_budget=True,
    ).forward_samples_spatiotemporal(n=2, random_seed=0)
    added = payload["additional_input_fields"]
    assert added[:, :-1, ..., 0].abs().max() > 1.0  # metres per 3 h
    assert payload["energy_budget"][:, :-1, PERIODIC_COLUMN].abs().min() > 0


def test_periodic_forcing_is_returned_at_the_frame_times():
    """periodic_forcing holds F [m/h] at the time of each saved frame, as applied."""
    sim = _small_simulator(
        T=9.0,
        skip_nt=1,
        periodic=PeriodicForcingSettings(),
        return_periodic_forcing=True,
    )
    payload = sim.forward_samples_spatiotemporal(n=2, random_seed=0)
    forcing = payload["periodic_forcing"]
    assert forcing.shape == (2, 3, 32, 64, 1)
    for frame, hours in enumerate((3.0, 6.0, 9.0)):
        field = periodic_height_forcing(sim.dynamics, THE_WELL, hours * HOUR)
        projected = sim.dynamics.to_grid(
            torch.tril(sim.dynamics.to_coefficients(field[None]))
        )[0]
        torch.testing.assert_close(
            forcing[0, frame, ..., 0],
            (HOUR * projected).float().cpu(),
            rtol=1e-5,
            atol=1e-5,
        )
    torch.testing.assert_close(forcing[0], forcing[1])
    with pytest.raises(ValueError, match="return_periodic_forcing"):
        _small_simulator(return_periodic_forcing=True)


def test_invalid_periodic_and_sppt_settings_raise():
    with pytest.raises(ValueError, match="amplitude"):
        PeriodicForcingSettings(amplitude=-1.0)
    with pytest.raises(ValueError, match="same length"):
        SpptSettings([0.5, 0.2], [500.0e3], [6.0])
    with pytest.raises(ValueError, match="positive"):
        SpptSettings([0.5], [0.0], [6.0])
