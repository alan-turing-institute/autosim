"""Tests of the forcing, drag, relaxation and energy budget on the sphere."""

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
    GlobalBackscatterSettings,
    LocalBackscatterSettings,
    SkebsForcing,
    SkebsSettings,
    SphereShallowWaterDynamics,
    SphericalShallowWater,
    adams_bashforth_step,
    advance_pattern,
    autoregressive_parameter,
    complex_noise,
    dissipation_taper,
    global_dissipation_rate,
    global_integral,
    integrate_trajectories,
    kinetic_energy_spectrum,
    kinetic_forcing_increment,
    local_dissipation_rate,
    stationary_pattern,
    step_damping_factors,
    total_energy,
    unit_rate_amplitudes,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RADIUS = 6.37122e6
OMEGA = 7.292e-5
GRAVITY = 9.80616
MEAN_DEPTH = 1.0e4
LMAX = 22
DT = 300.0
# Columns of the energy budget: the rate of the kinetic forcing, and the energy
# changes of the processes before it
RATE = ENERGY_BUDGET_NAMES.index("effective_forcing_energy_rate")
CHANGES = slice(1, RATE)
# The published Galewsky jet: equal bounds fix every parameter
FIXED_GALEWSKY = {
    "jet_speed": (80.0, 80.0),
    "height_perturbation_m": (120.0, 120.0),
    "perturbation_lon": (0.0, 0.0),
    "perturbation_lat": (math.pi / 4, math.pi / 4),
}
# A jet of zero speed without bump: with noise_amplitude=0, a fluid at rest
AT_REST = {
    "jet_speed": (0.0, 0.0),
    "height_perturbation_m": (0.0, 0.0),
    "perturbation_lon": (0.0, 0.0),
    "perturbation_lat": (math.pi / 4, math.pi / 4),
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


def _rest_state(batch, lmax=LMAX):
    state = torch.zeros(batch, 3, lmax, lmax, dtype=torch.complex128, device=DEVICE)
    state[:, 0, 0, 0] = math.sqrt(4 * math.pi) * GRAVITY * MEAN_DEPTH
    return state


def _skebs(dynamics, batch, **changes):
    """SKEBS forcing of a batch: 1e-5 m^2/s^3, white, all degrees, no backscatter."""
    zeros = torch.zeros(batch, dtype=torch.float64)
    values = {
        "energy_rate": zeros + 1e-5,
        "correlation_time": zeros,
        "spectral_slope": zeros - 1.27,
        "degree_min": 1,
        "degree_max": LMAX - 1,
    }
    values.update(changes)
    return SkebsForcing(dynamics, _generators(batch), **values)


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
    }
    return SphericalShallowWater(**(options | overrides))


def _random_walk(dynamics, batch, n_steps, tau_steps, slope):
    """Sum of n_steps forcing increments at unit rate, and their mean spectrum."""
    correlation_time = torch.full((batch,), tau_steps * dynamics.dt)
    alpha = autoregressive_parameter(correlation_time, dynamics.dt).to(DEVICE)
    amplitudes = unit_rate_amplitudes(
        dynamics, alpha, torch.full((batch,), slope), 1, dynamics.lmax - 1
    )
    amplitudes = amplitudes.unsqueeze(1)
    n_patterns = amplitudes.shape[1]
    generators = _generators(batch)
    lmax = dynamics.lmax
    noise = complex_noise(generators, n_patterns, lmax)
    pattern = stationary_pattern(noise, amplitudes, alpha)
    rate = torch.ones(batch, dtype=torch.float64, device=DEVICE)
    walk = torch.zeros(batch, 3, lmax, lmax, dtype=torch.complex128, device=DEVICE)
    spectrum = torch.zeros(2, lmax, dtype=torch.float64, device=DEVICE)
    for _ in range(n_steps):
        increment = kinetic_forcing_increment(dynamics, pattern, rate)
        walk += increment
        spectrum += kinetic_energy_spectrum(increment, RADIUS).mean(dim=0)
        noise = complex_noise(generators, n_patterns, lmax)
        pattern = advance_pattern(pattern, noise, amplitudes, alpha)
    return walk, spectrum / n_steps, alpha[0].item()


# ---------------------------------------------------------------------------
# Pattern: amplitudes, noise, AR(1) process
# ---------------------------------------------------------------------------


def test_autoregressive_parameter():
    times = torch.tensor([0.0, DT, 8 * DT])
    alpha = autoregressive_parameter(times, DT)
    torch.testing.assert_close(alpha, torch.tensor([1.0, 1.0, 0.125]).double())
    with pytest.raises(ValueError, match="at least one solver step"):
        autoregressive_parameter(torch.tensor([0.5 * DT]), DT)


def test_amplitudes_follow_the_power_law_on_the_forced_degrees():
    dynamics = _dynamics()
    alpha = torch.tensor([1.0, 0.125])
    slope = torch.tensor([-1.27, 0.5])
    amplitudes = unit_rate_amplitudes(dynamics, alpha, slope, 3, 12)
    assert amplitudes.shape == (2, LMAX)
    assert amplitudes[:, :3].abs().max() == 0.0
    assert amplitudes[:, 13:].abs().max() == 0.0
    ratio = amplitudes[:, 12] / amplitudes[:, 3]
    torch.testing.assert_close(ratio, (4.0 ** slope.double()).to(DEVICE))

    # the amplitude gives a unit energy rate: b^2 Gamma / (8 pi a^2 alpha) = dt
    degree = torch.arange(LMAX, dtype=torch.float64, device=DEVICE)
    energy = (degree * (degree + 1) * (2 * degree + 1) * amplitudes**2).sum(dim=1)
    energy = energy / (8 * math.pi * RADIUS**2 * alpha.double().to(DEVICE))
    torch.testing.assert_close(energy, torch.full_like(energy, DT))


def test_noise_has_unit_variance_and_real_zonal_coefficients():
    noise = complex_noise(_generators(64), 2, LMAX)
    assert noise.shape == (64, 2, LMAX, LMAX)
    assert noise[:, :, :, 0].imag.abs().max() == 0.0
    assert torch.triu(noise, diagonal=1).abs().max() == 0.0
    zonal = noise[:, :, 1:, 0].real
    assert zonal.var().item() == pytest.approx(1.0, rel=0.1)
    others = noise[:, :, 4, 1:5]
    assert (others.abs() ** 2).mean().item() == pytest.approx(1.0, rel=0.1)
    assert others.real.var().item() == pytest.approx(0.5, rel=0.15)


def test_noise_of_a_simulation_does_not_depend_on_the_batch():
    together = complex_noise(_generators(4), 1, LMAX)
    alone = complex_noise(_generators(1, first_seed=2), 1, LMAX)
    torch.testing.assert_close(together[2], alone[0], rtol=0.0, atol=0.0)
    assert (together[0] - together[1]).abs().max() > 0.1


def test_pattern_statistics_match_the_ar1_definition():
    """Stationary variance g^2 / (2 - alpha) and lag-one autocorrelation 1 - alpha."""
    dynamics = _dynamics()
    batch = 32
    alpha = torch.full((batch,), 0.2, dtype=torch.float64, device=DEVICE)
    amplitudes = unit_rate_amplitudes(
        dynamics, alpha, torch.zeros(batch), 1, LMAX - 1
    ).unsqueeze(1)
    generators = _generators(batch)
    pattern = stationary_pattern(complex_noise(generators, 1, LMAX), amplitudes, alpha)

    lag_product = 0.0
    variance = 0.0
    for _ in range(300):
        noise = complex_noise(generators, 1, LMAX)
        new_pattern = advance_pattern(pattern, noise, amplitudes, alpha)
        # one degree, all its orders m > 0: the same amplitude g_l
        lag_product += (
            (new_pattern[:, 0, 10, 1:11] * pattern[:, 0, 10, 1:11].conj())
            .real.sum()
            .item()
        )
        variance += (pattern[:, 0, 10, 1:11].abs() ** 2).sum().item()
        pattern = new_pattern
    samples = 300 * batch * 10
    expected_variance = amplitudes[0, 0, 10].item() ** 2 / (2 - 0.2)
    assert variance / samples == pytest.approx(expected_variance, rel=0.05)
    assert lag_product / variance == pytest.approx(1 - 0.2, abs=0.01)


@pytest.mark.parametrize("tau_steps", [0, 8])
def test_injected_energy_rate_matches_the_target(tau_steps):
    """Kinetic energy gained by a free random walk of the increments, per unit time.

    The target is 1 m^2/s^3. A walk of n steps that starts from the stationary
    pattern gains, on average, the fraction
    alpha / (2 - alpha) (1 + 2 rho / (1 - rho) (1 - (1 - rho^n) / (n (1 - rho))))
    of it, with rho = 1 - alpha: 1 for white noise, and 1 for long walks. The
    tolerance is 4 standard errors of the mean over the batch plus 1 per cent.
    """
    dynamics = _dynamics()
    batch = 64
    n_steps = 400
    walk, _, alpha = _random_walk(dynamics, batch, n_steps, tau_steps, -1.27)
    energy = kinetic_energy_spectrum(walk, RADIUS).sum(dim=(1, 2))
    measured = energy.mean().item() / (n_steps * DT)
    standard_error = energy.std().item() / (n_steps * DT) / math.sqrt(batch)

    rho = 1 - alpha
    expected = 1.0
    if alpha < 1:
        finite_walk = 1 - (1 - rho**n_steps) / (n_steps * (1 - rho))
        expected = alpha / (2 - alpha) * (1 + 2 * rho / (1 - rho) * finite_walk)
    assert abs(measured - expected) < 4 * standard_error + 0.01


@pytest.mark.parametrize("slope", [-1.27, 0.5])
def test_forcing_spectrum_follows_the_power_law(slope):
    """Berner et al. (2009), eq. (10): the kinetic-energy spectrum goes as l^(2p+3).

    The exact spectrum is l (l + 1) (2 l + 1) l^(2p), which tends to l^(2p+3) at
    large degree. It is compared at every forced degree, and its slope over the
    upper half of the band is within 0.1 of 2p + 3.
    """
    dynamics = _dynamics(lmax=43, nlat=64, nlon=128)
    _, spectrum, _ = _random_walk(dynamics, 64, 100, 0, slope)
    degree = torch.arange(43, dtype=torch.float64, device=DEVICE)
    exact = degree * (degree + 1) * (2 * degree + 1) * degree ** (2 * slope)
    exact[0] = 0.0
    exact = exact / exact.sum() * spectrum.sum()
    torch.testing.assert_close(spectrum[0, 1:], exact[1:], rtol=0.15, atol=0.0)
    assert spectrum[1].abs().max() == 0.0  # no divergent part

    high = slice(21, 43)
    design = torch.stack([torch.log(degree[high]), torch.ones(22, device=DEVICE)], 1)
    target = torch.log(spectrum[0, high]).unsqueeze(1)
    fitted_slope = torch.linalg.lstsq(design, target).solution[0].item()
    assert fitted_slope == pytest.approx(2 * slope + 3, abs=0.1)


def test_vortical_increment_changes_the_vorticity_only():
    dynamics = _dynamics()
    batch = 4
    alpha = torch.ones(batch, dtype=torch.float64, device=DEVICE)
    amplitudes = unit_rate_amplitudes(
        dynamics, alpha, torch.zeros(batch), 1, LMAX - 1
    ).unsqueeze(1)
    pattern = stationary_pattern(
        complex_noise(_generators(batch), 1, LMAX), amplitudes, alpha
    )
    rate = torch.full((batch,), 4.0, dtype=torch.float64, device=DEVICE)
    increment = kinetic_forcing_increment(dynamics, pattern, rate)
    assert increment[:, 0].abs().max() == 0.0
    assert increment[:, 2].abs().max() == 0.0
    # the rate multiplies the pattern by its square root
    unit = kinetic_forcing_increment(dynamics, pattern, torch.ones_like(rate))
    torch.testing.assert_close(increment, 2.0 * unit)
    # a uniform rate on the grid gives the same increment as the rate per simulation
    rate_field = torch.full((batch, 32, 64), 4.0, dtype=torch.float64, device=DEVICE)
    local = kinetic_forcing_increment(dynamics, pattern, rate_field)
    assert (local - increment).abs().max() < 1e-12 * increment.abs().max()


# ---------------------------------------------------------------------------
# Dissipation diagnosed for the backscatter
# ---------------------------------------------------------------------------


def test_dissipation_taper():
    taper = dissipation_taper(43, DEVICE)
    assert taper.shape == (43, 1)
    assert (taper[:11] == 1.0).all()
    assert taper[30:].abs().max() < 1e-15
    assert (taper[10:31, 0].diff() <= 0).all()
    assert taper[20, 0].item() == pytest.approx(0.5)


def test_global_dissipation_rate_is_the_loss_of_kinetic_energy():
    dynamics = _dynamics()
    generator = torch.Generator().manual_seed(0)
    state = torch.randn(3, 3, LMAX, LMAX, dtype=torch.complex128, generator=generator)
    state[:, :, :, 0] = state[:, :, :, 0].real
    state = torch.tril(state).to(DEVICE) * 1e-6
    damping = torch.linspace(1.0, 0.5, LMAX, dtype=torch.float64, device=DEVICE)
    damped = state.clone()
    damped[:, 1:] = damping.reshape(LMAX, 1) * state[:, 1:]
    before = kinetic_energy_spectrum(state, RADIUS).sum(dim=(1, 2))
    after = kinetic_energy_spectrum(damped, RADIUS).sum(dim=(1, 2))
    rate = global_dissipation_rate(dynamics, state, damping)
    torch.testing.assert_close(rate, (before - after) / DT)


def test_local_dissipation_rate_has_the_global_rate_as_its_mean():
    """For one degree the local rate is positive, so the clamp does not act."""
    dynamics = _dynamics()
    state = torch.zeros(2, 3, LMAX, LMAX, dtype=torch.complex128, device=DEVICE)
    state[0, 1, 5, 2] = torch.tensor(1e-5 + 2e-5j)
    state[1, 1, 5, 0] = 3e-5
    damping = torch.full((LMAX,), 0.9, dtype=torch.float64, device=DEVICE)
    no_smoothing = torch.ones(LMAX, 1, dtype=torch.float64, device=DEVICE)
    local = local_dissipation_rate(
        dynamics, state, damping.reshape(LMAX, 1), no_smoothing
    )
    assert local.shape == (2, 32, 64)
    assert (local >= 0).all()
    mean = global_integral(local, dynamics.quad_weights) / (4 * math.pi)
    torch.testing.assert_close(mean, global_dissipation_rate(dynamics, state, damping))

    # smoothing keeps the mean and removes the small scales
    smooth = local_dissipation_rate(
        dynamics, state, damping.reshape(LMAX, 1), dissipation_taper(LMAX, DEVICE)
    )
    smooth_mean = global_integral(smooth, dynamics.quad_weights) / (4 * math.pi)
    torch.testing.assert_close(smooth_mean, mean, rtol=1e-6, atol=0.0)


# ---------------------------------------------------------------------------
# Drag and relaxation
# ---------------------------------------------------------------------------


def test_step_damping_factors():
    dynamics = _dynamics()
    drag_time = torch.tensor([torch.inf, 10 * DT])
    relaxation_time = torch.tensor([5 * DT, torch.inf])
    factors = step_damping_factors(dynamics, drag_time, relaxation_time)
    assert factors.shape == (2, 3, LMAX, LMAX)
    hyperdiffusion = dynamics.step_damping[1]
    torch.testing.assert_close(factors[0, 1], hyperdiffusion)
    torch.testing.assert_close(factors[1, 2], math.exp(-0.1) * hyperdiffusion)
    torch.testing.assert_close(
        factors[0, 0], torch.full_like(hyperdiffusion, math.exp(-0.2))
    )
    torch.testing.assert_close(factors[1, 0], torch.ones_like(hyperdiffusion))


def test_relaxation_moves_the_height_towards_its_equilibrium():
    """A fluid at rest without rotation: one step relaxes the height exactly."""
    grid = th.as_grid("legendre-gauss", nlat=32, nlon=64)
    dynamics = SphereShallowWaterDynamics(
        grid, LMAX, RADIUS, 0.0, GRAVITY, MEAN_DEPTH, DT, DEVICE
    )
    state = _rest_state(1)
    equilibrium = state.clone()
    equilibrium[0, 0, 2, 0] = 50.0 * GRAVITY  # a zonal height anomaly
    relaxation_time = torch.tensor([4 * DT])
    drag_time = torch.tensor([torch.inf])
    factors = step_damping_factors(dynamics, drag_time, relaxation_time)
    new_state, _ = adams_bashforth_step(
        dynamics, state, [], None, None, factors, equilibrium
    )
    expected = (1 - math.exp(-0.25)) * 50.0 * GRAVITY
    assert new_state[0, 0, 2, 0].real.item() == pytest.approx(expected, rel=1e-12)
    torch.testing.assert_close(new_state[0, 0, 0, 0], state[0, 0, 0, 0])


def test_drag_damps_the_kinetic_energy():
    dynamics = _dynamics()
    sim = _small_simulator(parameters_range=FIXED_GALEWSKY)
    rows = sim.sample_inputs(2, random_seed=0)
    state, _, _ = sim.initial_problem(rows, _generators(2))
    free = integrate_trajectories(sim.dynamics, state, 1, 36, 0)["final_state"]
    drag_time = torch.full((2,), 6 * 3600.0)
    damped = integrate_trajectories(sim.dynamics, state, 1, 36, 0, drag_time=drag_time)[
        "final_state"
    ]
    energy_free = kinetic_energy_spectrum(free, RADIUS).sum(dim=(1, 2))
    energy_damped = kinetic_energy_spectrum(damped, RADIUS).sum(dim=(1, 2))
    # 36 steps of dynamics.dt: the drag alone would give exp(-2 t / tau)
    elapsed = 36 * sim.dynamics.dt
    expected = math.exp(-2 * elapsed / (6 * 3600.0))
    ratio = energy_damped / energy_free
    assert (ratio < 1).all()
    torch.testing.assert_close(
        ratio, torch.full_like(ratio, expected), rtol=0.2, atol=0.0
    )
    assert dynamics.lmax == LMAX


def test_relaxation_and_drag_through_the_simulator_conserve_the_mass():
    # A jet of 60 m/s without bump and noise: the mean height is zero, as for the
    # equilibrium (a jet of 80 m/s), and the relaxation acts on the difference
    sim = _small_simulator(
        damping=DampingSettings(relaxation_time=48.0, drag_time=96.0),
        return_energy_budget=True,
        T=12.0,
        parameters_range={**AT_REST, "jet_speed": (60.0, 60.0)},
        noise_amplitude=0.0,
    )
    payload = sim.forward_samples_spatiotemporal(n=2, random_seed=0)
    assert torch.isfinite(payload["data"]).all()
    budget = payload["energy_budget"]
    assert (budget[:, :-1, 3] < 0).all()  # the drag removes energy
    assert (budget[:, :-1, 4] != 0).all()  # the relaxation changes it
    # the equilibrium has the mean depth, so the mean height is unchanged
    height = payload["data"][..., 0].double().to(DEVICE)
    mean = global_integral(height, sim.dynamics.quad_weights) / (4 * math.pi)
    assert mean.abs().max() < 1e-2


def test_jet_target_over_orography_is_the_depth_under_the_jet():
    """Over a mountain, the equilibrium is the free surface of the jet minus g h_s."""
    sim = _small_simulator(
        initial_condition="mountain", damping=DampingSettings(relaxation_time=24.0)
    )
    rows = sim.sample_inputs(2, random_seed=0)
    _, coriolis, surface = sim.initial_problem(rows, [])
    assert surface is not None
    assert surface.abs().max() > 0
    _, _, free_surface = sim.damping_settings(rows, coriolis, None)
    _, _, equilibrium = sim.damping_settings(rows, coriolis, surface)
    assert free_surface is not None
    assert equilibrium is not None
    torch.testing.assert_close(equilibrium, free_surface - surface, rtol=0.0, atol=0.0)


# ---------------------------------------------------------------------------
# Forced runs: reproducibility, outputs, energy budget
# ---------------------------------------------------------------------------


def test_zero_rate_forcing_is_the_unforced_run():
    unforced = _small_simulator().forward_samples_spatiotemporal(n=2, random_seed=1)
    forced = _small_simulator(
        skebs=SkebsSettings(energy_rate=0.0)
    ).forward_samples_spatiotemporal(n=2, random_seed=1)
    torch.testing.assert_close(forced["data"], unforced["data"], rtol=0.0, atol=0.0)


def test_seed_reproduces_the_forcing_and_another_seed_changes_it():
    sim = _small_simulator(
        skebs=SkebsSettings(energy_rate=1e-4, correlation_time=1.0),
        parameters_range=FIXED_GALEWSKY,
        return_additional_input_fields=True,
    )
    first = sim.forward_samples_spatiotemporal(n=2, random_seed=3)
    second = sim.forward_samples_spatiotemporal(n=2, random_seed=3)
    other = sim.forward_samples_spatiotemporal(n=2, random_seed=4)
    torch.testing.assert_close(first["data"], second["data"], rtol=0.0, atol=0.0)
    forcing = first["additional_input_fields"]
    torch.testing.assert_close(forcing, second["additional_input_fields"])
    # independent forcing: the two simulations of a call, and the two seeds
    scale = forcing[0, 0, ..., 1].std()
    assert (forcing[0, 0, ..., 1] - forcing[1, 0, ..., 1]).std() > scale
    other_forcing = other["additional_input_fields"]
    assert (forcing[0, 0, ..., 1] - other_forcing[0, 0, ..., 1]).std() > scale
    correlation = torch.corrcoef(
        torch.stack(
            [forcing[0, 0, ..., 1].flatten(), other_forcing[0, 0, ..., 1].flatten()]
        )
    )[0, 1]
    assert correlation.abs() < 0.3


def test_forcing_of_a_simulation_does_not_depend_on_the_batch_size():
    options = {
        "skebs": SkebsSettings(energy_rate=1e-4, correlation_time=1.0),
        "return_additional_input_fields": True,
        "T": 3.0,
    }
    one_by_one = _small_simulator(batch_size=1, **options)
    together = _small_simulator(batch_size=3, **options)
    first = one_by_one.forward_samples_spatiotemporal(n=3, random_seed=5)
    second = together.forward_samples_spatiotemporal(n=3, random_seed=5)
    torch.testing.assert_close(
        first["additional_input_fields"],
        second["additional_input_fields"],
        rtol=1e-4,
        atol=1e-6,
    )


def test_forcing_is_returned_as_transition_aligned_additional_input():
    """With one solver step per frame and a fluid at rest, frame i + 1 - frame i."""
    sim = _small_simulator(
        parameters_range=AT_REST,
        noise_amplitude=0.0,
        skebs=SkebsSettings(energy_rate=1e-4),
        dt_save=0.05,
        T=0.15,
        return_additional_input_fields=True,
    )
    assert sim.steps_per_save == 1
    payload = sim.forward_samples_spatiotemporal(n=2, random_seed=0)
    data = payload["data"]
    forcing = payload["additional_input_fields"]
    assert forcing.shape == data.shape == (2, 4, 32, 64, 3)
    assert sim.additional_input_names == [
        "forcing_h",
        "forcing_u_phi",
        "forcing_u_theta",
    ]
    assert forcing[:, -1].abs().max() == 0.0
    assert forcing[:, 0, ..., 1:].abs().max() > 0
    # first transition from rest: the wind of frame 1 is the first increment
    torch.testing.assert_close(
        data[:, 1, ..., 1:] - data[:, 0, ..., 1:],
        forcing[:, 0, ..., 1:],
        rtol=1e-4,
        atol=1e-7,
    )


def test_unforced_additional_input_fields_are_zero():
    sim = _small_simulator(return_additional_input_fields=True)
    payload = sim.forward_samples_spatiotemporal(n=1, random_seed=0)
    assert payload["additional_input_fields"].abs().max() == 0.0


def test_forcing_state_is_returned_and_has_memory():
    sim = _small_simulator(
        skebs=SkebsSettings(energy_rate=1e-4, correlation_time=12.0),
        return_forcing_state=True,
    )
    payload = sim.forward_samples_spatiotemporal(n=2, random_seed=0)
    pattern = payload["forcing_state"]
    assert pattern.shape == (2, 3, 32, 64, 2)
    assert sim.forcing_state_names == ["forcing_state_u_phi", "forcing_state_u_theta"]
    assert torch.isfinite(pattern).all()
    # decorrelation time of 12 h: frames 3 h apart are correlated
    correlation = torch.corrcoef(
        torch.stack([pattern[0, 0].flatten(), pattern[0, 1].flatten()])
    )[0, 1]
    assert correlation > 0.5


def test_energy_budget_closes_in_double_precision():
    """Every term is an energy difference across one operation of the step."""
    dynamics = _dynamics()
    batch = 3
    skebs = _skebs(
        dynamics,
        batch,
        correlation_time=torch.full((batch,), 3600.0, dtype=torch.float64),
        global_fraction=torch.full((batch,), 0.5, dtype=torch.float64),
    )
    equilibrium = _rest_state(batch)[:, 0]
    equilibrium[:, 2, 0] = 30.0 * GRAVITY
    output = integrate_trajectories(
        dynamics,
        _rest_state(batch),
        n_saves=4,
        steps_per_save=12,
        skip_nt=1,
        drag_time=torch.full((batch,), 5 * 86400.0),
        relaxation_time=torch.full((batch,), 2 * 86400.0),
        equilibrium_geopotential=equilibrium,
        skebs=skebs,
        return_energy_budget=True,
    )
    budget = output["energy_budget"]
    assert budget.shape == (batch, 4, len(ENERGY_BUDGET_NAMES))
    assert budget.dtype == torch.float64
    change = budget[:, 1:, 0] - budget[:, :-1, 0]
    terms = budget[:, :-1, CHANGES].sum(dim=-1)
    assert (change - terms).abs().max() < 1e-12 * budget[:, :, 0].abs().max()
    assert budget[:, -1, 1:].abs().max() == 0.0
    assert (budget[:, :-1, 2] <= 0).all()  # hyperdiffusion removes energy
    assert (budget[:, :-1, 3] < 0).all()  # so does the drag
    assert (budget[:, :-1, RATE] > 1e-5).all()  # base rate plus backscatter
    # the first column is the energy of the saved states
    final_energy = total_energy(dynamics, output["final_state"])
    torch.testing.assert_close(budget[:, -1, 0], final_energy.cpu())


def test_energy_budget_through_the_simulator():
    options = {
        "skebs": SkebsSettings(energy_rate=1e-4),
        "T": 9.0,
        "parameters_range": FIXED_GALEWSKY,
    }
    sim = _small_simulator(return_energy_budget=True, **options)
    payload = sim.forward_samples_spatiotemporal(n=2, random_seed=0)
    budget = payload["energy_budget"]
    assert budget.shape == (2, 4, len(ENERGY_BUDGET_NAMES))
    assert sim.energy_budget_names == list(ENERGY_BUDGET_NAMES)
    change = budget[:, 1:, 0] - budget[:, :-1, 0]
    terms = budget[:, :-1, CHANGES].sum(dim=-1)
    # the saved budget is float32: its closure is that of the rounding of the energy
    torch.testing.assert_close(
        change, terms, rtol=1e-3, atol=1e-3 * budget[..., 0].max().item()
    )
    # the work of the forcing on the existing flow has either sign in one interval,
    # and it is positive on average
    assert budget[:, :-1, 5].mean() > 0
    torch.testing.assert_close(
        budget[:, :-1, RATE], torch.full_like(budget[:, :-1, RATE], 1e-4)
    )
    # without drag and relaxation their terms are zero
    assert budget[..., 3].abs().max() == 0.0
    assert budget[..., 4].abs().max() == 0.0

    # the budget does not change the trajectory
    plain = _small_simulator(**options).forward_samples_spatiotemporal(
        n=2, random_seed=0
    )
    torch.testing.assert_close(plain["data"], payload["data"], rtol=0.0, atol=0.0)
    assert "energy_budget" not in plain


def test_unforced_energy_budget_has_no_forcing_term():
    sim = _small_simulator(return_energy_budget=True)
    budget = sim.forward_samples_spatiotemporal(n=1, random_seed=0)["energy_budget"]
    assert budget[..., 5].abs().max() == 0.0
    assert budget[..., RATE].abs().max() == 0.0
    assert (budget[:, :-1, 2] <= 0).all()


BACKSCATTER_OPTIONS = {
    "skebs": SkebsSettings(energy_rate=0.0),
    "return_energy_budget": True,
    "return_additional_input_fields": True,
    "T": 3.0,
    "parameters_range": FIXED_GALEWSKY,
}


BACKSCATTER_HALF = {
    "global": {"global_backscatter": GlobalBackscatterSettings(fraction=0.5)},
    "local": {"local_backscatter": LocalBackscatterSettings(fraction=0.5)},
}


def test_global_backscatter_drives_the_forcing_from_the_dissipation():
    """With a zero base rate the forcing is the backscatter of the dissipation."""
    half = _small_simulator(
        global_backscatter=GlobalBackscatterSettings(fraction=0.5),
        **BACKSCATTER_OPTIONS,
    )
    full = _small_simulator(
        global_backscatter=GlobalBackscatterSettings(fraction=1.0),
        **BACKSCATTER_OPTIONS,
    )
    budget_half = half.forward_samples_spatiotemporal(n=2, random_seed=0)[
        "energy_budget"
    ]
    payload_full = full.forward_samples_spatiotemporal(n=2, random_seed=0)
    budget_full = payload_full["energy_budget"]
    assert (budget_half[:, 0, RATE] > 0).all()
    assert payload_full["additional_input_fields"][:, 0].abs().max() > 0
    # the rate is proportional to the fraction, up to the feedback of the forcing
    # on the dissipation over these 3 h
    ratio = budget_full[:, 0, RATE] / budget_half[:, 0, RATE]
    torch.testing.assert_close(ratio, torch.full_like(ratio, 2.0), rtol=0.15, atol=0.0)


def test_local_backscatter_rate_is_at_least_the_global_one():
    """The local rate is the smoothed dissipation with its negative values removed.

    Before this clamp the smoothed field has the global dissipation as its mean
    (``test_local_dissipation_rate_has_the_global_rate_as_its_mean``), so the mean
    local rate is at least the global rate. The excess depends on the resolution:
    it was measured as 40 to 60 per cent at lmax = 22 and 43, and 8 to 10 per cent
    at lmax = 86, with the filter of Berner et al. (degrees 10 to 30).
    """
    rates = {}
    for mode, setting in BACKSCATTER_HALF.items():
        sim = _small_simulator(**setting, **BACKSCATTER_OPTIONS)
        payload = sim.forward_samples_spatiotemporal(n=2, random_seed=0)
        assert torch.isfinite(payload["data"]).all()
        assert payload["additional_input_fields"][:, 0].abs().max() > 0
        rates[mode] = payload["energy_budget"][:, 0, RATE]
    assert (rates["local"] >= rates["global"]).all()
    assert (rates["local"] < 4 * rates["global"]).all()


def test_global_and_local_backscatter_add_up():
    """Both backscatters at once: the rate is the sum of the two rates.

    Up to the feedback of the forcing on the dissipation over the first 3 h, as in
    ``test_global_backscatter_drives_the_forcing_from_the_dissipation``.
    """
    rates = {}
    both = BACKSCATTER_HALF["global"] | BACKSCATTER_HALF["local"]
    for name, setting in [*BACKSCATTER_HALF.items(), ("both", both)]:
        sim = _small_simulator(**setting, **BACKSCATTER_OPTIONS)
        payload = sim.forward_samples_spatiotemporal(n=2, random_seed=0)
        assert torch.isfinite(payload["data"]).all()
        rates[name] = payload["energy_budget"][:, 0, RATE]
    torch.testing.assert_close(
        rates["both"], rates["global"] + rates["local"], rtol=0.15, atol=0.0
    )


def test_sampled_forcing_scalars_are_returned_in_constant_scalars():
    sim = _small_simulator(
        skebs=SkebsSettings(),
        damping=DampingSettings(),
        parameters_range={
            **FIXED_GALEWSKY,
            "skebs.energy_rate": (1e-5, 1e-4),
            "damping.drag_time": (48.0, 96.0),
        },
        return_energy_budget=True,
        T=3.0,
    )
    payload = sim.forward_samples_spatiotemporal(n=4, random_seed=0)
    scalars = payload["constant_scalars"]
    assert scalars.shape == (4, 6)
    rate = scalars[:, sim.get_parameter_idx("skebs.energy_rate")]
    assert rate.min() >= 1e-5
    assert rate.max() <= 1e-4
    assert rate.unique().numel() == 4
    # each simulation was forced at its own rate
    torch.testing.assert_close(
        payload["energy_budget"][:, 0, RATE], rate, rtol=1e-5, atol=0.0
    )


# ---------------------------------------------------------------------------
# Rejected settings
# ---------------------------------------------------------------------------


def test_invalid_forcing_settings_raise():
    with pytest.raises(ValueError, match="non-negative"):
        GlobalBackscatterSettings(fraction=-0.1)
    with pytest.raises(ValueError, match="require skebs"):
        _small_simulator(local_backscatter=LocalBackscatterSettings(fraction=0.5))
    with pytest.raises(ValueError, match="return_forcing_state requires"):
        _small_simulator(return_forcing_state=True)
    with pytest.raises(ValueError, match="must be >= 0"):
        SkebsSettings(energy_rate=-1.0)
    with pytest.raises(ValueError, match="skebs degrees"):
        _small_simulator(skebs=SkebsSettings(degree_max=LMAX))
    with pytest.raises(ValueError, match="relaxation_target must be one of"):
        DampingSettings(relaxation_target="unknown")
    with pytest.raises(ValueError, match="unknown keys"):
        _small_simulator(parameters_range={**FIXED_GALEWSKY, "viscosity": (0.0, 1.0)})
    with pytest.raises(ValueError, match="which requires damping"):
        _small_simulator(
            parameters_range={**FIXED_GALEWSKY, "damping.drag_time": (48.0, 96.0)}
        )


# ---------------------------------------------------------------------------
# W92 case 4: a known answer for the splitting of the forcing
# ---------------------------------------------------------------------------


def _case4_state(dynamics, background, time, flow_speed):
    """Analytic state of W92 case 4 at one time: a low translating with the flow.

    The stream function of the low (W92 eqs. 128, 129) is added to a zonal flow:
    the wind gains k x grad(psi), so the vorticity gains the Laplacian of psi, and
    the geopotential gains f psi (eqs. 123-125).

    Returns:
        The state and its part that moves with the low, both (1, 3, L, L).
    """
    reference_geopotential = 1.0e5
    f0 = 2 * OMEGA * math.sin(math.pi / 4)
    psi0 = -0.03 * reference_geopotential / f0
    sigma = 12.74244**2
    centre_lon = -7 * math.pi / 12
    centre_lat = math.pi / 4

    latitude = dynamics.latitude
    longitude = dynamics.longitude
    cosine = math.sin(centre_lat) * torch.sin(latitude) + math.cos(
        centre_lat
    ) * torch.cos(latitude) * torch.cos(
        longitude - flow_speed / RADIUS * time - centre_lon
    )
    psi = psi0 * torch.exp(-sigma * (1 - cosine) / (1 + cosine))

    moving = torch.zeros_like(background)
    moving[0, 0] = dynamics.to_coefficients((dynamics.coriolis * psi).unsqueeze(0))[0]
    moving[0, 1] = dynamics.laplacian * dynamics.to_coefficients(psi.unsqueeze(0))[0]
    moving = torch.tril(moving)
    return background + moving, moving


@pytest.mark.slow
def test_williamson_case_4_through_the_forcing_splitting():
    """W92 case 4: a forced system with a known solution (eqs. 116-130).

    The forcing is the one that makes the analytic state a solution,
    F = d/dt(analytic state) - N(analytic state), with N the tendencies of the
    solver. The low moves at the angular speed u_0 / a, so its time derivative is
    -(u_0 / a) d/dlambda, a multiplication by -i m u_0 / a. The increment dt F, taken
    at the middle of the step, is added after each Adams-Bashforth step: the
    splitting of ``integrate_trajectories``. The hyperdiffusion is switched off,
    since the analytic state solves the inviscid equations.

    After 5 days the perturbation height h' (W92 eq. 131) agrees with the analytic
    one at the level of the time discretisation: the measured l2 error is 1.7e-5
    with a step of 150 s and 6.8e-6 with 75 s. The factor 2.6 lies between first
    order (the splitting of the forcing from the dynamics) and second order.
    """
    from autosim.experimental.simulations.shallow_water_sphere import (  # noqa: PLC0415
        balanced_geopotential,
        height_error_norms,
        shallow_water_tendencies,
        spectral_to_fields,
    )

    lmax, nlat, nlon = 43, 64, 128
    flow_speed = 20.0
    mean_depth = 1.0e5 / GRAVITY
    grid = th.as_grid("legendre-gauss", nlat=nlat, nlon=nlon)
    order = torch.arange(lmax, dtype=torch.float64, device=DEVICE)  # m of each column

    errors = []
    for steps_per_hour in (24, 48):
        dt = 3600.0 / steps_per_hour
        dynamics = SphereShallowWaterDynamics(
            grid, lmax, RADIUS, OMEGA, GRAVITY, mean_depth, dt, DEVICE
        )
        dynamics.step_damping = torch.ones_like(dynamics.step_damping)

        # Zonal flow u = u_0 sin^14(2 lat) (eq. 126) and its balanced height (eq. 127),
        # with g h = 1e5 m^2/s^2 at the south pole
        eastward = flow_speed * torch.sin(2 * dynamics.latitude) ** 14
        wind = torch.stack([eastward, torch.zeros_like(eastward)]).unsqueeze(0)
        vorticity, divergence = dynamics.vorticity_divergence(wind)
        geopotential = balanced_geopotential(dynamics, vorticity, divergence)
        south = dynamics.to_grid(geopotential)[0, -1, 0]
        geopotential[0, 0, 0] += math.sqrt(4 * math.pi) * (1.0e5 - south)
        background = torch.tril(torch.stack([geopotential, vorticity, divergence], 1))

        state, _ = _case4_state(dynamics, background, 0.0, flow_speed)
        previous = []
        n_steps = 5 * 24 * steps_per_hour
        for step in range(n_steps):
            middle = (step + 0.5) * dt
            analytic, moving = _case4_state(dynamics, background, middle, flow_speed)
            time_derivative = -1j * order * (flow_speed / RADIUS) * moving
            forcing = time_derivative - shallow_water_tendencies(dynamics, analytic)
            state, previous = adams_bashforth_step(dynamics, state, previous)
            state = state + dt * forcing

        analytic, _ = _case4_state(dynamics, background, n_steps * dt, flow_speed)
        background_height = spectral_to_fields(dynamics, background)[..., 0]
        height = spectral_to_fields(dynamics, state)[..., 0] - background_height
        height_true = spectral_to_fields(dynamics, analytic)[..., 0] - background_height
        _, l2, _ = height_error_norms(height, height_true, dynamics.quad_weights)
        errors.append(l2.item())

    assert errors[0] < 1e-4
    assert errors[1] < errors[0] / 2
