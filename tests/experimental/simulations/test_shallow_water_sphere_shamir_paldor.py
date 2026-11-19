"""Tests of the Shamir and Paldor (2016) waves of the spherical shallow water solver."""

from __future__ import annotations

import math

import pytest
import torch

th = pytest.importorskip("torch_harmonics")
if not hasattr(th, "as_grid"):
    pytest.skip("torch-harmonics without grid descriptors", allow_module_level=True)

from autosim.experimental.simulations.shallow_water_sphere import (  # noqa: E402
    SphereShallowWaterDynamics,
    SphericalShallowWater,
    adams_bashforth_step,
    gegenbauer,
    meridional_structure,
    phase_speeds,
    shamir_paldor_initial_condition,
    spectral_to_fields,
    wave_phase_speed,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RADIUS = 6.37122e6
OMEGA = 7.292e-5
GRAVITY = 9.80616
# Depth of the published test of Shamir and Paldor (2016): gH = 5e4 m^2/s^2
GRAVITY_DEPTH = 5.0e4
DEPTH = GRAVITY_DEPTH / GRAVITY
HOUR = 3600.0
DAY = 86400.0
# Tolerances, about ten times the measured values given in the test docstrings
PEAK_TOLERANCE = 1e-4  # measured: 1.3e-5
STRUCTURE_CORRELATION = 0.99  # measured: 0.9986 after 6 h


def _dynamics(lmax=43, nlat=64, nlon=128):
    grid = th.as_grid("legendre-gauss", nlat=nlat, nlon=nlon)
    dt = 0.4 * (RADIUS / lmax) / (math.sqrt(GRAVITY_DEPTH) + 80.0)
    steps_per_hour = math.ceil(HOUR / dt)
    dynamics = SphereShallowWaterDynamics(
        grid, lmax, RADIUS, OMEGA, GRAVITY, DEPTH, HOUR / steps_per_hour, DEVICE
    )
    return dynamics, steps_per_hour


def _wave(dynamics, wave_type, k, n, amplitude):
    one = torch.ones(1, dtype=torch.float64)
    return shamir_paldor_initial_condition(
        dynamics, wave_type * one, k * one, n * one, amplitude * one
    )


def _advance(dynamics, state, steps):
    previous = []
    for _ in range(steps):
        state, previous = adams_bashforth_step(dynamics, state, previous)
    return state


def _zonal_coefficient(dynamics, state, k):
    """Fourier coefficient of order k of the height on every latitude ring, (nlat,)."""
    height = spectral_to_fields(dynamics, state)[0, ..., 0]
    return torch.fft.rfft(height, dim=-1)[:, k]


# ---------------------------------------------------------------------------
# Phase speeds and structure
# ---------------------------------------------------------------------------


def test_published_wave_has_the_published_periods():
    """(n, k) = (5, 10) at gH = 5e4: Rossby period 12.0 days, eastward 3.2 hours.

    Brachet and Croisille (2021) quote 12.03 days and 3.16 hours for this wave.
    """
    rossby, westward, eastward = phase_speeds(5, 10, GRAVITY_DEPTH, OMEGA, RADIUS)
    assert westward < rossby < 0
    assert eastward > 0
    rossby_period = 2 * math.pi / (10 * abs(rossby)) / DAY
    eastward_period = 2 * math.pi / (10 * eastward) / HOUR
    assert rossby_period == pytest.approx(12.0, abs=0.05)
    assert eastward_period == pytest.approx(3.2, abs=0.05)
    assert wave_phase_speed(0, 5, 10, GRAVITY_DEPTH, OMEGA, RADIUS) == rossby
    assert wave_phase_speed(1, 5, 10, GRAVITY_DEPTH, OMEGA, RADIUS) == eastward
    assert wave_phase_speed(2, 5, 10, GRAVITY_DEPTH, OMEGA, RADIUS) == westward


def test_gegenbauer_polynomial_matches_the_direct_formula():
    """SP16 eq. 19: C_5 = {4 (a)_5 x^5 - 20 (a)_4 x^3 + 15 (a)_3 x} / 15."""
    order = 10.6
    x = torch.linspace(-1.0, 1.0, 101, dtype=torch.float64)

    def pochhammer(value, count):
        return math.prod(value + index for index in range(count))

    expected = (
        4 * pochhammer(order, 5) * x**5
        - 20 * pochhammer(order, 4) * x**3
        + 15 * pochhammer(order, 3) * x
    ) / 15
    torch.testing.assert_close(gegenbauer(5, order, x), expected, rtol=1e-12, atol=0.0)


def test_slow_gravity_wave_is_rejected():
    """An inertia-gravity wave slower than 2 Omega has a singular structure."""
    latitude = torch.linspace(-1.5, 1.5, 61, dtype=torch.float64)
    with pytest.raises(ValueError, match="singular"):
        meridional_structure(1, 1, 1, latitude, GRAVITY, DEPTH, OMEGA, RADIUS)


@pytest.mark.parametrize("wave_type", [0, 1, 2])
def test_wave_has_the_requested_peak_wind_and_zonal_order(wave_type):
    dynamics, _ = _dynamics()
    state = _wave(dynamics, wave_type, 6, 3, 12.0)
    fields = spectral_to_fields(dynamics, state)[0]
    speed = torch.sqrt(fields[..., 1] ** 2 + fields[..., 2] ** 2)
    assert speed.max().item() == pytest.approx(12.0, rel=PEAK_TOLERANCE)
    # the zonal mean of the height anomaly is zero: the wave has zonal order k only
    coefficients = torch.fft.rfft(fields[..., 0], dim=-1)
    assert coefficients[:, 1:6].abs().max() < 1e-6 * coefficients[:, 6].abs().max()


# ---------------------------------------------------------------------------
# Propagation in the solver
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("wave_type", "duration_hours", "tolerance"),
    [
        (0, 24.0, 0.05),  # Rossby, measured 3.0 %
        (1, 6.0, 0.01),  # eastward inertia-gravity, measured 0.19 %
        (2, 6.0, 0.05),  # westward inertia-gravity, measured 2.4 %
    ],
)
def test_small_wave_travels_at_its_phase_speed(wave_type, duration_hours, tolerance):
    """The published wave (5, 10) at a small amplitude travels at C (SP16 eqs. 7-11).

    The phase speed is measured as in SP16 (section 5), from the phase of the zonal
    Fourier coefficient of order k at the latitude of maximum amplitude. The formula
    approximates the exact linear wave: SP16 find 1.8 % (Rossby) and 0.3 % (eastward)
    for this wave. Measured at lmax = 43: 3.0 % (Rossby, the same after 72 h), 0.19 %
    (eastward) and 2.4 % (westward).
    """
    dynamics, steps_per_hour = _dynamics()
    state = _wave(dynamics, wave_type, 10, 5, 0.01)
    start = _zonal_coefficient(dynamics, state, 10)
    ring = int(start.abs().argmax())
    steps = round(duration_hours * steps_per_hour)
    later = _zonal_coefficient(dynamics, _advance(dynamics, state, steps), 10)

    # e^{ik(lambda - Ct)}: the phase of the coefficient decreases by k C t
    phase = torch.angle(later[ring] / start[ring]).item()
    expected = wave_phase_speed(wave_type, 5, 10, GRAVITY_DEPTH, OMEGA, RADIUS)
    turns = round((-10 * expected * steps * dynamics.dt - phase) / (2 * math.pi))
    measured = -(phase + 2 * math.pi * turns) / (10 * steps * dynamics.dt)
    assert measured == pytest.approx(expected, rel=tolerance)


def test_small_wave_keeps_its_meridional_structure():
    """After a few periods, the latitude profile of the wave equals the initial one.

    Measured correlation: 0.9986 after 6 h.
    """
    dynamics, steps_per_hour = _dynamics()
    state = _wave(dynamics, 1, 10, 5, 0.01)
    start = _zonal_coefficient(dynamics, state, 10).abs()
    later = _zonal_coefficient(
        dynamics, _advance(dynamics, state, 12 * steps_per_hour), 10
    ).abs()
    correlation = (start * later).sum() / torch.sqrt(
        (start**2).sum() * (later**2).sum()
    )
    assert correlation.item() > STRUCTURE_CORRELATION


# ---------------------------------------------------------------------------
# Through the simulator
# ---------------------------------------------------------------------------


def test_integer_parameters_are_rounded_and_returned():
    sim = SphericalShallowWater(
        lmax=22,
        nlat=32,
        nlon=64,
        T=0.0,
        dt_save=1.0,
        skip_nt=0,
        batch_size=4,
        device=DEVICE,
        log_level="warning",
        initial_condition="shamir_paldor",
        mean_depth=DEPTH,
    )
    payload = sim.forward_samples_spatiotemporal(n=12, random_seed=0)
    for name, low, high in [
        ("wave_type", 0, 2),
        ("zonal_wavenumber", 3, 8),
        ("mode_number", 1, 5),
    ]:
        values = payload["constant_scalars"][:, sim.get_parameter_idx(name)]
        torch.testing.assert_close(values, values.round())
        assert values.min() >= low
        assert values.max() <= high
    assert torch.isfinite(payload["data"]).all()
