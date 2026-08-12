import inspect
from typing import Any

import pytest
import torch

from autosim.experimental.simulations import ShallowWater2D
from autosim.experimental.simulations._spectral import (
    gaussian_ring_spectrum,
    spectral_wavenumbers,
    two_thirds_mask,
)
from autosim.experimental.simulations.shallow_water import (
    _coriolis_grid,
    _sample_swe_forcing_field,
    simulate_swe_2d,
)

FORCING_GRID_SIZE = 24
FORCING_WAVENUMBER = 4.0 * 2.0 * torch.pi / FORCING_GRID_SIZE
FORCING_BANDWIDTH = 1.5 * 2.0 * torch.pi / FORCING_GRID_SIZE
FORCING_OPTIONS: dict[str, Any] = {
    "nx": FORCING_GRID_SIZE,
    "ny": FORCING_GRID_SIZE,
    "Lx": 24.0,
    "Ly": 24.0,
    "nu": 0.0,
    "drag": 0.0,
    "coriolis_mode": "f_plane",
    "forcing_wavenumber": FORCING_WAVENUMBER,
    "forcing_bandwidth": FORCING_BANDWIDTH,
}


def _small_simulator(**overrides: Any) -> ShallowWater2D:
    options: dict[str, Any] = {
        "return_timeseries": True,
        "log_level": "warning",
        "nx": 18,
        "ny": 18,
        "Lx": 18.0,
        "Ly": 18.0,
        "T": 0.1,
        "dt_save": 0.1,
        "parameters_range": {"amp": (0.1, 0.1)},
    }
    options.update(overrides)
    return ShallowWater2D(**options)


def _run_small_swe(**overrides: Any) -> torch.Tensor:
    options: dict[str, Any] = {
        "amp": 0.1,
        "return_timeseries": True,
        "nx": 18,
        "ny": 18,
        "Lx": 18.0,
        "Ly": 18.0,
        "T": 0.1,
        "dt_save": 0.1,
        "cfl": 0.12,
        "g": 9.81,
        "h_mean": 1.0,
        "nu": 5e-4,
        "drag": 2e-3,
    }
    options.update(overrides)
    return simulate_swe_2d(**options)


def test_full_swe_timeseries_shape_and_finite() -> None:
    sim = _small_simulator(
        nx=32,
        ny=32,
        Lx=32.0,
        Ly=32.0,
        T=10.0,
        dt_save=1.0,
    )

    out = sim.forward_samples_spatiotemporal(n=1, random_seed=0)
    data = out["data"]

    expected_frames = int(sim.T / sim.dt_save) + 1
    assert data.shape == (1, expected_frames, sim.nx, sim.ny, 3)
    assert torch.isfinite(data).all()

    h = data[0, ..., 0]
    assert (h[-1] - h[0]).abs().max().item() > 1e-4


def test_swe_preserves_original_positional_parameter_prefix() -> None:
    class_prefix = [
        "parameters_range",
        "output_names",
        "return_timeseries",
        "log_level",
        "nx",
        "ny",
        "Lx",
        "Ly",
        "T",
        "dt_save",
        "skip_nt",
        "cfl",
        "g",
        "h_mean",
        "nu",
        "drag",
        "dtype",
    ]
    function_prefix = [
        "amp",
        "return_timeseries",
        "nx",
        "ny",
        "Lx",
        "Ly",
        "T",
        "dt_save",
        "cfl",
        "g",
        "h_mean",
        "nu",
        "drag",
        "dtype",
        "skip_nt",
    ]

    class_parameters = list(inspect.signature(ShallowWater2D).parameters)
    function_parameters = list(inspect.signature(simulate_swe_2d).parameters)
    assert class_parameters[: len(class_prefix)] == class_prefix
    assert function_parameters[: len(function_prefix)] == function_prefix


def test_full_swe_skip_nt_reduces_timeseries_length() -> None:
    sim = _small_simulator(
        nx=24,
        ny=24,
        Lx=24.0,
        Ly=24.0,
        T=5.0,
        dt_save=1.0,
        skip_nt=2,
    )

    out = sim.forward_samples_spatiotemporal(n=1, random_seed=0)
    data = out["data"]

    expected_frames = int(sim.T / sim.dt_save) + 1 - sim.skip_nt
    assert data.shape == (1, expected_frames, sim.nx, sim.ny, 3)


def test_full_swe_skip_nt_too_large_raises() -> None:
    with pytest.raises(ValueError, match="skip_nt is too large"):
        _run_small_swe(
            amp=0.12,
            nx=24,
            ny=24,
            Lx=24.0,
            Ly=24.0,
            T=1.0,
            dt_save=1.0,
            skip_nt=2,
        )


def test_full_swe_terminal_only_run_is_finite() -> None:
    result = _run_small_swe(return_timeseries=False, dt_save=1.0)
    assert result.shape == (1, 18, 18, 3)
    assert torch.isfinite(result).all()


def test_full_swe_includes_non_regular_terminal_snapshot() -> None:
    sim = _small_simulator(T=0.25)
    data = sim.forward_samples_spatiotemporal(n=1, random_seed=2)["data"]
    assert data.shape[1] == 4  # 0.0, 0.1, 0.2, and terminal 0.25


def test_full_swe_preserves_mean_depth_without_clipping() -> None:
    sim = _small_simulator(
        T=0.2,
        nu=0.0,
        drag=0.0,
        parameters_range={"amp": (0.05, 0.05)},
    )
    height = sim.forward_samples_spatiotemporal(n=1, random_seed=4)["data"][0, ..., 0]
    torch.testing.assert_close(height.mean(dim=(-2, -1)), height[0].mean().expand(3))


def test_zero_amplitude_is_a_rest_state() -> None:
    sim = _small_simulator(parameters_range={"amp": (0.0, 0.0)})
    data = sim.forward_samples_spatiotemporal(n=1, random_seed=5)["data"]
    torch.testing.assert_close(data[..., 0], torch.ones_like(data[..., 0]))
    assert torch.count_nonzero(data[..., 1:]) == 0


def test_periodic_beta_coriolis_matches_across_boundary() -> None:
    Ly = 12.0
    y = torch.tensor([0.0, Ly / 2.0, Ly], dtype=torch.float64)
    f = _coriolis_grid(y, f0=1.0, beta=0.2, Ly=Ly, mode="periodic_beta")
    torch.testing.assert_close(f[0], f[-1])
    torch.testing.assert_close(f[1], torch.tensor(1.0, dtype=f.dtype))


def test_zero_f0_disables_rotation_and_flattens_initial_height() -> None:
    y = torch.linspace(0.0, 18.0, 18, dtype=torch.float64)
    f = _coriolis_grid(y, f0=0.0, beta=0.0, Ly=18.0, mode="f_plane")
    assert torch.count_nonzero(f) == 0

    sim = _small_simulator(
        T=0.0,
        f0=0.0,
        coriolis_mode="f_plane",
    )
    initial = sim.forward_samples_spatiotemporal(n=1, random_seed=5)["data"][0, 0]

    torch.testing.assert_close(initial[..., 0], torch.ones_like(initial[..., 0]))
    assert torch.count_nonzero(initial[..., 1:]) > 0


def test_default_f0_matches_explicit_derived_value() -> None:
    torch.manual_seed(7)
    default = _run_small_swe(T=0.0)
    torch.manual_seed(7)
    explicit = _run_small_swe(T=0.0, f0=(9.81**0.5) / 8.0)

    torch.testing.assert_close(default, explicit)


def test_vortical_forcing_is_divergence_free() -> None:
    energy_rate = 2e-3
    T = 0.01
    result = _run_small_swe(
        amp=0.0,
        T=T,
        dt_save=T,
        forcing_type="vortical",
        forcing_energy_rate=energy_rate,
        **FORCING_OPTIONS,
    )
    final_u = result[-1, ..., 1].double()
    final_v = result[-1, ..., 2].double()
    kinetic_energy = 0.5 * (final_u.square() + final_v.square()).mean()

    kx = 2j * torch.pi * torch.fft.fftfreq(24, d=1.0)
    ky = 2j * torch.pi * torch.fft.rfftfreq(24, d=1.0)
    divergence = torch.fft.irfft2(
        kx[:, None] * torch.fft.rfft2(final_u) + ky[None, :] * torch.fft.rfft2(final_v),
        s=(24, 24),
    )
    assert kinetic_energy.item() > 0
    assert divergence.std().item() < 1e-7


def test_forcing_is_returned_as_transition_aligned_additional_input() -> None:
    sim = _small_simulator(
        return_additional_input_fields=True,
        T=0.01,
        dt_save=0.01,
        forcing_type="vortical",
        forcing_energy_rate=2e-3,
        parameters_range={"amp": (0.0, 0.0)},
        **FORCING_OPTIONS,
    )
    result = sim.forward_samples_spatiotemporal(n=1, random_seed=3)
    data = result["data"]
    additional = result["additional_input_fields"]

    assert additional is not None
    assert additional.shape == data.shape == (1, 2, 24, 24, 3)
    assert sim.additional_input_names == ["forcing_h", "forcing_u", "forcing_v"]
    torch.testing.assert_close(additional[:, -1], torch.zeros_like(additional[:, -1]))
    torch.testing.assert_close(data[:, 1] - data[:, 0], additional[:, 0])
    assert torch.count_nonzero(additional[:, 0, ..., 1:]) > 0


def test_unforced_additional_input_fields_are_zero() -> None:
    sim = _small_simulator(
        return_additional_input_fields=True,
        parameters_range={"amp": (0.05, 0.05)},
    )
    result = sim.forward_samples_spatiotemporal(n=1, random_seed=4)
    additional = result["additional_input_fields"]

    assert additional is not None
    assert torch.count_nonzero(additional) == 0


def test_additional_input_fields_require_timeseries() -> None:
    with pytest.raises(ValueError, match="requires return_timeseries=True"):
        ShallowWater2D(return_additional_input_fields=True)


def test_balanced_forcing_is_geostrophic() -> None:
    energy_rate = 2e-3
    T = 0.01
    g = 9.81
    h_mean = 1.0
    result = _run_small_swe(
        amp=0.0,
        T=T,
        dt_save=T,
        g=g,
        h_mean=h_mean,
        forcing_type="balanced",
        forcing_energy_rate=energy_rate,
        **FORCING_OPTIONS,
    )
    dh = result[-1, ..., 0].double() - h_mean
    du = result[-1, ..., 1].double()
    dv = result[-1, ..., 2].double()
    linear_energy = (
        0.5 * (du.square() + dv.square() + (g / h_mean) * dh.square()).mean()
    )

    kx = 2j * torch.pi * torch.fft.fftfreq(24, d=1.0)
    ky = 2j * torch.pi * torch.fft.rfftfreq(24, d=1.0)
    dh_hat = torch.fft.rfft2(dh)
    dh_dx = torch.fft.irfft2(kx[:, None] * dh_hat, s=(24, 24))
    dh_dy = torch.fft.irfft2(ky[None, :] * dh_hat, s=(24, 24))
    f0 = (g * h_mean) ** 0.5 / 8.0

    assert dh.std().item() > 0
    assert dh.mean().item() == pytest.approx(0.0, abs=1e-8)
    assert linear_energy.item() > 0
    torch.testing.assert_close(f0 * dv, g * dh_dx, rtol=1e-4, atol=2e-6)
    torch.testing.assert_close(-f0 * du, g * dh_dy, rtol=1e-4, atol=2e-6)
    assert torch.count_nonzero(du) > 0
    assert torch.count_nonzero(dv) > 0


@pytest.mark.parametrize("g", [0.0, 9.81])
def test_balanced_forcing_matches_vortical_when_f0_is_zero(g: float) -> None:
    forcing_options: dict[str, Any] = {
        "amp": 0.0,
        "T": 0.01,
        "dt_save": 0.01,
        "g": g,
        "nu": 0.0,
        "drag": 0.0,
        "f0": 0.0,
        "coriolis_mode": "f_plane",
        "forcing_energy_rate": 1e-3,
    }
    torch.manual_seed(7)
    vortical = _run_small_swe(**forcing_options, forcing_type="vortical")
    torch.manual_seed(7)
    balanced = _run_small_swe(**forcing_options, forcing_type="balanced")

    torch.testing.assert_close(balanced, vortical)


def test_zero_gravity_velocity_is_independent_of_height() -> None:
    options: dict[str, Any] = {
        "amp": 0.01,
        "T": 0.2,
        "dt_save": 0.1,
        "g": 0.0,
        "nu": 0.05,
        "drag": 0.05,
        "f0": 0.0,
        "beta": 0.0,
        "coriolis_mode": "f_plane",
        "forcing_type": "vortical",
        "forcing_energy_rate": 1e-4,
        "forcing_correlation_time": 0.1,
    }

    torch.manual_seed(7)
    shallow = _run_small_swe(h_mean=1.0, **options)
    torch.manual_seed(7)
    deep = _run_small_swe(h_mean=2.0, **options)

    torch.testing.assert_close(deep[..., 1:], shallow[..., 1:])
    torch.testing.assert_close(deep[..., 0], 2.0 * shallow[..., 0])
    torch.testing.assert_close(shallow[0, ..., 0], torch.ones_like(shallow[0, ..., 0]))


def test_zero_gravity_balanced_forcing_requires_zero_f0() -> None:
    with pytest.raises(ValueError, match="balanced forcing with g=0 requires f0=0"):
        _run_small_swe(
            g=0.0,
            f0=1.0,
            forcing_type="balanced",
        )


def test_negative_gravity_raises() -> None:
    with pytest.raises(ValueError, match="g must be non-negative"):
        _run_small_swe(g=-1.0)


def test_momentum_forcing_includes_divergent_velocity() -> None:
    energy_rate = 2e-3
    T = 0.01
    nx = ny = 24
    result = _run_small_swe(
        amp=0.0,
        T=T,
        dt_save=T,
        forcing_type="momentum",
        forcing_energy_rate=energy_rate,
        **FORCING_OPTIONS,
    )
    u = result[-1, ..., 1].double()
    v = result[-1, ..., 2].double()
    kinetic_energy = 0.5 * (u.square() + v.square()).mean()

    kx = 2j * torch.pi * torch.fft.fftfreq(nx, d=1.0)
    ky = 2j * torch.pi * torch.fft.rfftfreq(ny, d=1.0)
    divergence = torch.fft.irfft2(
        kx[:, None] * torch.fft.rfft2(u) + ky[None, :] * torch.fft.rfft2(v),
        s=(nx, ny),
    )
    assert kinetic_energy.item() > 0
    assert divergence.std().item() > 0


@pytest.mark.parametrize("forcing_type", ["vortical", "balanced", "momentum"])
def test_forcing_energy_is_an_ensemble_mean(forcing_type: str) -> None:
    nx = ny = 24
    Lx = Ly = 24.0
    g = 9.81
    h_mean = 1.0
    target_energy = 2e-3
    _, _, dKx, dKy = spectral_wavenumbers(nx, ny, Lx, Ly, dtype=torch.float64)
    K2 = dKx.square() + dKy.square()
    K2_inv = torch.where(K2 > 0, -1.0 / K2, torch.zeros_like(K2))
    mask = two_thirds_mask(nx, ny)
    spectrum = gaussian_ring_spectrum(
        K2,
        4.0 * 2.0 * torch.pi / Lx,
        1.5 * 2.0 * torch.pi / Lx,
        mask,
    )
    f0 = (g * h_mean) ** 0.5 / 8.0

    torch.manual_seed(123)
    energies = []
    for _ in range(128):
        dh, du, dv = _sample_swe_forcing_field(
            forcing_type=forcing_type,
            nx=nx,
            ny=ny,
            target_energy=target_energy,
            g=g,
            h_mean=h_mean,
            f0=f0,
            dtype=torch.float64,
            spectrum=spectrum,
            mask=mask,
            K2_inv=K2_inv,
            dKx=dKx,
            dKy=dKy,
        )
        energies.append(
            0.5 * (du.square() + dv.square() + (g / h_mean) * dh.square()).mean()
        )

    sampled_energies = torch.stack(energies)
    assert sampled_energies.mean().item() == pytest.approx(target_energy, rel=0.08)
    assert sampled_energies.std().item() > 0.02 * target_energy


def test_ou_forcing_is_temporally_correlated() -> None:
    forcing_options: dict[str, Any] = {
        "return_additional_input_fields": True,
        "T": 1.0,
        "forcing_type": "momentum",
        "forcing_energy_rate": 1e-4,
        "parameters_range": {"amp": (0.0, 0.0)},
        **FORCING_OPTIONS,
    }
    white = _small_simulator(**forcing_options, forcing_correlation_time=0.0)
    correlated = _small_simulator(**forcing_options, forcing_correlation_time=0.5)

    white_fields = white.forward_samples_spatiotemporal(n=1, random_seed=9)[
        "additional_input_fields"
    ][0, :-1, ..., 1:].flatten(start_dim=1)
    correlated_fields = correlated.forward_samples_spatiotemporal(n=1, random_seed=9)[
        "additional_input_fields"
    ][0, :-1, ..., 1:].flatten(start_dim=1)

    white_lag_one = torch.nn.functional.cosine_similarity(
        white_fields[:-1], white_fields[1:], dim=1
    ).mean()
    correlated_lag_one = torch.nn.functional.cosine_similarity(
        correlated_fields[:-1], correlated_fields[1:], dim=1
    ).mean()

    assert correlated_lag_one.item() > 0.5
    assert correlated_lag_one.item() > white_lag_one.item() + 0.4


@pytest.mark.parametrize("forcing_type", ["thermal", "height"])
def test_invalid_forcing_type_raises(forcing_type: str) -> None:
    with pytest.raises(ValueError, match="forcing_type must be one of"):
        ShallowWater2D(forcing_type=forcing_type)


def test_negative_forcing_correlation_time_raises() -> None:
    with pytest.raises(ValueError, match="forcing_correlation_time"):
        _run_small_swe(
            amp=0.0,
            nu=0.0,
            drag=0.0,
            forcing_type="vortical",
            forcing_correlation_time=-1.0,
        )


def test_stochastic_swe_seed_reproduces_forcing_path() -> None:
    sim = _small_simulator(
        T=0.05,
        dt_save=0.05,
        forcing_type="vortical",
        forcing_energy_rate=1e-3,
        parameters_range={"amp": (0.0, 0.0)},
        **FORCING_OPTIONS,
    )
    first = sim.forward_samples_spatiotemporal(n=1, random_seed=11)["data"]
    second = sim.forward_samples_spatiotemporal(n=1, random_seed=11)["data"]
    different = sim.forward_samples_spatiotemporal(n=1, random_seed=12)["data"]

    torch.testing.assert_close(first, second)
    assert not torch.equal(first, different)


def test_zero_rate_forcing_matches_unforced_solver() -> None:
    unforced = _small_simulator(parameters_range={"amp": (0.05, 0.05)})
    zero_rate = _small_simulator(
        parameters_range={"amp": (0.05, 0.05)},
        forcing_type="vortical",
        forcing_energy_rate=0.0,
    )
    expected = unforced.forward_samples_spatiotemporal(n=1, random_seed=6)["data"]
    actual = zero_rate.forward_samples_spatiotemporal(n=1, random_seed=6)["data"]
    torch.testing.assert_close(actual, expected)
