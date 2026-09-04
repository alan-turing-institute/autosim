import inspect
import math
from typing import Any

import pytest
import torch

import autosim.experimental.simulations.shallow_water as shallow_water_module
from autosim.experimental.simulations import ShallowWater2D
from autosim.experimental.simulations._spectral import (
    gaussian_ring_spectrum,
    spectral_wavenumbers,
    two_thirds_mask,
)
from autosim.experimental.simulations.shallow_water import (
    _coriolis_grid,
    _ou_step_coefficients,
    _sample_swe_forcing_field,
    _swe_forcing_expected_unit_energy,
    _swe_forcing_streamfunction_transfer,
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
    result = simulate_swe_2d(**options)
    if isinstance(result, tuple):
        msg = "_run_small_swe does not request energy-budget diagnostics"
        raise RuntimeError(msg)
    return result


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


def test_full_swe_validates_initial_state_when_t_is_zero() -> None:
    with pytest.raises(RuntimeError, match="initial state saturated"):
        _run_small_swe(amp=1e7, T=0.0)


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


def test_random_initial_condition_remains_default() -> None:
    torch.manual_seed(7)
    default = _run_small_swe(T=0.0)
    torch.manual_seed(7)
    explicit = _run_small_swe(T=0.0, initial_condition="random")

    torch.testing.assert_close(default, explicit)


def test_balanced_double_jet_is_periodic_and_geostrophic() -> None:
    amp = 0.1
    g = 9.81
    h_mean = 1.0
    nx = ny = 24
    result = _run_small_swe(
        amp=amp,
        nx=nx,
        ny=ny,
        Lx=24.0,
        Ly=24.0,
        T=0.0,
        g=g,
        h_mean=h_mean,
        coriolis_mode="f_plane",
        initial_condition="balanced_double_jet",
        jet_mode=1,
        jet_perturbation_fraction=0.0,
    )[0]
    h, u, v = result.double().unbind(dim=-1)
    f0 = (g * h_mean) ** 0.5 / 8.0
    ky = 2j * torch.pi * torch.fft.rfftfreq(ny, d=1.0)
    dh_dy = torch.fft.irfft2(ky[None, :] * torch.fft.rfft2(h), s=(nx, ny))

    assert float(u.mean()) == pytest.approx(0.0, abs=1e-7)
    assert float(torch.sqrt(u.square().mean())) == pytest.approx(amp, rel=1e-5)
    assert torch.count_nonzero(v) == 0
    torch.testing.assert_close(
        -f0 * u,
        g * dh_dy,
        rtol=2e-5,
        atol=2e-6,
    )
    torch.testing.assert_close(h.mean(), torch.tensor(h_mean, dtype=h.dtype))


def test_balanced_random_pv_is_non_zonal_and_geostrophic() -> None:
    amp = 0.1
    g = 9.81
    h_mean = 1.0
    nx = ny = 24
    domain_size = 24.0
    result = _run_small_swe(
        amp=amp,
        nx=nx,
        ny=ny,
        Lx=domain_size,
        Ly=domain_size,
        T=0.0,
        g=g,
        h_mean=h_mean,
        coriolis_mode="f_plane",
        initial_condition="balanced_random_pv",
        initial_wavenumber=4.0 * 2.0 * torch.pi / domain_size,
        initial_bandwidth=0.5 * 2.0 * torch.pi / domain_size,
    )[0]
    h, u, v = result.double().unbind(dim=-1)
    f0 = (g * h_mean) ** 0.5 / 8.0
    kx = 2j * torch.pi * torch.fft.fftfreq(nx, d=domain_size / nx)
    ky = 2j * torch.pi * torch.fft.rfftfreq(ny, d=domain_size / ny)
    dh_hat = torch.fft.rfft2(h)
    dh_dx = torch.fft.irfft2(kx[:, None] * dh_hat, s=(nx, ny))
    dh_dy = torch.fft.irfft2(ky[None, :] * dh_hat, s=(nx, ny))

    assert float(torch.sqrt((u.square() + v.square()).mean())) == pytest.approx(
        amp, rel=1e-5
    )
    assert u.std().item() > 0
    assert v.std().item() > 0
    torch.testing.assert_close(f0 * v, g * dh_dx, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(-f0 * u, g * dh_dy, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(h.mean(), torch.tensor(h_mean, dtype=h.dtype))


def test_balanced_random_pv_seed_is_reproducible() -> None:
    options: dict[str, Any] = {
        "T": 0.0,
        "initial_condition": "balanced_random_pv",
    }
    torch.manual_seed(7)
    first = _run_small_swe(**options)
    torch.manual_seed(7)
    repeated = _run_small_swe(**options)
    torch.manual_seed(8)
    different = _run_small_swe(**options)

    torch.testing.assert_close(repeated, first, rtol=0.0, atol=0.0)
    assert not torch.equal(different, first)


def test_balanced_random_pv_wavenumber_is_sampled_and_used() -> None:
    nx = ny = 24
    domain_size = 24.0
    fundamental_wavenumber = float(2.0 * torch.pi / domain_size)
    bandwidth = 0.4 * fundamental_wavenumber
    sim = _small_simulator(
        nx=nx,
        ny=ny,
        Lx=domain_size,
        Ly=domain_size,
        T=0.0,
        initial_condition="balanced_random_pv",
        parameters_range={
            "initial_bandwidth": (bandwidth, bandwidth),
            "amp": (0.1, 0.1),
            "initial_wavenumber": (
                2.0 * fundamental_wavenumber,
                6.0 * fundamental_wavenumber,
            ),
        },
    )
    result = sim.forward_samples_spatiotemporal(n=8, random_seed=7)
    states = result["data"][:, 0]
    parameters = result["constant_scalars"]

    sampled_wavenumbers = parameters[:, sim.get_parameter_idx("initial_wavenumber")]
    sampled_bandwidths = parameters[:, sim.get_parameter_idx("initial_bandwidth")]
    speed_rms = torch.sqrt(states[..., 1:].square().sum(dim=-1).mean(dim=(1, 2)))

    kx = 2.0 * torch.pi * torch.fft.fftfreq(nx, d=domain_size / nx)
    ky = 2.0 * torch.pi * torch.fft.fftfreq(ny, d=domain_size / ny)
    Kx, Ky = torch.meshgrid(kx, ky, indexing="ij")
    wavenumber_magnitude = torch.sqrt(Kx.square() + Ky.square())
    velocity_power = torch.fft.fft2(states[..., 1:], dim=(1, 2)).abs().square()
    velocity_power = velocity_power.sum(dim=-1)
    spectral_centroids = (velocity_power * wavenumber_magnitude.unsqueeze(0)).sum(
        dim=(1, 2)
    ) / velocity_power.sum(dim=(1, 2))

    assert sampled_wavenumbers.std().item() > 0
    torch.testing.assert_close(
        sampled_bandwidths, torch.full_like(sampled_bandwidths, bandwidth)
    )
    torch.testing.assert_close(
        speed_rms, torch.full_like(speed_rms, 0.1), rtol=1e-5, atol=1e-6
    )
    correlation = torch.corrcoef(
        torch.stack((sampled_wavenumbers, spectral_centroids))
    )[0, 1]
    assert correlation.item() > 0.9


def test_balanced_random_pv_sampled_bandwidth_controls_spectral_width() -> None:
    nx = ny = 24
    domain_size = 24.0
    fundamental_wavenumber = float(2.0 * torch.pi / domain_size)
    central_wavenumber = 4.0 * fundamental_wavenumber
    sim = _small_simulator(
        nx=nx,
        ny=ny,
        Lx=domain_size,
        Ly=domain_size,
        T=0.0,
        initial_condition="balanced_random_pv",
        parameters_range={
            "initial_bandwidth": (
                0.2 * fundamental_wavenumber,
                1.5 * fundamental_wavenumber,
            ),
            "amp": (0.1, 0.1),
            "initial_wavenumber": (central_wavenumber, central_wavenumber),
        },
    )

    def run_with_bandwidth(bandwidth: float) -> torch.Tensor:
        inputs = torch.tensor([[bandwidth, 0.1, central_wavenumber]])
        torch.manual_seed(7)
        output = sim.forward(inputs, allow_failures=False)
        assert output is not None
        return output.reshape(nx, ny, 3)

    narrow = run_with_bandwidth(0.2 * fundamental_wavenumber)
    broad = run_with_bandwidth(1.5 * fundamental_wavenumber)

    kx = 2.0 * torch.pi * torch.fft.fftfreq(nx, d=domain_size / nx)
    ky = 2.0 * torch.pi * torch.fft.fftfreq(ny, d=domain_size / ny)
    Kx, Ky = torch.meshgrid(kx, ky, indexing="ij")
    wavenumber_magnitude = torch.sqrt(Kx.square() + Ky.square())

    def spectral_width(state: torch.Tensor) -> torch.Tensor:
        power = torch.fft.fft2(state[..., 1:], dim=(0, 1)).abs().square().sum(dim=-1)
        centroid = (power * wavenumber_magnitude).sum() / power.sum()
        return torch.sqrt(
            (power * (wavenumber_magnitude - centroid).square()).sum() / power.sum()
        )

    assert float(spectral_width(broad)) > 2.0 * float(spectral_width(narrow))


def test_restart_initial_condition_preserves_supplied_state() -> None:
    source = _run_small_swe(T=0.0)[0]
    restarted = _run_small_swe(
        T=0.0,
        initial_condition="restart",
        initial_state=source,
    )[0]

    torch.testing.assert_close(restarted, source, rtol=0.0, atol=0.0)


def test_restart_accepts_singleton_batch_dimension() -> None:
    source = _run_small_swe(T=0.0)
    restarted = _run_small_swe(
        T=0.0,
        initial_condition="restart",
        initial_state=source,
    )

    torch.testing.assert_close(restarted, source, rtol=0.0, atol=0.0)


def test_restart_dataset_omits_dead_amp_conditioning() -> None:
    source = _run_small_swe(T=0.0)[0]
    simulator = _small_simulator(
        initial_condition="restart",
        initial_state=source,
        parameters_range={},
    )

    result = simulator.forward_samples_spatiotemporal(n=4, random_seed=7)

    assert result["constant_scalars"].shape == (4, 0)
    expected = result["data"][:1].expand_as(result["data"])
    torch.testing.assert_close(result["data"], expected, rtol=0.0, atol=0.0)


def test_restart_batch_branches_into_distinct_stochastic_futures() -> None:
    source = _run_small_swe(T=0.0)[0]
    simulator = _small_simulator(
        T=0.05,
        dt_save=0.05,
        initial_condition="restart",
        initial_state=source,
        forcing_type="vortical",
        forcing_energy_rate=1e-3,
        forcing_correlation_time=0.1,
        parameters_range={},
    )

    data = simulator.forward_samples_spatiotemporal(n=4, random_seed=7)["data"]

    torch.testing.assert_close(
        data[:, 0], source.unsqueeze(0).expand_as(data[:, 0]), rtol=0.0, atol=0.0
    )
    assert all(not torch.equal(data[0, -1], data[index, -1]) for index in range(1, 4))


def test_restart_requires_valid_initial_state() -> None:
    with pytest.raises(ValueError, match="initial_state is required"):
        ShallowWater2D(initial_condition="restart")

    with pytest.raises(ValueError, match="initial_state must have shape"):
        _run_small_swe(
            initial_condition="restart",
            initial_state=torch.zeros(4, 4, 3),
        )

    invalid_height = torch.zeros(18, 18, 3)
    with pytest.raises(ValueError, match="height must be strictly positive"):
        _run_small_swe(
            initial_condition="restart",
            initial_state=invalid_height,
        )

    source = _run_small_swe(T=0.0)[0]
    with pytest.raises(ValueError, match="amp is not valid"):
        ShallowWater2D(
            initial_condition="restart",
            initial_state=source,
            parameters_range={"amp": (0.1, 0.2)},
        )


def test_invalid_initial_condition_raises() -> None:
    with pytest.raises(ValueError, match="initial_condition must be one of"):
        ShallowWater2D(initial_condition="jet")

    with pytest.raises(ValueError, match="initial_wavenumber"):
        ShallowWater2D(initial_wavenumber=0.0)

    with pytest.raises(ValueError, match="initial_bandwidth"):
        ShallowWater2D(initial_bandwidth=-1.0)

    with pytest.raises(ValueError, match="initial_wavenumber range"):
        ShallowWater2D(
            initial_condition="balanced_random_pv",
            parameters_range={
                "amp": (0.1, 0.1),
                "initial_wavenumber": (0.0, 1.0),
            },
        )

    with pytest.raises(ValueError, match="forcing_bandwidth range"):
        ShallowWater2D(
            parameters_range={
                "amp": (0.1, 0.1),
                "forcing_bandwidth": (float("nan"), 1.0),
            }
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [("forcing_wavenumber", 0.0), ("forcing_bandwidth", float("nan"))],
)
def test_invalid_forcing_scale_raises(name: str, value: float) -> None:
    options: dict[str, Any] = {name: value}
    with pytest.raises(ValueError, match=name):
        ShallowWater2D(**options)


def test_constructor_validates_grid_before_spectral_ring() -> None:
    with pytest.raises(ValueError, match="nx and ny must be at least 6"):
        ShallowWater2D(nx=5)

    with pytest.raises(ValueError, match="Lx and Ly must be positive"):
        ShallowWater2D(Lx=0.0)


@pytest.mark.parametrize(
    ("grid_size", "options"),
    [
        (6, {"forcing_type": "vortical"}),
        (12, {"initial_condition": "balanced_random_pv"}),
    ],
)
def test_small_grid_spectral_ring_defaults_are_clamped(
    grid_size: int, options: dict[str, Any]
) -> None:
    sim = _small_simulator(
        nx=grid_size,
        ny=grid_size,
        Lx=float(grid_size),
        Ly=float(grid_size),
        T=0.0,
        **options,
    )

    data = sim.forward_samples_spatiotemporal(n=1, random_seed=0)["data"]

    assert data.shape == (1, 1, grid_size, grid_size, 3)
    assert torch.isfinite(data).all()


@pytest.mark.parametrize(
    ("ring_name", "options"),
    [
        (
            "forcing_wavenumber",
            {
                "forcing_type": "vortical",
                "parameters_range": {
                    "amp": (0.1, 0.1),
                    "forcing_wavenumber": (4.0, 20.0),
                },
            },
        ),
        (
            "initial_wavenumber",
            {
                "initial_condition": "balanced_random_pv",
                "parameters_range": {
                    "amp": (0.1, 0.1),
                    "initial_wavenumber": (4.0, 20.0),
                },
            },
        ),
    ],
)
def test_sampled_spectral_ring_must_fit_isotropic_band(
    ring_name: str, options: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match=f"{ring_name}.*isotropically retained"):
        ShallowWater2D(
            nx=24,
            ny=24,
            Lx=2.0 * math.pi,
            Ly=2.0 * math.pi,
            **options,
        )


def test_direct_forcing_ring_must_fit_isotropic_band() -> None:
    fundamental_wavenumber = 2.0 * math.pi / 24.0

    with pytest.raises(ValueError, match=r"forcing_wavenumber.*isotropically retained"):
        _run_small_swe(
            nx=24,
            ny=24,
            Lx=24.0,
            Ly=24.0,
            T=0.0,
            forcing_type="vortical",
            forcing_wavenumber=8.0 * fundamental_wavenumber,
            forcing_bandwidth=0.5 * fundamental_wavenumber,
        )


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


def test_energy_budget_is_transition_aligned_and_closes() -> None:
    sim = _small_simulator(
        return_energy_budget=True,
        return_additional_input_fields=True,
        T=0.2,
        dt_save=0.1,
        forcing_type="vortical",
        forcing_energy_rate=1e-4,
        parameters_range={"amp": (0.05, 0.05)},
    )
    result = sim.forward_samples_spatiotemporal(n=1, random_seed=4)
    data = result["data"]
    budget = result["energy_budget"]

    assert budget is not None
    assert budget.shape == (1, data.shape[1], len(sim.energy_budget_names))
    assert sim.energy_budget_names == [
        "total_energy",
        "deterministic_energy_change",
        "hyperviscous_energy_change",
        "forcing_energy_change",
        "viscous_dissipation_estimate",
        "drag_dissipation_estimate",
        "effective_forcing_energy_rate",
    ]
    total_change = budget[:, 1:, 0] - budget[:, :-1, 0]
    decomposed_change = budget[:, :-1, 1:4].sum(dim=-1)
    torch.testing.assert_close(total_change, decomposed_change, rtol=2e-4, atol=2e-7)
    torch.testing.assert_close(budget[:, -1, 1:], torch.zeros_like(budget[:, -1, 1:]))
    assert torch.all(budget[:, :-1, 4:6] >= 0)
    torch.testing.assert_close(
        budget[:, :-1, 6],
        torch.full_like(budget[:, :-1, 6], sim.forcing_energy_rate),
    )


def test_energy_budget_does_not_change_trajectory() -> None:
    baseline = _small_simulator(
        forcing_type="vortical",
        forcing_energy_rate=1e-4,
        parameters_range={"amp": (0.05, 0.05)},
    )
    diagnosed = _small_simulator(
        return_energy_budget=True,
        forcing_type="vortical",
        forcing_energy_rate=1e-4,
        parameters_range={"amp": (0.05, 0.05)},
    )

    expected = baseline.forward_samples_spatiotemporal(n=1, random_seed=4)["data"]
    actual = diagnosed.forward_samples_spatiotemporal(n=1, random_seed=4)["data"]
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_unforced_energy_budget_has_no_forcing_terms() -> None:
    sim = _small_simulator(return_energy_budget=True)
    budget = sim.forward_samples_spatiotemporal(n=1, random_seed=4)["energy_budget"]

    assert budget is not None
    assert torch.count_nonzero(budget[..., 3]) == 0
    assert torch.count_nonzero(budget[..., 6]) == 0


def test_energy_budget_requires_timeseries() -> None:
    with pytest.raises(ValueError, match="requires return_timeseries=True"):
        ShallowWater2D(return_energy_budget=True)


@pytest.mark.parametrize("forcing_type", ["balanced", "pv_balanced"])
def test_balanced_forcing_is_geostrophic(forcing_type: str) -> None:
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
        forcing_type=forcing_type,
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
@pytest.mark.parametrize("forcing_type", ["balanced", "pv_balanced"])
def test_balanced_forcing_matches_vortical_when_f0_is_zero(
    g: float, forcing_type: str
) -> None:
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
    balanced = _run_small_swe(**forcing_options, forcing_type=forcing_type)

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


@pytest.mark.parametrize("forcing_type", ["balanced", "pv_balanced"])
def test_zero_gravity_balanced_forcing_requires_zero_f0(forcing_type: str) -> None:
    with pytest.raises(ValueError, match="forcing with g=0 requires f0=0"):
        _run_small_swe(
            g=0.0,
            f0=1.0,
            forcing_type=forcing_type,
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


@pytest.mark.parametrize(
    "forcing_type", ["vortical", "balanced", "pv_balanced", "momentum"]
)
def test_forcing_energy_is_an_ensemble_mean(forcing_type: str) -> None:
    nx = ny = 24
    Lx = Ly = 24.0
    g = 9.81
    h_mean = 1.0
    target_energy = 2e-3
    Kx, Ky, dKx, dKy = spectral_wavenumbers(nx, ny, Lx, Ly, dtype=torch.float64)
    K2 = Kx.square() + Ky.square()
    K2_inv = torch.where(K2 > 0, -1.0 / K2, torch.zeros_like(K2))
    mask = two_thirds_mask(nx, ny)
    spectrum = gaussian_ring_spectrum(
        K2,
        4.0 * 2.0 * torch.pi / Lx,
        1.5 * 2.0 * torch.pi / Lx,
        mask,
    )
    f0 = 2.0 if forcing_type == "pv_balanced" else (g * h_mean) ** 0.5 / 8.0
    expected_unit_energy = _swe_forcing_expected_unit_energy(
        forcing_type=forcing_type,
        nx=nx,
        ny=ny,
        g=g,
        h_mean=h_mean,
        f0=f0,
        spectrum=spectrum,
        K2=K2,
        K2_inv=K2_inv,
        dKx=dKx,
        dKy=dKy,
    )

    sample_options: dict[str, Any] = {
        "forcing_type": forcing_type,
        "nx": nx,
        "ny": ny,
        "target_energy": target_energy,
        "g": g,
        "h_mean": h_mean,
        "f0": f0,
        "dtype": torch.float64,
        "spectrum": spectrum,
        "mask": mask,
        "K2": K2,
        "K2_inv": K2_inv,
        "dKx": dKx,
        "dKy": dKy,
    }

    torch.manual_seed(122)
    fallback_sample = _sample_swe_forcing_field(**sample_options)
    torch.manual_seed(122)
    cached_sample = _sample_swe_forcing_field(
        **sample_options,
        expected_unit_energy=expected_unit_energy,
    )
    for fallback_field, cached_field in zip(
        fallback_sample, cached_sample, strict=True
    ):
        torch.testing.assert_close(cached_field, fallback_field, rtol=0.0, atol=0.0)

    torch.manual_seed(123)
    energies = []
    for _ in range(128):
        dh, du, dv = _sample_swe_forcing_field(
            **sample_options,
            expected_unit_energy=expected_unit_energy,
        )
        energies.append(
            0.5 * (du.square() + dv.square() + (g / h_mean) * dh.square()).mean()
        )

    sampled_energies = torch.stack(energies)
    assert sampled_energies.mean().item() == pytest.approx(target_energy, rel=0.08)
    assert sampled_energies.std().item() > 0.02 * target_energy


def test_pv_transfer_uses_true_laplacian_wavenumbers() -> None:
    nx = ny = 12
    Kx, Ky, dKx, dKy = spectral_wavenumbers(nx, ny, 12.0, 12.0, dtype=torch.float64)
    K2 = Kx.square() + Ky.square()
    K2_inv = torch.where(K2 > 0, -1.0 / K2, torch.zeros_like(K2))
    h_mean = 2.0
    g = 4.0
    f0 = 3.0

    transfer = _swe_forcing_streamfunction_transfer(
        forcing_type="pv_balanced",
        g=g,
        h_mean=h_mean,
        f0=f0,
        K2=K2,
        K2_inv=K2_inv,
    )
    expected = torch.where(
        K2 > 0,
        -h_mean / (K2 + f0**2 / (g * h_mean)),
        torch.zeros_like(K2),
    )

    torch.testing.assert_close(transfer, expected, rtol=0.0, atol=0.0)
    assert not torch.equal(K2, dKx.square() + dKy.square())


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


def test_ou_forcing_uses_exact_integrated_energy() -> None:
    energy_rate = 2e-3
    correlation_time = 1e-3
    step_dt = 0.01
    sim = _small_simulator(
        return_additional_input_fields=True,
        T=step_dt,
        dt_save=step_dt,
        g=0.0,
        f0=0.0,
        beta=0.0,
        forcing_type="momentum",
        forcing_energy_rate=energy_rate,
        forcing_correlation_time=correlation_time,
        parameters_range={"amp": (0.0, 0.0)},
        **FORCING_OPTIONS,
    )

    impulses = sim.forward_samples_spatiotemporal(n=128, random_seed=12)[
        "additional_input_fields"
    ][:, 0]
    sampled_energies = 0.5 * impulses[..., 1:].square().sum(dim=-1).mean(dim=(-2, -1))
    expected_integrated_energy = energy_rate * (
        step_dt - correlation_time * (1.0 - math.exp(-step_dt / correlation_time))
    )

    assert sampled_energies.mean().item() == pytest.approx(
        expected_integrated_energy, rel=0.08
    )
    assert sampled_energies.std().item() > 0.02 * expected_integrated_energy


def test_ou_integral_converges_to_white_noise_variance() -> None:
    step_dt = 0.04
    correlation_time = 1e-6
    _, _, endpoint_weight, innovation_variance = _ou_step_coefficients(
        step_dt=step_dt,
        correlation_time=correlation_time,
    )

    assert endpoint_weight == pytest.approx(correlation_time)
    assert innovation_variance == pytest.approx(step_dt, rel=1e-4)


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


def test_dissipation_backscatter_drives_zero_base_rate() -> None:
    sim = _small_simulator(
        return_energy_budget=True,
        return_additional_input_fields=True,
        T=0.1,
        dt_save=0.1,
        nu=0.05,
        forcing_type="pv_balanced",
        forcing_energy_rate=0.0,
        forcing_backscatter_fraction=0.5,
        parameters_range={"amp": (0.1, 0.1)},
    )
    result = sim.forward_samples_spatiotemporal(n=1, random_seed=8)
    budget = result["energy_budget"]
    additional = result["additional_input_fields"]

    assert budget is not None
    assert additional is not None
    assert float(budget[0, 0, 6]) > 0
    assert torch.count_nonzero(additional[0, 0]) > 0


def test_ou_backscatter_scales_both_exact_innovations(monkeypatch) -> None:
    target_energies: list[float] = []
    original_sampler = shallow_water_module._sample_swe_forcing_field

    def record_target_energy(**kwargs: Any):
        target_energies.append(kwargs["target_energy"])
        return original_sampler(**kwargs)

    monkeypatch.setattr(
        shallow_water_module,
        "_sample_swe_forcing_field",
        record_target_energy,
    )
    correlation_time = 0.1
    step_dt = 0.01
    simulator = _small_simulator(
        return_energy_budget=True,
        T=step_dt,
        dt_save=step_dt,
        nu=0.05,
        forcing_type="vortical",
        forcing_energy_rate=0.0,
        forcing_correlation_time=correlation_time,
        forcing_backscatter_fraction=0.5,
        parameters_range={"amp": (0.1, 0.1)},
    )

    budget = simulator.forward_samples_spatiotemporal(n=1, random_seed=8)[
        "energy_budget"
    ]

    assert budget is not None
    effective_rate = float(budget[0, 0, 6])
    _, _, _, integral_innovation_variance = _ou_step_coefficients(
        step_dt=step_dt,
        correlation_time=correlation_time,
    )
    assert target_energies == pytest.approx(
        [
            0.0,
            effective_rate / (2.0 * correlation_time),
            effective_rate * integral_innovation_variance,
        ],
        rel=1e-6,
        abs=1e-12,
    )


def test_backscatter_fraction_scales_diagnosed_rate() -> None:
    common: dict[str, Any] = {
        "return_energy_budget": True,
        "T": 0.01,
        "dt_save": 0.01,
        "nu": 0.05,
        "forcing_type": "vortical",
        "forcing_energy_rate": 0.0,
        "parameters_range": {"amp": (0.1, 0.1)},
    }
    weak = _small_simulator(**common, forcing_backscatter_fraction=0.25)
    strong = _small_simulator(**common, forcing_backscatter_fraction=0.5)

    weak_budget = weak.forward_samples_spatiotemporal(n=1, random_seed=8)[
        "energy_budget"
    ]
    strong_budget = strong.forward_samples_spatiotemporal(n=1, random_seed=8)[
        "energy_budget"
    ]

    assert weak_budget is not None
    assert strong_budget is not None
    assert float(weak_budget[0, 0, 6]) > 0
    assert float(strong_budget[0, 0, 6]) == pytest.approx(
        2.0 * float(weak_budget[0, 0, 6]), rel=1e-5
    )


def test_zero_backscatter_fraction_preserves_fixed_forcing() -> None:
    common: dict[str, Any] = {
        "forcing_type": "vortical",
        "forcing_energy_rate": 1e-4,
        "parameters_range": {"amp": (0.05, 0.05)},
    }
    default = _small_simulator(**common)
    explicit_zero = _small_simulator(**common, forcing_backscatter_fraction=0.0)

    expected = default.forward_samples_spatiotemporal(n=1, random_seed=6)["data"]
    actual = explicit_zero.forward_samples_spatiotemporal(n=1, random_seed=6)["data"]
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_negative_backscatter_fraction_raises() -> None:
    with pytest.raises(ValueError, match="forcing_backscatter_fraction"):
        ShallowWater2D(forcing_backscatter_fraction=-0.1)


def test_backscatter_requires_stochastic_forcing() -> None:
    with pytest.raises(ValueError, match="requires stochastic forcing"):
        ShallowWater2D(
            forcing_type="none",
            forcing_backscatter_fraction=50.0,
        )

    with pytest.raises(ValueError, match="requires stochastic forcing"):
        _run_small_swe(
            forcing_type="none",
            forcing_backscatter_fraction=50.0,
        )
