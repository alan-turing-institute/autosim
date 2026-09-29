"""Forward and backward contracts for the deterministic training stepper."""

import math

import pytest
import torch

from autosim.experimental.simulations import advance_swe_2d
from autosim.experimental.simulations.shallow_water import simulate_swe_2d

DOMAIN = 2 * math.pi
PHYSICS = {
    "Lx": DOMAIN,
    "Ly": DOMAIN,
    "g": 9.81,
    "h_mean": 1.0,
    "nu": 1e-4,
    "drag": 0.05,
    "f0": 1.0,
    "beta": 0.5,
    "coriolis_mode": "periodic_beta",
}


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def state(nx=12, ny=12, dtype=torch.float64, device="cpu"):
    x = torch.arange(nx, dtype=dtype, device=device)[:, None] * DOMAIN / nx
    y = torch.arange(ny, dtype=dtype, device=device)[None, :] * DOMAIN / ny
    h = 1 + 0.001 * torch.cos(x + y)
    u = 0.03 * torch.sin(x) * torch.cos(y)
    v = -0.02 * torch.cos(x) * torch.sin(y)
    return torch.stack((h, u, v), dim=-1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(12, 12), (12, 18), (9, 13)])
@pytest.mark.parametrize("rotation", ["f_plane", "periodic_beta"])
@pytest.mark.parametrize("dealias", [True, False])
def test_matches_existing_solver_when_timestep_matches(dtype, shape, rotation, dealias):
    initial = state(*shape, dtype=dtype)
    physics = {**PHYSICS, "coriolis_mode": rotation, "dealias": dealias}
    # Short enough that the adaptive solver takes exactly one RK4/filter step.
    dt = 0.001
    reference = simulate_swe_2d(
        amp=0,
        return_timeseries=False,
        nx=shape[0],
        ny=shape[1],
        T=dt,
        dt_save=dt,
        cfl=0.12,
        dtype=dtype,
        initial_condition="restart",
        initial_state=initial,
        forcing_type="none",
        **physics,
    )
    predicted = advance_swe_2d(initial, dt, n_substeps=1, **physics)
    assert isinstance(reference, torch.Tensor)
    assert predicted.dtype == dtype
    torch.testing.assert_close(predicted.float(), reference[0], atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_leading_batch_and_member_dimensions_are_independent(dtype):
    initial = torch.stack((state(dtype=dtype), state(dtype=dtype) * 1.01))
    initial = initial[:, None].expand(-1, 3, -1, -1, -1).clone().requires_grad_()
    result = advance_swe_2d(initial, 0.01, n_substeps=2, **PHYSICS)
    assert result.shape == initial.shape
    for batch in range(2):
        for member in range(3):
            expected = advance_swe_2d(
                initial[batch, member], 0.01, n_substeps=2, **PHYSICS
            )
            torch.testing.assert_close(result[batch, member], expected)
    (grad,) = torch.autograd.grad(result[0, 0].square().sum(), initial)
    assert grad[0, 0].abs().max() > 0
    assert torch.count_nonzero(grad[1]) == 0
    assert torch.count_nonzero(grad[0, 1:]) == 0


def test_input_rng_precision_and_identity_are_preserved():
    initial = state().requires_grad_()
    original = initial.detach().clone()
    rng = torch.random.get_rng_state()
    result = advance_swe_2d(initial, 0.01, n_substeps=2, **PHYSICS)
    torch.testing.assert_close(initial, original, atol=0, rtol=0)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert result.dtype == torch.float64
    assert result.device == initial.device
    identity = advance_swe_2d(initial, 0, n_substeps=1, **PHYSICS)
    torch.testing.assert_close(identity, initial, atol=0, rtol=0)
    assert identity.requires_grad


def test_uniform_inertial_motion_with_drag():
    initial = torch.zeros(12, 12, 3, dtype=torch.float64)
    initial[..., 0], initial[..., 1] = 1, 0.1
    dt = 0.25
    result = advance_swe_2d(
        initial, dt, n_substeps=32, **{**PHYSICS, "coriolis_mode": "f_plane"}
    )
    expected = torch.tensor(
        [
            1,
            0.1 * math.exp(-0.05 * dt) * math.cos(dt),
            -0.1 * math.exp(-0.05 * dt) * math.sin(dt),
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(result, expected.expand_as(result), atol=1e-11, rtol=0)


def test_autograd_agrees_with_finite_differences():
    initial = state(6, 6).requires_grad_()
    assert torch.autograd.gradcheck(
        lambda x: advance_swe_2d(x, 0.02, n_substeps=3, **PHYSICS),
        (initial,),
        eps=1e-6,
        atol=1e-6,
        rtol=1e-4,
        fast_mode=True,
    )


def test_later_height_loss_reaches_earlier_velocity_correction():
    initial = state(8, 8)
    basis = torch.zeros_like(initial)
    basis[..., 1] = state(8, 8)[..., 1]
    coefficient = torch.tensor(0.02, dtype=torch.float64, requires_grad=True)

    def loss(weight):
        first = advance_swe_2d(initial, 0.02, n_substeps=3, **PHYSICS)
        corrected = first + weight * basis
        last = advance_swe_2d(corrected, 0.02, n_substeps=3, **PHYSICS)
        # The correction has no height component. Only physics couples it here.
        return ((last[..., 0] - 1) * initial[..., 0]).mean()

    (grad,) = torch.autograd.grad(loss(coefficient), coefficient)
    eps = 1e-5
    difference = (
        loss(coefficient.detach() + eps) - loss(coefficient.detach() - eps)
    ) / (2 * eps)
    assert grad.abs() > 1e-9
    torch.testing.assert_close(grad, difference, atol=1e-10, rtol=1e-4)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dt": -1},
        {"dt": float("nan")},
        {"n_substeps": 0},
        {"n_substeps": 1.5},
        {"n_substeps": True},
        {"Lx": 0},
        {"g": -1},
        {"nu": -1},
        {"drag": -1},
        {"cfl": 0},
        {"f0": float("inf")},
        {"beta": float("nan")},
        {"coriolis_mode": "unknown"},
    ],
)
def test_invalid_configuration_fails(kwargs):
    options = {**PHYSICS, "dt": 0.01, "n_substeps": 2, **kwargs}
    with pytest.raises(ValueError, match=r"parameter|n_substeps|coriolis_mode|finite"):
        advance_swe_2d(state(), **options)


@pytest.mark.parametrize("kind", ["nan", "height", "velocity", "shape", "dtype"])
def test_invalid_state_is_rejected_not_sanitized(kind):
    initial = state()
    if kind == "nan":
        initial[0, 0, 1] = float("nan")
    elif kind == "height":
        initial[0, 0, 0] = -1
    elif kind == "velocity":
        initial[0, 0, 1] = 101
    elif kind == "shape":
        initial = initial[..., :2]
    else:
        initial = initial.half()
    with pytest.raises(ValueError, match="state"):
        advance_swe_2d(initial, 0.01, n_substeps=2, **PHYSICS)


def test_insufficient_substeps_fail_instead_of_silently_adapting():
    with pytest.raises(ValueError, match="CFL"):
        advance_swe_2d(state(), 0.25, n_substeps=1, **PHYSICS)


def test_viscosity_bound_is_checked():
    with pytest.raises(ValueError, match="viscosity"):
        advance_swe_2d(state(), 0.01, n_substeps=2, **{**PHYSICS, "nu": 100})


@pytest.mark.parametrize("parameter", ["dt", "g", "f0", "beta"])
def test_tensor_parameters_are_not_silently_detached(parameter):
    options = {**PHYSICS, "dt": 0.01, "n_substeps": 2}
    options[parameter] = torch.tensor(0.01, requires_grad=True)
    with pytest.raises(TypeError, match="Python scalars"):
        advance_swe_2d(state(), **options)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_preserves_device_and_backpropagates():
    initial = state(dtype=torch.float32, device="cuda").requires_grad_()
    result = advance_swe_2d(initial, 0.02, n_substeps=4, **PHYSICS)
    assert result.device == initial.device
    assert result.dtype == initial.dtype
    (grad,) = torch.autograd.grad(result.square().mean(), initial)
    assert torch.isfinite(grad).all()
    assert grad.abs().max() > 0
    reference = advance_swe_2d(initial.detach().cpu(), 0.02, n_substeps=4, **PHYSICS)
    torch.testing.assert_close(result.cpu(), reference, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("n_substeps", [1, 4])
def test_unresolved_inputs_do_not_alias_into_retained_modes(axis, n_substeps):
    coordinate = torch.arange(12, dtype=torch.float64) * DOMAIN / 12
    mode = torch.cos(5 * coordinate)
    basis = torch.zeros(12, 12, 3, dtype=torch.float64)
    basis[..., axis + 1] = mode[:, None] if axis == 0 else mode[None, :]
    rest = torch.zeros_like(basis)
    rest[..., 0] = 1.0
    amplitude = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
    initial = rest + amplitude * basis
    physics = {**PHYSICS, "g": 0.0, "f0": 0.0, "beta": 0.0, "nu": 0.0, "drag": 0.0}

    result = advance_swe_2d(initial, 0.001, n_substeps=n_substeps, **physics)
    torch.testing.assert_close(result, rest, atol=1e-12, rtol=0)
    weight = torch.sin(2 * coordinate)
    loss = (result[..., axis + 1] * (weight[:, None] if axis == 0 else weight)).mean()
    (gradient,) = torch.autograd.grad(loss, amplitude)
    torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-12, rtol=0)

    # Projection belongs to evolution: a zero lead remains an exact identity,
    # and disabling dealiasing keeps the supplied high-frequency mode.
    identity = advance_swe_2d(initial, 0, n_substeps=1, **physics)
    torch.testing.assert_close(identity, initial, atol=0, rtol=0)
    unfiltered = advance_swe_2d(initial, 0.001, n_substeps=1, dealias=False, **physics)
    assert (unfiltered[..., axis + 1] * basis[..., axis + 1]).mean().abs() > 0.01


def test_projected_state_must_remain_in_physical_range():
    initial = torch.zeros(12, 12, 3, dtype=torch.float64)
    initial[..., 0] = 0.01
    initial[0, :, 0] += 1.0
    with pytest.raises(ValueError, match="valid unclipped physical range"):
        advance_swe_2d(initial, 0.001, n_substeps=1, **PHYSICS)
