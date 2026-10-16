"""Periodic viscous Burgers evolution with explicit initial velocity fields."""

from __future__ import annotations

import math

import torch

from ._deterministic_1d import (
    Deterministic1DSimulator,
    check_range,
    check_solver_settings,
    check_state,
    generator,
    save_times,
)


def simulate_burgers_1d(
    initial_state: torch.Tensor,
    *,
    nu: float = 0.1,
    L: float = 1.0,
    T: float = 1.0,
    dt_save: float = 0.01,
    dt_max: float = 0.005,
    cfl: float = 0.4,
) -> torch.Tensor:
    r"""Evolve ``u_t + (u²/2)_x = nu*u_xx`` with periodic boundaries.

    Args:
        initial_state: Float32/float64 field ``(..., nx, 1)`` on points
            ``x_j = j*L/nx``. Leading batch axes are supported.
        nu: Positive viscosity, shared by the supplied fields.
        L: Periodic domain length.
        T: Forecast duration, including a saved state at time zero.
        dt_save: Saved-frame interval; the terminal time is always included.
        dt_max: Maximum internal integration step, independent of ``dt_save``.
        cfl: Advection timestep safety factor, at most 0.5.

    Returns:
        Physical velocities of shape ``(..., nt, nx, 1)``. Precision, device
        and state gradients are retained, and the input is never mutated.

    Notes:
        Strang splitting solves diffusion exactly in Fourier space around a
        conservative, two-thirds-dealiased RK4 advection step. Time accuracy is
        second order because of splitting. No process noise or clipping is used.
    """
    check_state(initial_state, 1)
    check_solver_settings(L, cfl, dt_max)
    if not math.isfinite(nu) or nu <= 0:
        msg = "nu must be finite and positive"
        raise ValueError(msg)
    times = save_times(T, dt_save)
    nx = initial_state.shape[-2]
    dx = L / nx
    k = (
        2
        * math.pi
        * torch.fft.rfftfreq(
            nx, d=dx, dtype=initial_state.dtype, device=initial_state.device
        )
    )
    mask = torch.arange(len(k), device=k.device) <= (nx - 1) // 3

    def rhs(u: torch.Tensor) -> torch.Tensor:
        resolved = torch.fft.irfft(torch.fft.rfft(u) * mask, n=nx)
        flux_hat = torch.fft.rfft(0.5 * resolved.square())
        return torch.fft.irfft(-1j * k * flux_hat * mask, n=nx)

    u = initial_state[..., 0].clone()
    snapshots = [initial_state.clone()]
    t = 0.0
    for target_tensor in times[1:]:
        target = float(target_tensor)
        while t < target:
            speed = max(float(u.detach().abs().max()), 1e-12)
            dt = min(dt_max, cfl * dx / speed, target - t)
            half_heat = torch.exp(-nu * k.square() * (0.5 * dt))
            u = torch.fft.irfft(torch.fft.rfft(u) * half_heat, n=nx)
            k1 = rhs(u)
            k2 = rhs(u + 0.5 * dt * k1)
            k3 = rhs(u + 0.5 * dt * k2)
            k4 = rhs(u + dt * k3)
            u = u + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            u = torch.fft.irfft(torch.fft.rfft(u) * half_heat, n=nx)
            if not bool(torch.isfinite(u).all()):
                msg = "Burgers evolution became non-finite; refine the timestep"
                raise RuntimeError(msg)
            t += dt
            if math.isclose(t, target, rel_tol=0, abs_tol=1e-14):
                t = target
        snapshots.append(u[..., None])
    return torch.stack(snapshots, dim=-3)


class Burgers1D(Deterministic1DSimulator):
    r"""Deterministic Burgers trajectories conditioned on a velocity field.

    The default sampler follows the periodic FNO Burgers covariance
    ``sigma²*(-Delta + tau² I)^(-alpha)``, with ``sigma=25``, ``tau=5`` and
    ``alpha=2`` on the unit domain. ``zero_mean=True`` projects out the constant
    mode. There is no sample-wise amplitude normalization. See Li et al.,
    https://arxiv.org/html/2010.08895v3#A3.SS1.

    ``initial_condition`` may also be ``"sine"`` or ``"fourier"``. Their
    ``initial_condition_kwargs`` configure ``amplitude_range`` and
    ``phase_range`` (sine), or ``n_modes``, ``spectral_decay`` and
    ``peak_amplitude_range`` (custom Fourier fields). Dataset sampling and
    physical ``parameters_range={"nu": (...)}`` are independent.

    Use :meth:`rollout` to evolve a supplied field. Legacy scalar-only
    ``forward`` uses ``initial_state`` or the reproducible seed-zero field.
    """

    def __init__(
        self,
        parameters_range: dict[str, tuple[float, float]] | None = None,
        output_names: list[str] | None = None,
        return_timeseries: bool = True,
        log_level: str = "progress_bar",
        nx: int = 256,
        L: float = 1.0,
        T: float = 1.0,
        dt_save: float = 0.01,
        nu: float = 0.1,
        dt_max: float = 0.005,
        cfl: float = 0.4,
        initial_condition: str = "gaussian_random_field",
        initial_condition_kwargs: dict | None = None,
        initial_state: torch.Tensor | None = None,
    ) -> None:
        """Configure physical parameters, initial-field sampling and numerics."""
        names = ["u"] if output_names is None else output_names
        if len(names) != 1:
            msg = "Burgers1D requires one output name"
            raise ValueError(msg)
        check_solver_settings(L, cfl, dt_max)
        super().__init__(
            parameter_name="nu",
            parameter_value=nu,
            parameters_range=parameters_range,
            output_names=names,
            nx=nx,
            L=L,
            T=T,
            dt_save=dt_save,
            return_timeseries=return_timeseries,
            initial_state=initial_state,
            grid_offset=0.0,
            log_level=log_level,
        )
        self.nu, self.dt_max, self.cfl = nu, dt_max, cfl
        defaults = {
            "gaussian_random_field": {
                "alpha": 2.0,
                "tau": 5.0,
                "sigma": 25.0,
                "zero_mean": True,
            },
            "sine": {"amplitude_range": (0.5, 1.0), "phase_range": (0.0, 2 * math.pi)},
            "fourier": {
                "n_modes": 4,
                "spectral_decay": 2.0,
                "peak_amplitude_range": (0.5, 1.0),
            },
        }
        if initial_condition not in defaults:
            msg = "initial_condition must be gaussian_random_field, sine or fourier"
            raise ValueError(msg)
        settings = dict(initial_condition_kwargs or {})
        if set(settings) - set(defaults[initial_condition]):
            msg = "Unsupported initial_condition_kwargs"
            raise ValueError(msg)
        self.initial_condition = initial_condition
        self.initial_condition_kwargs = defaults[initial_condition] | settings
        settings = self.initial_condition_kwargs
        if initial_condition == "gaussian_random_field":
            if not all(
                math.isfinite(settings[key]) and settings[key] > 0
                for key in ("alpha", "tau", "sigma")
            ):
                msg = "GRF alpha, tau and sigma must be finite and positive"
                raise ValueError(msg)
            if not isinstance(settings["zero_mean"], bool):
                msg = "zero_mean must be a boolean"
                raise TypeError(msg)
        elif initial_condition == "sine":
            check_range(settings["amplitude_range"], "amplitude_range", positive=True)
            check_range(settings["phase_range"], "phase_range")
        else:
            if (
                not isinstance(settings["n_modes"], int)
                or not 1 <= settings["n_modes"] < nx / 2
            ):
                msg = "n_modes must be a positive integer below nx/2"
                raise ValueError(msg)
            if (
                not math.isfinite(settings["spectral_decay"])
                or settings["spectral_decay"] < 0
            ):
                msg = "spectral_decay must be finite and non-negative"
                raise ValueError(msg)
            check_range(
                settings["peak_amplitude_range"], "peak_amplitude_range", positive=True
            )

    def sample_initial_conditions(
        self, n: int, random_seed: int | None = None
    ) -> torch.Tensor:
        """Sample nodal velocity fields, independently of viscosity."""
        fixed = self._fixed_initial_states(n)
        if fixed is not None:
            return fixed
        if n == 0:
            return torch.empty((0, self.nx, 1), dtype=torch.float64)
        rng = generator(random_seed)
        settings = self.initial_condition_kwargs
        if self.initial_condition == "gaussian_random_field":
            white = torch.randn((n, self.nx), generator=rng, dtype=torch.float64)
            k = (
                2
                * math.pi
                * torch.fft.rfftfreq(self.nx, d=self.L / self.nx, dtype=torch.float64)
            )
            spectrum = settings["sigma"] * (k.square() + settings["tau"] ** 2).pow(
                -settings["alpha"] / 2
            )
            if settings["zero_mean"]:
                spectrum[0] = 0
            # Remove the even-grid Nyquist mode; other modes have paired sines/cosines.
            if self.nx % 2 == 0:
                spectrum[-1] = 0
            field = torch.fft.irfft(
                torch.fft.rfft(white, norm="ortho") * spectrum,
                n=self.nx,
                norm="ortho",
            ) * math.sqrt(self.nx / self.L)
        elif self.initial_condition == "sine":
            amplitude = torch.empty((n, 1), dtype=torch.float64).uniform_(
                *settings["amplitude_range"], generator=rng
            )
            phase = torch.empty((n, 1), dtype=torch.float64).uniform_(
                *settings["phase_range"], generator=rng
            )
            field = amplitude * torch.sin(2 * math.pi * self.x / self.L + phase)
        else:
            modes = torch.arange(1, settings["n_modes"] + 1, dtype=torch.float64)
            phases = 2 * math.pi * modes[:, None] * self.x[None, :] / self.L
            coefficients = torch.randn(
                (n, 2, len(modes)), generator=rng, dtype=torch.float64
            )
            weights = modes.pow(-settings["spectral_decay"])
            field = (coefficients[:, 0] * weights) @ phases.cos() + (
                coefficients[:, 1] * weights
            ) @ phases.sin()
            amplitude = torch.empty((n, 1), dtype=torch.float64).uniform_(
                *settings["peak_amplitude_range"], generator=rng
            )
            field = amplitude * field / field.abs().amax(dim=-1, keepdim=True)
        return field[..., None]

    def rollout(
        self, initial_state: torch.Tensor, *, nu: float | None = None
    ) -> torch.Tensor:
        """Evolve an explicit ``(..., nx, 1)`` field without sampling randomness."""
        check_state(initial_state, 1)
        if initial_state.shape[-2] != self.nx:
            msg = "initial_state spatial size must match nx"
            raise ValueError(msg)
        return simulate_burgers_1d(
            initial_state,
            nu=self._resolve_parameter(nu),
            L=self.L,
            T=self.T,
            dt_save=self.dt_save,
            dt_max=self.dt_max,
            cfl=self.cfl,
        )

    def _rollout(self, state: torch.Tensor, parameter: float) -> torch.Tensor:
        return self.rollout(state, nu=parameter)
