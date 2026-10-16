"""Deterministic periodic Kuramoto-Sivashinsky evolution and chaos diagnostics."""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import lru_cache

import torch
from einops import rearrange

from ._deterministic_1d import (
    Deterministic1DSimulator,
    check_range,
    check_state,
    generator,
    save_times,
)


def _check_settings(nu: float, L: float, dt_max: float) -> None:
    """Validate geometry, fourth-order dissipation and timestep."""
    if not all(math.isfinite(value) and value > 0 for value in (nu, L, dt_max)):
        msg = "nu, L and dt_max must be finite and positive"
        raise ValueError(msg)


def _etdrk4_coefficients(linear: torch.Tensor, dt: float) -> tuple[torch.Tensor, ...]:
    """Evaluate real ETDRK4 coefficients using Kassam-Trefethen contours.

    Contour means avoid cancellation at zero and small linear eigenvalues.
    ``f2`` is multiplied by ``2*(Na+Nb)`` in the step, including at zero.
    """
    angles = (
        math.pi
        * (torch.arange(32, dtype=linear.dtype, device=linear.device) + 0.5)
        / 32
    )
    roots = torch.polar(torch.ones_like(angles), angles)
    z = dt * linear[:, None] + roots
    exp_z = torch.exp(z)
    Q = dt * ((torch.exp(z / 2) - 1) / z).mean(dim=-1).real
    f1 = dt * ((-4 - z + exp_z * (4 - 3 * z + z.square())) / z.pow(3)).mean(dim=-1).real
    f2 = dt * ((2 + z + exp_z * (-2 + z)) / z.pow(3)).mean(dim=-1).real
    f3 = dt * ((-4 - 3 * z - z.square() + exp_z * (4 - z)) / z.pow(3)).mean(dim=-1).real
    return torch.exp(dt * linear), torch.exp(0.5 * dt * linear), Q, f1, f2, f3


def _spectral_advance(
    state: torch.Tensor, nu: float, L: float, dt_max: float
) -> Callable[[torch.Tensor, float], torch.Tensor]:
    """Build a spectral propagator, reusing operators and timestep coefficients."""
    nx = state.shape[-2]
    k = (
        2
        * math.pi
        * torch.fft.rfftfreq(nx, d=L / nx, dtype=state.dtype, device=state.device)
    )
    linear = k.square() - nu * k.pow(4)
    mask = torch.arange(len(k), device=k.device) <= (nx - 1) // 3

    def nonlinear(v: torch.Tensor) -> torch.Tensor:
        resolved = torch.fft.irfft(v * mask, n=nx)
        return -0.5j * k * torch.fft.rfft(resolved.square()) * mask

    @lru_cache(maxsize=8)
    def coefficients(dt: float) -> tuple[torch.Tensor, ...]:
        return _etdrk4_coefficients(linear, dt)

    def advance(v: torch.Tensor, duration: float) -> torch.Tensor:
        # Use an integer count to avoid accumulating time-rounding errors.
        for index in range(math.ceil(duration / dt_max)):
            dt = min(dt_max, duration - index * dt_max)
            E, E2, Q, f1, f2, f3 = coefficients(dt)
            Nv = nonlinear(v)
            a = E2 * v + Q * Nv
            Na = nonlinear(a)
            b = E2 * v + Q * Na
            Nb = nonlinear(b)
            c = E2 * a + Q * (2 * Nb - Nv)
            Nc = nonlinear(c)
            v = E * v + f1 * Nv + 2 * f2 * (Na + Nb) + f3 * Nc
            if not bool(torch.isfinite(v).all()):
                msg = "KS evolution became non-finite; refine timestep and grid"
                raise RuntimeError(msg)
        return v

    return advance


def simulate_kuramoto_sivashinsky_1d(
    initial_state: torch.Tensor,
    *,
    nu: float = 1.0,
    L: float = 22.0,
    T: float = 100.0,
    dt_save: float = 0.5,
    dt_max: float = 0.1,
) -> torch.Tensor:
    r"""Evolve ``u_t + (u²/2)_x + u_xx + nu*u_xxxx = 0`` periodically.

    Args:
        initial_state: Float32/float64 nodal fields ``(..., nx, 1)`` at
            ``x_j=j*L/nx``. Leading batch axes are supported.
        nu: Positive fourth-order dissipation coefficient, shared by fields.
        L: Periodic domain length. The standard chaotic benchmark uses 22.
        T: Forecast duration, including the supplied state at time zero.
        dt_save: Saved-frame interval; the terminal time is always included.
        dt_max: Maximum internal step. Refine it to check time convergence.

    Returns:
        Fields ``(..., nt, nx, 1)`` preserving input precision, device and
        state gradients. The supplied field is never mutated.

    Notes:
        Uses Fourier differentiation, two-thirds-dealiased conservative
        advection and fourth-order ETDRK4 with contour-evaluated coefficients.
        The linear multiplier is ``k² - nu*k⁴``: long wavelengths grow while
        short wavelengths decay. The spatial mean is conserved. Modes above
        the dealiasing cutoff evolve linearly. There is no noise or forcing.
        The stiff linear term is integrated exponentially; ``dt_max`` still
        needs to resolve nonlinear evolution. No warm-up is applied here.
    """
    check_state(initial_state, 1)
    _check_settings(nu, L, dt_max)
    times = save_times(T, dt_save)
    nx = initial_state.shape[-2]
    advance = _spectral_advance(initial_state, nu, L, dt_max)
    v = torch.fft.rfft(initial_state[..., 0])
    snapshots = [initial_state.clone()]
    t = 0.0
    for target_tensor in times[1:]:
        target = float(target_tensor)
        v = advance(v, target - t)
        t = target
        snapshots.append(torch.fft.irfft(v, n=nx)[..., None])
    return torch.stack(snapshots, dim=-3)


@torch.no_grad()
def estimate_ks_lyapunov(
    initial_state: torch.Tensor,
    *,
    nu: float = 1.0,
    L: float = 22.0,
    T: float = 200.0,
    warmup_time: float = 100.0,
    dt_max: float = 0.1,
    renormalize_interval: float = 1.0,
    perturbation: float = 1e-7,
    random_seed: int = 0,
) -> torch.Tensor:
    """Estimate the largest Lyapunov exponent using renormalized trajectories.

    Args:
        initial_state: Float64 field ``(..., nx, 1)``. Perturbations have zero
            spatial mean so both trajectories conserve the same mean.
        nu: Fourth-order dissipation coefficient.
        L: Domain length.
        T: Positive duration over which logarithmic growth is accumulated.
        warmup_time: Non-negative deterministic warm-up before perturbing.
        dt_max: Maximum internal ETDRK4 step.
        renormalize_interval: Positive time between perturbation resets.
        perturbation: Positive initial/reset RMS distance between trajectories.
        random_seed: Local seed for the initial perturbation direction.

    Returns:
        A float64 tensor with the input's leading batch shape, in inverse
        simulation time units. An unbatched field gives a scalar tensor.

    Notes:
        This is a finite-time estimate, not a proof of asymptotic chaos.
        Check longer windows, multiple seeds, perturbation sizes and numerical
        refinement. Positive growth at an unstable equilibrium alone is not
        evidence of a chaotic attractor. Float64 is required to resolve the
        small separations. No stochastic forcing is added to either rollout.
    """
    check_state(initial_state, 1)
    _check_settings(nu, L, dt_max)
    if initial_state.dtype != torch.float64:
        msg = "Lyapunov estimation requires float64 fields"
        raise TypeError(msg)
    if not math.isfinite(T) or T <= 0:
        msg = "Lyapunov estimation T must be finite and positive"
        raise ValueError(msg)
    if not math.isfinite(warmup_time) or warmup_time < 0:
        msg = "warmup_time must be finite and non-negative"
        raise ValueError(msg)
    if not math.isfinite(perturbation) or perturbation <= 0:
        msg = "perturbation must be finite and positive"
        raise ValueError(msg)
    intervals = save_times(T, renormalize_interval)
    base = simulate_kuramoto_sivashinsky_1d(
        initial_state,
        nu=nu,
        L=L,
        T=warmup_time,
        dt_save=max(warmup_time, 1.0),
        dt_max=dt_max,
    )[..., -1, :, :]
    direction = torch.randn(
        base.shape, dtype=torch.float64, generator=generator(random_seed)
    ).to(base.device)
    direction -= direction.mean(dim=-2, keepdim=True)
    direction /= direction.square().mean(dim=(-2, -1), keepdim=True).sqrt()
    perturbed = base + perturbation * direction
    total = torch.zeros_like(base[..., :1, :1])
    advance = _spectral_advance(base, nu, L, dt_max)
    for duration in torch.diff(intervals):
        spectrum = advance(
            torch.fft.rfft(torch.stack((base, perturbed))[..., 0]), float(duration)
        )
        pair = torch.fft.irfft(spectrum, n=base.shape[-2])[..., None]
        base, perturbed = pair.unbind(dim=0)
        delta = perturbed - base
        delta -= delta.mean(dim=-2, keepdim=True)
        distance = delta.square().mean(dim=(-2, -1), keepdim=True).sqrt()
        if not bool((distance > 0).all()):
            msg = "Perturbation vanished; increase its size or numerical precision"
            raise RuntimeError(msg)
        total += torch.log(distance / perturbation)
        perturbed = base + perturbation * delta / distance
    return rearrange(total / T, "... 1 1 -> ...")


class KuramotoSivashinsky1D(Deterministic1DSimulator):
    """KS datasets with explicit initial fields and optional chaos screening.

    ``sample_initial_conditions`` draws generic zero-mean Fourier fields,
    independently of ``parameters_range={"nu": (...)}``. ``rollout`` evolves
    those supplied fields directly. Dataset generation can first apply
    ``warmup_time`` and then a finite-time Lyapunov acceptance criterion.
    The recorded first frame is the actual post-warm-up forecast state; raw
    sampled fields and warm-up durations are retained in payload metadata.

    Standard coefficients ``nu=1`` and ``L=22`` target the chaotic benchmark
    of Cvitanovic, Davidchack and Siminos (2010):
    https://arxiv.org/abs/0709.2944. ``L < 2*pi*sqrt(nu)`` gives a stable control.
    Passing a finite-time criterion does not prove every output is chaotic.
    """

    def __init__(
        self,
        parameters_range: dict[str, tuple[float, float]] | None = None,
        output_names: list[str] | None = None,
        return_timeseries: bool = True,
        log_level: str = "progress_bar",
        nx: int = 128,
        L: float = 22.0,
        T: float = 100.0,
        dt_save: float = 0.5,
        nu: float = 1.0,
        dt_max: float = 0.1,
        initial_condition_kwargs: dict | None = None,
        initial_state: torch.Tensor | None = None,
        warmup_time: float = 0.0,
        chaos_validation_time: float = 0.0,
        lyapunov_threshold: float = 0.01,
        activity_threshold: float = 1e-6,
    ) -> None:
        """Configure Fourier sampling, dynamics, dataset warm-up and screening."""
        names = ["u"] if output_names is None else output_names
        if len(names) != 1:
            msg = "KuramotoSivashinsky1D requires one output name"
            raise ValueError(msg)
        _check_settings(nu, L, dt_max)
        for value in (
            warmup_time,
            chaos_validation_time,
            lyapunov_threshold,
            activity_threshold,
        ):
            if not math.isfinite(value) or value < 0:
                msg = (
                    "Warm-up, validation time and screening thresholds "
                    "must be finite and non-negative"
                )
                raise ValueError(msg)
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
        self.nu, self.dt_max = nu, dt_max
        self.warmup_time = warmup_time
        self.chaos_validation_time = chaos_validation_time
        self.lyapunov_threshold = lyapunov_threshold
        self.activity_threshold = activity_threshold
        defaults = {
            "n_modes": 8,
            "spectral_decay": 1.5,
            "peak_amplitude_range": (0.5, 1.0),
        }
        settings = dict(initial_condition_kwargs or {})
        if set(settings) - set(defaults):
            msg = "Unsupported initial_condition_kwargs"
            raise ValueError(msg)
        self.initial_condition_kwargs = defaults | settings
        settings = self.initial_condition_kwargs
        if (
            not isinstance(settings["n_modes"], int)
            or not 1 <= settings["n_modes"] <= (nx - 1) // 3
        ):
            msg = "n_modes must be a positive integer within the dealiasing cutoff"
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
        """Sample unburned zero-mean nodal fields, independently of nu."""
        fixed = self._fixed_initial_states(n)
        if fixed is not None:
            return fixed
        if n == 0:
            return torch.empty((0, self.nx, 1), dtype=torch.float64)
        settings = self.initial_condition_kwargs
        rng = generator(random_seed)
        modes = torch.arange(1, settings["n_modes"] + 1, dtype=torch.float64)
        phases = 2 * math.pi * modes[:, None] * self.x[None, :] / self.L
        coefficients = torch.randn(
            (n, 2, len(modes)), dtype=torch.float64, generator=rng
        )
        weights = modes.pow(-settings["spectral_decay"])
        field = (coefficients[:, 0] * weights) @ phases.cos() + (
            coefficients[:, 1] * weights
        ) @ phases.sin()
        amplitude = torch.empty((n, 1), dtype=torch.float64).uniform_(
            *settings["peak_amplitude_range"], generator=rng
        )
        return (amplitude * field / field.abs().amax(dim=-1, keepdim=True))[..., None]

    def rollout(
        self, initial_state: torch.Tensor, *, nu: float | None = None
    ) -> torch.Tensor:
        """Evolve supplied fields directly; dataset warm-up/screening is separate."""
        self._check_grid(initial_state)
        return simulate_kuramoto_sivashinsky_1d(
            initial_state,
            nu=self._resolve_parameter(nu),
            L=self.L,
            T=self.T,
            dt_save=self.dt_save,
            dt_max=self.dt_max,
        )

    def warmup_state(
        self, initial_state: torch.Tensor, *, nu: float | None = None
    ) -> torch.Tensor:
        """Return the field after the configured deterministic warm-up."""
        self._check_grid(initial_state)
        return simulate_kuramoto_sivashinsky_1d(
            initial_state,
            nu=self._resolve_parameter(nu),
            L=self.L,
            T=self.warmup_time,
            dt_save=max(self.warmup_time, 1.0),
            dt_max=self.dt_max,
        )[..., -1, :, :]

    def _check_grid(self, state: torch.Tensor) -> None:
        """Validate an explicit field against the configured spatial grid."""
        check_state(state, 1)
        if state.shape[-2] != self.nx:
            msg = "initial_state spatial size must match nx"
            raise ValueError(msg)

    def _prepare_initial_conditions(
        self, states: torch.Tensor, parameters: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """Warm each field using its sampled nu; optionally reject failed criteria."""
        metadata = {
            "seed_fields": states,
            "warmup_times": torch.full(
                (len(states),), self.warmup_time, dtype=torch.float64
            ),
        }
        prepared, estimates, activities = [], [], []
        for index, (state, parameter) in enumerate(
            zip(states, parameters, strict=True)
        ):
            nu = float(parameter[0])
            warmed = self.warmup_state(state, nu=nu)
            if self.chaos_validation_time > 0:
                # An unstable equilibrium can have positive perturbation growth
                # without its own trajectory being chaotic. Reject quiescent
                # forecast states before using the Lyapunov criterion.
                evolved = simulate_kuramoto_sivashinsky_1d(
                    warmed,
                    nu=nu,
                    L=self.L,
                    T=1.0,
                    dt_save=1.0,
                    dt_max=self.dt_max,
                )[-1]
                activity = (evolved - warmed).square().mean().sqrt()
                if float(activity) <= self.activity_threshold:
                    msg = f"KS sample {index} failed the forecast activity criterion"
                    raise RuntimeError(msg)
                estimate = estimate_ks_lyapunov(
                    warmed,
                    nu=nu,
                    L=self.L,
                    T=self.chaos_validation_time,
                    warmup_time=0,
                    dt_max=self.dt_max,
                    random_seed=index,
                )
                if float(estimate) <= self.lyapunov_threshold:
                    msg = (
                        f"KS sample {index} failed the finite-time Lyapunov criterion: "
                        f"{float(estimate):.6g} <= {self.lyapunov_threshold}. "
                        "No sample was silently replaced; "
                        "use another seed or review the regime."
                    )
                    raise RuntimeError(msg)
                estimates.append(estimate)
                activities.append(activity)
            prepared.append(warmed)
        if self.chaos_validation_time > 0:
            metadata["lyapunov_estimates"] = (
                rearrange(torch.stack(estimates), "b -> b 1")
                if estimates
                else torch.empty((0, 1), dtype=torch.float64)
            )
            metadata["activity_estimates"] = (
                rearrange(torch.stack(activities), "b -> b 1")
                if activities
                else torch.empty((0, 1), dtype=torch.float64)
            )
        return torch.stack(prepared) if prepared else states, metadata

    def _rollout(self, state: torch.Tensor, parameter: float) -> torch.Tensor:
        """Evolve a prepared forecast state."""
        return self.rollout(state, nu=parameter)
