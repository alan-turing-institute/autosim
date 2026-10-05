"""Non-rotating shallow water on a flat, periodic one-dimensional domain."""

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


def simulate_shallow_water_1d(
    initial_state: torch.Tensor,
    *,
    g: float = 1.0,
    L: float = 1.0,
    T: float = 1.0,
    dt_save: float = 0.01,
    dt_max: float = 0.005,
    cfl: float = 0.4,
) -> torch.Tensor:
    r"""Evolve SWE from explicit ``[h, u]`` fields with no forcing or rotation.

    Args:
        initial_state: Float32/float64 fields ``(..., nx, 2)`` in ``[h, u]``
            order. Values represent cell averages on centers ``(j+0.5)*L/nx``.
            Depth must be strictly positive. Leading batch axes are supported.
        g: Positive gravity coefficient shared by all supplied fields.
        L: Periodic domain length; the bed is flat.
        T: Forecast duration, including a saved state at time zero.
        dt_save: Saved-frame interval, independent of the integration step.
        dt_max: Maximum internal step.
        cfl: Wave/advection timestep factor, at most 0.5.

    Returns:
        Physical fields ``(..., nt, nx, 2)`` in ``[h, u]`` order, including
        the initial state. Precision/device and state gradients are retained.

    Notes:
        Internally evolves conserved depth and momentum ``[h, h*u]`` using
        MUSCL reconstruction with minmod slopes, Rusanov fluxes and SSP-RK2.
        Positivity is checked at every stage; dry states raise rather than
        being clipped. No stochastic forcing, explicit viscosity or drag is
        present; finite-volume fluxes introduce numerical dissipation.
    """
    check_state(initial_state, 2)
    check_solver_settings(L, cfl, dt_max)
    if not math.isfinite(g) or g <= 0:
        msg = "g must be finite and positive"
        raise ValueError(msg)
    if not bool((initial_state[..., 0] > 0).all()):
        msg = "SWE depth must be strictly positive"
        raise ValueError(msg)
    times = save_times(T, dt_save)
    dx = L / initial_state.shape[-2]

    def primitive(q: torch.Tensor) -> torch.Tensor:
        if not bool(torch.isfinite(q).all()) or not bool((q[..., 0] > 0).all()):
            msg = "SWE evolution requires finite, positive depth; refine the timestep"
            raise RuntimeError(msg)
        return torch.stack((q[..., 0], q[..., 1] / q[..., 0]), dim=-1)

    def conserved(state: torch.Tensor) -> torch.Tensor:
        return torch.stack((state[..., 0], state[..., 0] * state[..., 1]), dim=-1)

    def flux(state: torch.Tensor) -> torch.Tensor:
        h, u = state.unbind(dim=-1)
        return torch.stack((h * u, h * u.square() + 0.5 * g * h.square()), dim=-1)

    def rhs(q: torch.Tensor) -> torch.Tensor:
        state = primitive(q)
        backward = state - torch.roll(state, 1, dims=-2)
        forward = torch.roll(state, -1, dims=-2) - state
        slope = torch.where(
            backward * forward > 0,
            backward.sign() * torch.minimum(backward.abs(), forward.abs()),
            torch.zeros_like(state),
        )
        left = state + 0.5 * slope
        right = torch.roll(state - 0.5 * slope, -1, dims=-2)
        speed = torch.maximum(
            left[..., 1].abs() + torch.sqrt(g * left[..., 0]),
            right[..., 1].abs() + torch.sqrt(g * right[..., 0]),
        )
        interface_flux = 0.5 * (flux(left) + flux(right)) - 0.5 * speed[..., None] * (
            conserved(right) - conserved(left)
        )
        return -(interface_flux - torch.roll(interface_flux, 1, dims=-2)) / dx

    q = conserved(initial_state)
    snapshots = [initial_state.clone()]
    t = 0.0
    for target_tensor in times[1:]:
        target = float(target_tensor)
        while t < target:
            state = primitive(q)
            # Separate maxima also bound speeds of reconstructed interface states.
            speed = float(state[..., 1].detach().abs().max()) + math.sqrt(
                g * float(state[..., 0].detach().max())
            )
            dt = min(dt_max, cfl * dx / speed, target - t)
            stage = q + dt * rhs(q)
            q = 0.5 * q + 0.5 * (stage + dt * rhs(stage))
            primitive(q)
            t += dt
            if math.isclose(t, target, rel_tol=0, abs_tol=1e-14):
                t = target
        snapshots.append(primitive(q))
    return torch.stack(snapshots, dim=-3)


class ShallowWater1D(Deterministic1DSimulator):
    """Deterministic shallow-water trajectories with physical ``[h, u]`` outputs.

    The sampler places a Gaussian height bump over resting water, following
    the perturbation example in Clawpack:
    https://www.clawpack.org/gallery/pyclaw/gallery/dam_break.html.
    Gaussian images make the bump periodic. Amplitude, width and position are
    sampled independently; these choices live in ``initial_condition_kwargs``.
    Only gravity is a sampled simulator parameter. Neither Coriolis nor
    stochastic forcing is included.

    ``rollout`` takes explicit fields. Legacy scalar ``forward`` evolves a
    supplied ``initial_state`` or the reproducible seed-zero sampled field.
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
        g: float = 1.0,
        dt_max: float = 0.005,
        cfl: float = 0.4,
        initial_condition_kwargs: dict | None = None,
        initial_state: torch.Tensor | None = None,
    ) -> None:
        """Configure the Gaussian-bump sampler, gravity and finite-volume solver."""
        names = ["h", "u"] if output_names is None else output_names
        if len(names) != 2:
            msg = "ShallowWater1D requires two output names in [h, u] order"
            raise ValueError(msg)
        check_solver_settings(L, cfl, dt_max)
        super().__init__(
            parameter_name="g",
            parameter_value=g,
            parameters_range=parameters_range,
            output_names=names,
            nx=nx,
            L=L,
            T=T,
            dt_save=dt_save,
            return_timeseries=return_timeseries,
            initial_state=initial_state,
            grid_offset=0.5,
            log_level=log_level,
        )
        if self.initial_state is not None and not bool(
            (self.initial_state[..., 0] > 0).all()
        ):
            msg = "initial_state depth must be strictly positive"
            raise ValueError(msg)
        self.g, self.dt_max, self.cfl = g, dt_max, cfl
        defaults = {
            "h_mean": 1.0,
            "amplitude_range": (0.05, 0.2),
            "width_fraction_range": (0.05, 0.15),
            "position_fraction_range": (0.0, 1.0),
        }
        settings = dict(initial_condition_kwargs or {})
        if set(settings) - set(defaults):
            msg = "Unsupported initial_condition_kwargs"
            raise ValueError(msg)
        self.initial_condition_kwargs = defaults | settings
        settings = self.initial_condition_kwargs
        if not math.isfinite(settings["h_mean"]) or settings["h_mean"] <= 0:
            msg = "h_mean must be finite and positive"
            raise ValueError(msg)
        check_range(settings["amplitude_range"], "amplitude_range", positive=True)
        check_range(
            settings["width_fraction_range"], "width_fraction_range", positive=True
        )
        check_range(settings["position_fraction_range"], "position_fraction_range")
        if settings["width_fraction_range"][1] > 0.25:
            msg = "width_fraction_range must be at most 0.25"
            raise ValueError(msg)
        if (
            not 0
            <= settings["position_fraction_range"][0]
            <= settings["position_fraction_range"][1]
            <= 1
        ):
            msg = "position_fraction_range must lie in [0, 1]"
            raise ValueError(msg)

    def sample_initial_conditions(
        self, n: int, random_seed: int | None = None
    ) -> torch.Tensor:
        """Sample periodic Gaussian height bumps with zero initial velocity."""
        fixed = self._fixed_initial_states(n)
        if fixed is not None:
            return fixed
        settings = self.initial_condition_kwargs
        rng = generator(random_seed)
        amplitude = torch.empty((n, 1), dtype=torch.float64).uniform_(
            *settings["amplitude_range"], generator=rng
        )
        width = self.L * torch.empty((n, 1), dtype=torch.float64).uniform_(
            *settings["width_fraction_range"], generator=rng
        )
        position = self.L * torch.empty((n, 1), dtype=torch.float64).uniform_(
            *settings["position_fraction_range"], generator=rng
        )
        distance = (
            torch.remainder(self.x - position + 0.5 * self.L, self.L) - 0.5 * self.L
        )
        # Five images give negligible omitted tails for widths <= L/4.
        bump = sum(
            torch.exp(-0.5 * ((distance + image * self.L) / width).square())
            for image in range(-2, 3)
        )
        h = settings["h_mean"] + amplitude * bump
        return torch.stack((h, torch.zeros_like(h)), dim=-1)

    def rollout(
        self, initial_state: torch.Tensor, *, g: float | None = None
    ) -> torch.Tensor:
        """Evolve an explicit ``(..., nx, 2)`` field in physical [h, u] order."""
        check_state(initial_state, 2)
        if initial_state.shape[-2] != self.nx:
            msg = "initial_state spatial size must match nx"
            raise ValueError(msg)
        return simulate_shallow_water_1d(
            initial_state,
            g=self._resolve_parameter(g),
            L=self.L,
            T=self.T,
            dt_save=self.dt_save,
            dt_max=self.dt_max,
            cfl=self.cfl,
        )

    def _rollout(self, state: torch.Tensor, parameter: float) -> torch.Tensor:
        return self.rollout(state, g=parameter)
