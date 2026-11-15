r"""Time integration of a batch of shallow water simulations on the sphere.

One solver step applies, in this order:

1. the Adams-Bashforth 3 step of the conservative dynamics, with the linear damping
   (hyperdiffusion, drag, relaxation) as an integrating factor;
2. the forcing increment, added to the state by operator splitting.

The forcing increment is added after the damping of its own step, so it is damped
from the next step on, with the rest of the state. It never enters the history of the
Adams-Bashforth scheme: inside the history the weights 23/12, -16/12 and 5/12 would
turn a random increment into a different random process. The tendencies of the
history are those of the states that the simulation went through, forcing included.

**Energy budget.** The energy of ``dynamics.total_energy`` is evaluated between these
operations, so every term is the exact energy difference across one operation and
the terms add up to the change of the total energy. The state before the damping is
recovered from the state after it, which leaves the arithmetic of the time step
unchanged. The columns of the budget are ``ENERGY_BUDGET_NAMES``.
"""

from __future__ import annotations

import torch

from autosim.types import TensorLike

from .diagnostics import global_integral
from .dynamics import (
    SphereShallowWaterDynamics,
    adams_bashforth_step,
    increment_to_fields,
    spectral_to_fields,
    step_damping_factors,
    total_energy,
)
from .forcing import (
    ForcingSettings,
    advance_pattern,
    autoregressive_parameter,
    complex_noise,
    dissipation_taper,
    global_dissipation_rate,
    kinetic_forcing_increment,
    local_dissipation_rate,
    mass_pulse_increment,
    no_forcing,
    sample_pulse_events,
    stationary_pattern,
    unit_rate_amplitudes,
)

# Columns of the energy budget of one saved interval. The names of the planar
# simulator are kept where the quantity is the same.
ENERGY_BUDGET_NAMES = (
    "total_energy",  # at the start of the interval [m^3/s^2]
    "deterministic_energy_change",  # conservative step: its energy error
    "hyperviscous_energy_change",  # hyperdiffusion
    "drag_energy_change",  # linear drag
    "relaxation_energy_change",  # Newtonian relaxation of the height
    "forcing_energy_change",  # work of the forcing increments
    "effective_forcing_energy_rate",  # mean rate of the kinetic forcing [m^2/s^3]
)


def backscatter_energy_rate(
    dynamics: SphereShallowWaterDynamics,
    forcing: ForcingSettings,
    undamped_state: TensorLike,
    hyperdiffusion: TensorLike,
    drag: TensorLike | None,
    taper: TensorLike,
) -> TensorLike:
    """Energy rate of the kinetic forcing with the backscatter of the dissipation.

    rate = base rate + backscatter fraction x dissipation of kinetic energy in this
    step. The dissipation is that of the hyperdiffusion, plus that of the drag when
    ``forcing.backscatter_include_drag`` is set. In "global" mode it is its global
    mean, and the rate is one value per simulation. In "local" mode it is a smoothed
    field, and the rate is a field (B09 eq. 13).

    Args:
        dynamics: Fixed operators and time step.
        forcing: Settings of the forcing.
        undamped_state: State before the damping of this step, (B, 3, L, L).
        hyperdiffusion: Factor of the hyperdiffusion for one step, (L, L).
        drag: Factor of the drag for one step, (B,), or None without drag.
        taper: Spectral filter of the local dissipation, (L, 1).

    Returns:
        Energy rate [m^2/s^3], shape (B,) or (B, *grid.shape).
    """
    batch = undamped_state.shape[0]
    device = undamped_state.device
    base_rate = forcing.energy_rate.to(device, torch.float64)
    fraction = forcing.backscatter_fraction.to(device, torch.float64)

    # Damping of each degree that counts as dissipation
    lost = hyperdiffusion[:, :1].unsqueeze(0)  # (1, L, 1)
    if forcing.backscatter_include_drag and drag is not None:
        lost = lost * drag.reshape(batch, 1, 1)  # (B, L, 1)

    if forcing.backscatter_mode == "global":
        dissipation = global_dissipation_rate(dynamics, undamped_state, lost[..., 0])
        return base_rate + fraction * dissipation
    dissipation = local_dissipation_rate(dynamics, undamped_state, lost, taper)
    grid_shape = (batch,) + (1,) * (dissipation.ndim - 1)
    return base_rate.reshape(grid_shape) + fraction.reshape(grid_shape) * dissipation


def energy_changes_of_step(
    dynamics: SphereShallowWaterDynamics,
    energy_before: TensorLike,
    undamped_state: TensorLike,
    damped_state: TensorLike,
    forced_state: TensorLike | None,
    hyperdiffusion: TensorLike,
    drag: TensorLike | None,
    has_relaxation: bool,
    surface_geopotential: TensorLike | None,
) -> tuple[TensorLike, TensorLike]:
    """Energy change of each operation of one solver step.

    The energy is evaluated after the conservative step (undamped state), after the
    hyperdiffusion, after the drag, after the relaxation (damped state) and after the
    forcing increment. Each change is the difference across one operation.

    Args:
        dynamics: Fixed operators.
        energy_before: Energy at the start of the step, (B,).
        undamped_state: State after the conservative step, (B, 3, L, L).
        damped_state: State after the whole damping, (B, 3, L, L).
        forced_state: State after the forcing increment, or None without increment.
        hyperdiffusion: Factor of the hyperdiffusion for one step, (L, L).
        drag: Factor of the drag for one step, (B,), or None without drag.
        has_relaxation: Whether the damping relaxes the geopotential.
        surface_geopotential: Coefficients of g h_s, (B, L, L), or None.

    Returns:
        The five changes (deterministic, hyperviscous, drag, relaxation, forcing),
        shape (B, 5), and the energy at the end of the step, (B,).
    """
    batch = undamped_state.shape[0]
    energy_undamped = total_energy(dynamics, undamped_state, surface_geopotential)

    after_hyperdiffusion = undamped_state.clone()
    after_hyperdiffusion[:, 1:] = hyperdiffusion * undamped_state[:, 1:]
    energy_hyperdiffusion = total_energy(
        dynamics, after_hyperdiffusion, surface_geopotential
    )

    energy_drag = energy_hyperdiffusion
    if drag is not None:
        after_drag = after_hyperdiffusion.clone()
        after_drag[:, 1:] = drag.reshape(batch, 1, 1, 1) * after_hyperdiffusion[:, 1:]
        energy_drag = total_energy(dynamics, after_drag, surface_geopotential)

    energy_damped = energy_drag
    if has_relaxation:
        energy_damped = total_energy(dynamics, damped_state, surface_geopotential)

    energy_forced = energy_damped
    if forced_state is not None:
        energy_forced = total_energy(dynamics, forced_state, surface_geopotential)

    changes = torch.stack(
        [
            energy_undamped - energy_before,
            energy_hyperdiffusion - energy_undamped,
            energy_drag - energy_hyperdiffusion,
            energy_damped - energy_drag,
            energy_forced - energy_damped,
        ],
        dim=1,
    )
    return changes, energy_forced


def integrate_trajectories(  # noqa: PLR0912, PLR0915
    dynamics: SphereShallowWaterDynamics,
    state: TensorLike,
    n_saves: int,
    steps_per_save: int,
    skip_nt: int,
    coriolis: TensorLike | None = None,
    surface_geopotential: TensorLike | None = None,
    drag_time: TensorLike | None = None,
    relaxation_time: TensorLike | None = None,
    equilibrium_geopotential: TensorLike | None = None,
    forcing: ForcingSettings | None = None,
    forcing_generators: list[torch.Generator] | None = None,
    return_additional_input_fields: bool = False,
    return_energy_budget: bool = False,
    return_forcing_state: bool = False,
    previous_tendencies: list[TensorLike] | None = None,
    initial_pattern: TensorLike | None = None,
    redraw_pattern_at_first_save: bool = False,
    output_dynamics: SphereShallowWaterDynamics | None = None,
) -> dict:
    """Advance B simulations and save h, u_phi, u_theta at regular times.

    The state is saved at time 0 and after every ``steps_per_save`` steps, ``n_saves``
    times. The first ``skip_nt`` of these snapshots are dropped.

    The first saved frame is the branch point of reference ensembles: its spectral
    state, the history of the time stepping and the forcing pattern are returned, so
    that other members can continue from it exactly as this run does.

    Args:
        dynamics: Fixed operators and time step.
        state: Initial spectral state (B, 3, L, L), float64 coefficients.
        n_saves: Number of save intervals, T / dt_save.
        steps_per_save: Solver steps per save interval.
        skip_nt: Number of leading snapshots dropped (spin-up).
        coriolis: Coriolis parameter (B, *grid.shape); None is the polar one.
        surface_geopotential: Coefficients of g h_s, (B, L, L); None is a flat bottom.
        drag_time: Time of the linear drag on vorticity and divergence [s], (B,).
            None is no drag.
        relaxation_time: Time of the relaxation of the geopotential [s], (B,). None
            is no relaxation.
        equilibrium_geopotential: Coefficients of the geopotential towards which the
            state relaxes, (B, L, L). Required with ``relaxation_time``.
        forcing: Settings of the stochastic forcing; None is no forcing.
        forcing_generators: One random generator per simulation, for the forcing.
        return_additional_input_fields: Return the forcing increments summed over
            each saved interval.
        return_energy_budget: Return the energy budget of each saved interval.
        return_forcing_state: Return the forcing pattern at each saved time.
        previous_tendencies: Tendencies of the steps before the initial state, most
            recent first, to continue a run without restarting the Adams-Bashforth
            scheme. None starts it with an Euler step.
        initial_pattern: Forcing pattern at the initial time, (B, P, L, L). None
            draws it from its stationary distribution.
        redraw_pattern_at_first_save: Draw a new forcing pattern from its stationary
            distribution at the first saved frame. The forecast from that frame then
            does not depend on the earlier forcing.
        output_dynamics: Operators of the grid of the saved fields. With a lower band
            limit than ``dynamics`` the state is truncated before it is saved
            (coarse-graining). None saves on the grid of ``dynamics``.

    Returns:
        A dictionary with, n_frames = n_saves + 1 - skip_nt, and the grid that of
        ``output_dynamics``:

        - ``"data"``: the saved fields (B, n_frames, *grid.shape, 3), float32 on the
          CPU;
        - ``"final_state"``: the spectral state at the last time, (B, 3, L, L);
        - ``"branch_state"``, ``"branch_tendencies"``, ``"branch_pattern"``: the
          spectral state (B, 3, L, L), the list of previous tendencies and the
          forcing pattern (B, P, L, L) at the first saved frame, the pattern before
          it is redrawn;
        - ``"additional_input_fields"`` (on request): at frame i, the increments of
          h, u_phi, u_theta added by the forcing between frames i and i + 1, zero at
          the last frame, (B, n_frames, *grid.shape, 3), float32 on the CPU;
        - ``"energy_budget"`` (on request): at frame i, the energy at frame i and the
          terms of ``ENERGY_BUDGET_NAMES`` between frames i and i + 1, zero at the
          last frame, (B, n_frames, 7), float64 on the CPU;
        - ``"forcing_state"`` (on request): the wind tendency of the forcing pattern
          for a unit energy rate at each frame, eastward and colatitude components,
          (B, n_frames, *grid.shape, 2), float32 on the CPU.
    """
    batch = state.shape[0]
    device = state.device
    lmax = dynamics.lmax
    dt = dynamics.dt
    hyperdiffusion = dynamics.step_damping[1]  # (L, L), read now: tests may change it

    # --- Grid of the saved fields. Coefficients are truncated to its band limit by
    # keeping the degrees and orders below it: [..., :output_lmax, :output_lmax].
    if output_dynamics is None:
        output_dynamics = dynamics
    output_lmax = output_dynamics.lmax
    output_surface = None
    if surface_geopotential is not None:
        output_surface = surface_geopotential[..., :output_lmax, :output_lmax]

    # --- Linear damping of one step, per simulation when there is drag or relaxation
    step_damping = dynamics.step_damping
    damping_argument = None  # None keeps the arithmetic of the plain time step
    equilibrium = None
    drag = None
    has_relaxation = False
    if drag_time is not None or relaxation_time is not None:
        no_damping_time = torch.full((batch,), torch.inf, dtype=torch.float64)
        if drag_time is None:
            drag_time = no_damping_time
        if relaxation_time is None:
            relaxation_time = no_damping_time
        step_damping = step_damping_factors(dynamics, drag_time, relaxation_time)
        damping_argument = step_damping
        drag = torch.exp(-dt / drag_time.to(device, torch.float64))
        has_relaxation = bool(torch.isfinite(relaxation_time).any())
        if has_relaxation:
            if equilibrium_geopotential is None:
                msg = "relaxation_time requires equilibrium_geopotential"
                raise ValueError(msg)
            equilibrium = torch.zeros_like(state)
            equilibrium[:, 0] = equilibrium_geopotential

    # --- Forcing: patterns of the kinetic-energy forcing, or events of mass pulses
    if forcing is None:
        forcing = no_forcing(batch)
    kinetic_forcing = forcing.forcing_type in ("vortical", "momentum")
    if forcing.forcing_type != "none" and forcing_generators is None:
        msg = "forcing requires forcing_generators"
        raise ValueError(msg)
    if forcing_generators is None:
        forcing_generators = []
    if return_forcing_state and not kinetic_forcing:
        msg = "return_forcing_state requires forcing_type 'vortical' or 'momentum'"
        raise ValueError(msg)

    # Without kinetic forcing the patterns are empty (no pattern, P = 0)
    alpha = torch.ones(batch, dtype=torch.float64, device=device)
    amplitudes = torch.zeros(batch, 0, lmax, dtype=torch.float64, device=device)
    pattern = torch.zeros(batch, 0, lmax, lmax, dtype=torch.complex128, device=device)
    base_rate = forcing.energy_rate.to(device, torch.float64)
    backscatter = False
    taper = dissipation_taper(lmax, device)
    unit_rate = torch.ones(batch, dtype=torch.float64, device=device)
    if kinetic_forcing:
        alpha = autoregressive_parameter(forcing.correlation_time, dt).to(device)
        amplitudes = unit_rate_amplitudes(
            dynamics,
            alpha,
            forcing.spectral_slope,
            forcing.degree_min,
            forcing.degree_max,
        )  # (B, L)
        if forcing.forcing_type == "vortical":
            amplitudes = amplitudes.unsqueeze(1)  # (B, 1, L): streamfunction
        else:
            fraction = forcing.divergent_fraction.to(device, torch.float64)
            fraction = fraction.reshape(batch, 1)
            amplitudes = torch.stack(
                [
                    torch.sqrt(1 - fraction) * amplitudes,  # streamfunction
                    torch.sqrt(fraction) * amplitudes,  # velocity potential
                ],
                dim=1,
            )  # (B, 2, L)
        if initial_pattern is None:
            noise = complex_noise(forcing_generators, amplitudes.shape[1], lmax)
            pattern = stationary_pattern(noise, amplitudes, alpha)
        else:
            pattern = initial_pattern.to(device)
        backscatter = bool((forcing.backscatter_fraction > 0).any())

    pulse_events: dict[int, list[tuple[int, float, float]]] = {}
    if forcing.forcing_type == "mass_pulses":
        pulse_events = sample_pulse_events(
            forcing_generators, forcing.pulse_rate, n_saves * steps_per_save, dt
        )

    # --- Time loop
    tendencies_history: list[TensorLike] = []
    if previous_tendencies is not None:
        tendencies_history = list(previous_tendencies)

    # Branch point: the first saved frame (time 0 when no snapshot is dropped)
    branch_state = state
    branch_tendencies = list(tendencies_history)
    branch_pattern = pattern

    frames = []
    increment_frames = []
    budget_frames = []
    pattern_frames = []
    if skip_nt == 0:
        truncated = state[..., :output_lmax, :output_lmax]
        fields = spectral_to_fields(output_dynamics, truncated, output_surface)
        frames.append(fields.float().cpu())
        if return_forcing_state:
            tendency = kinetic_forcing_increment(dynamics, pattern, unit_rate) / dt
            truncated = tendency[..., :output_lmax, :output_lmax]
            pattern_fields = increment_to_fields(output_dynamics, truncated)[..., 1:]
            pattern_frames.append(pattern_fields.float().cpu())
    energy = torch.zeros(batch, dtype=torch.float64, device=device)
    if return_energy_budget:
        energy = total_energy(dynamics, state, surface_geopotential)

    undamped_state = state  # set at each step when the budget or backscatter need it
    step_index = 0
    for save in range(1, n_saves + 1):
        increments_since_save = torch.zeros_like(state)
        changes_since_save = torch.zeros(batch, 5, dtype=torch.float64, device=device)
        rate_since_save = torch.zeros(batch, dtype=torch.float64, device=device)
        energy_at_save = energy

        for _ in range(steps_per_save):
            # 1. Conservative dynamics and linear damping
            damped_state, tendencies_history = adams_bashforth_step(
                dynamics,
                state,
                tendencies_history,
                coriolis,
                surface_geopotential,
                damping_argument,
                equilibrium,
            )
            # State before the damping, from damped = s_eq + E (undamped - s_eq)
            if return_energy_budget or backscatter:
                if equilibrium is None:
                    undamped_state = damped_state / step_damping
                else:
                    departure = (damped_state - equilibrium) / step_damping
                    undamped_state = equilibrium + departure

            # 2. Forcing increment, added by operator splitting
            increment = None
            if kinetic_forcing:
                energy_rate = base_rate
                if backscatter:
                    energy_rate = backscatter_energy_rate(
                        dynamics, forcing, undamped_state, hyperdiffusion, drag, taper
                    )
                increment = kinetic_forcing_increment(dynamics, pattern, energy_rate)
                noise = complex_noise(forcing_generators, amplitudes.shape[1], lmax)
                pattern = advance_pattern(pattern, noise, amplitudes, alpha)
                if energy_rate.ndim == 1:
                    rate_since_save += energy_rate
                else:
                    rate_since_save += global_integral(
                        energy_rate, dynamics.quad_weights
                    ) / (4 * torch.pi)
            if step_index in pulse_events:
                increment = mass_pulse_increment(
                    dynamics,
                    pulse_events[step_index],
                    batch,
                    forcing.pulse_amplitude,
                    forcing.pulse_width,
                    forcing.pulse_remove_mean,
                )

            forced_state = None
            state = damped_state
            if increment is not None:
                forced_state = damped_state + increment
                state = forced_state
                increments_since_save += increment

            if return_energy_budget:
                changes, energy = energy_changes_of_step(
                    dynamics,
                    energy,
                    undamped_state,
                    damped_state,
                    forced_state,
                    hyperdiffusion,
                    drag,
                    has_relaxation,
                    surface_geopotential,
                )
                changes_since_save += changes
            step_index += 1

        # Branch point at the first saved frame
        if save == skip_nt:
            branch_state = state
            branch_tendencies = list(tendencies_history)
            branch_pattern = pattern
            if kinetic_forcing and redraw_pattern_at_first_save:
                noise = complex_noise(forcing_generators, amplitudes.shape[1], lmax)
                pattern = stationary_pattern(noise, amplitudes, alpha)

        # Outputs of the interval that ends here, attached to the frame at its start
        if save - 1 >= skip_nt:
            if return_additional_input_fields:
                truncated = increments_since_save[..., :output_lmax, :output_lmax]
                fields = increment_to_fields(output_dynamics, truncated)
                increment_frames.append(fields.float().cpu())
            if return_energy_budget:
                row = torch.cat(
                    [
                        energy_at_save.reshape(batch, 1),
                        changes_since_save,
                        (rate_since_save / steps_per_save).reshape(batch, 1),
                    ],
                    dim=1,
                )
                budget_frames.append(row.cpu())
        if save >= skip_nt:
            truncated = state[..., :output_lmax, :output_lmax]
            fields = spectral_to_fields(output_dynamics, truncated, output_surface)
            frames.append(fields.float().cpu())
            if return_forcing_state:
                tendency = kinetic_forcing_increment(dynamics, pattern, unit_rate) / dt
                truncated = tendency[..., :output_lmax, :output_lmax]
                pattern_fields = increment_to_fields(output_dynamics, truncated)
                pattern_frames.append(pattern_fields[..., 1:].float().cpu())

    output = {
        "data": torch.stack(frames, dim=1),
        "final_state": state,
        "branch_state": branch_state,
        "branch_tendencies": branch_tendencies,
        "branch_pattern": branch_pattern,
    }
    if return_additional_input_fields:
        increment_frames.append(torch.zeros_like(frames[-1]))  # last frame
        output["additional_input_fields"] = torch.stack(increment_frames, dim=1)
    if return_energy_budget:
        last_row = torch.zeros(batch, 7, dtype=torch.float64)
        last_row[:, 0] = energy.cpu()
        budget_frames.append(last_row)
        output["energy_budget"] = torch.stack(budget_frames, dim=1)
    if return_forcing_state:
        output["forcing_state"] = torch.stack(pattern_frames, dim=1)
    return output
