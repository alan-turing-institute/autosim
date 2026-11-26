r"""Time integration of a batch of shallow water simulations on the sphere.

One solver step applies, in this order:

1. the Adams-Bashforth 3 step of the conservative dynamics, with the linear damping
   (hyperdiffusion, drag, relaxation) as an integrating factor;
2. the forcing increments, added to the state by operator splitting, in the order:
   SKEBS (``SkebsForcing``), SPPT (``SpptForcing``, the physics increment of step 1
   times a random pattern), The Well's periodic height forcing (``PeriodicForcing``).

The forcing increments are added after the damping of their own step, so they are
damped from the next step on, with the rest of the state. They never enter the
history of the Adams-Bashforth scheme: inside the history the weights 23/12, -16/12
and 5/12 would turn a random increment into a different random process. The
tendencies of the history are those of the states that the simulation went through,
forcing included.

**Energy budget.** The energy of ``dynamics.total_energy`` is evaluated between these
operations, so every term is the exact energy difference across one operation and
the terms add up to the change of the total energy. The state before the damping is
recovered from the state after it, which leaves the arithmetic of the time step
unchanged. The columns of the budget are ``ENERGY_BUDGET_NAMES``.
"""

from __future__ import annotations

import torch

from autosim.types import TensorLike

from .dynamics import (
    SphereShallowWaterDynamics,
    adams_bashforth_step,
    increment_to_fields,
    spectral_to_fields,
    step_damping_factors,
    total_energy,
)
from .forcing import (
    PeriodicForcing,
    SkebsForcing,
    SpptForcing,
    StepState,
)
from .settings import PeriodicForcingSettings, SpptSettings

# Columns of the energy budget of one saved interval. The names of the planar
# simulator are kept where the quantity is the same.
ENERGY_BUDGET_NAMES = (
    "total_energy",  # at the start of the interval [m^3/s^2]
    "deterministic_energy_change",  # conservative step: its energy error
    "hyperviscous_energy_change",  # hyperdiffusion
    "drag_energy_change",  # linear drag
    "relaxation_energy_change",  # Newtonian relaxation of the height
    "forcing_energy_change",  # work of the SKEBS increments
    "sppt_energy_change",  # work of the SPPT increments
    "periodic_energy_change",  # work of The Well's periodic height forcing
    "effective_forcing_energy_rate",  # mean rate of the SKEBS forcing [m^2/s^3]
)
# Energy changes of the processes: the columns between the energy and the rate
N_ENERGY_CHANGES = len(ENERGY_BUDGET_NAMES) - 2


def energy_changes_of_step(
    dynamics: SphereShallowWaterDynamics,
    energy_before: TensorLike,
    undamped_state: TensorLike,
    damped_state: TensorLike,
    increments: list[TensorLike | None],
    hyperdiffusion: TensorLike,
    drag: TensorLike | None,
    has_relaxation: bool,
    surface_geopotential: TensorLike | None,
) -> tuple[TensorLike, TensorLike]:
    """Energy change of each operation of one solver step.

    The energy is evaluated after the conservative step (undamped state), after the
    hyperdiffusion, after the drag, after the relaxation (damped state) and after each
    forcing increment, added in turn. Each change is the difference across one
    operation.

    Args:
        dynamics: Fixed operators.
        energy_before: Energy at the start of the step, (B,).
        undamped_state: State after the conservative step, (B, 3, L, L).
        damped_state: State after the whole damping, (B, 3, L, L).
        increments: Forcing increments in the order of ``ENERGY_BUDGET_NAMES``
            (SKEBS, SPPT, periodic), each (B, 3, L, L) or None when absent.
        hyperdiffusion: Factor of the hyperdiffusion for one step, (L, L).
        drag: Factor of the drag for one step, (B,), or None without drag.
        has_relaxation: Whether the damping relaxes the geopotential.
        surface_geopotential: Coefficients of g h_s, (B, L, L), or None.

    Returns:
        The changes (deterministic, hyperviscous, drag, relaxation, then one per
        increment), shape (B, ``N_ENERGY_CHANGES``), and the energy at the end of the
        step, (B,).
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

    changes = [
        energy_undamped - energy_before,
        energy_hyperdiffusion - energy_undamped,
        energy_drag - energy_hyperdiffusion,
        energy_damped - energy_drag,
    ]
    state = damped_state
    energy = energy_damped
    for increment in increments:
        if increment is None:
            changes.append(torch.zeros_like(energy))
            continue
        state = state + increment
        energy_after = total_energy(dynamics, state, surface_geopotential)
        changes.append(energy_after - energy)
        energy = energy_after
    return torch.stack(changes, dim=1), energy


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
    skebs: SkebsForcing | None = None,
    sppt: SpptSettings | None = None,
    sppt_generators: list[torch.Generator] | None = None,
    periodic: PeriodicForcingSettings | None = None,
    start_time: float = 0.0,
    return_additional_input_fields: bool = False,
    return_energy_budget: bool = False,
    return_forcing_state: bool = False,
    output_dynamics: SphereShallowWaterDynamics | None = None,
) -> dict:
    """Advance B simulations and save h, u_phi, u_theta at regular times.

    The state is saved at time 0 and after every ``steps_per_save`` steps, ``n_saves``
    times. The first ``skip_nt`` of these snapshots are dropped.

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
        skebs: SKEBS forcing of the batch, with its values and its pattern, built on
            ``dynamics``; None is no SKEBS. The loop advances its pattern.
        sppt: Settings of SPPT; None is no SPPT. Requires drag or relaxation.
        sppt_generators: One random generator per simulation, for SPPT.
        periodic: Settings of The Well's periodic forcing; None is no such forcing.
        start_time: Time of the initial state [s], the phase of the periodic forcing.
        return_additional_input_fields: Return the forcing increments summed over
            each saved interval.
        return_energy_budget: Return the energy budget of each saved interval.
        return_forcing_state: Return the SKEBS pattern at each saved time.
        output_dynamics: Operators of the grid of the saved fields. With a lower band
            limit than ``dynamics`` the state is truncated before it is saved
            (coarse-graining). None saves on the grid of ``dynamics``.

    Returns:
        A dictionary with, n_frames = n_saves + 1 - skip_nt, and the grid that of
        ``output_dynamics``:

        - ``"data"``: the saved fields (B, n_frames, *grid.shape, 3), float32 on the
          CPU;
        - ``"final_state"``: the spectral state at the last time, (B, 3, L, L);
        - ``"first_frame_state"``: the spectral state at the first saved frame,
          (B, 3, L, L), the start of the baseline of the simulator;
        - ``"additional_input_fields"`` (on request): at frame i, the increments of
          h, u_phi, u_theta added by all the forcings between frames i and i + 1,
          zero at the last frame, (B, n_frames, *grid.shape, 3), float32 on the CPU;
        - ``"energy_budget"`` (on request): at frame i, the energy at frame i and the
          terms of ``ENERGY_BUDGET_NAMES`` between frames i and i + 1, zero at the
          last frame, (B, n_frames, len(ENERGY_BUDGET_NAMES)), float64 on the CPU;
        - ``"forcing_state"`` (on request): the wind tendency of the SKEBS pattern
          for a unit energy rate at each frame, eastward and colatitude components,
          (B, n_frames, *grid.shape, 2), float32 on the CPU.
    """
    batch = state.shape[0]
    device = state.device
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

    # --- Forcings, in the order of ENERGY_BUDGET_NAMES; None when absent
    skebs_forcing = skebs
    if return_forcing_state and skebs_forcing is None:
        msg = "return_forcing_state requires skebs"
        raise ValueError(msg)
    sppt_forcing = None
    if sppt is not None:
        if sppt_generators is None:
            msg = "sppt requires sppt_generators"
            raise ValueError(msg)
        if damping_argument is None:
            msg = "sppt requires drag or relaxation, the physics that it perturbs"
            raise ValueError(msg)
        sppt_forcing = SpptForcing(dynamics, sppt, sppt_generators)
    periodic_forcing = None
    if periodic is not None:
        periodic_forcing = PeriodicForcing(dynamics, periodic, batch)
    forcings = [skebs_forcing, sppt_forcing, periodic_forcing]

    def pattern_frame() -> TensorLike:
        """Wind tendency of the SKEBS pattern for a unit rate, on the output grid."""
        assert skebs_forcing is not None  # return_forcing_state requires skebs
        tendency = skebs_forcing.unit_tendency()[..., :output_lmax, :output_lmax]
        return increment_to_fields(output_dynamics, tendency)[..., 1:].float().cpu()

    # --- Time loop
    tendencies_history: list[TensorLike] = []
    first_frame_state = state
    frames = []
    increment_frames = []
    budget_frames = []
    pattern_frames = []
    if skip_nt == 0:
        truncated = state[..., :output_lmax, :output_lmax]
        fields = spectral_to_fields(output_dynamics, truncated, output_surface)
        frames.append(fields.float().cpu())
        if return_forcing_state:
            pattern_frames.append(pattern_frame())
    energy = torch.zeros(batch, dtype=torch.float64, device=device)
    if return_energy_budget:
        energy = total_energy(dynamics, state, surface_geopotential)

    # The state before the damping, when the budget or a forcing needs it
    needs_undamped = return_energy_budget or any(
        f is not None and f.needs_undamped for f in forcings
    )
    undamped_state = state
    step = 0
    for save in range(1, n_saves + 1):
        increments_since_save = torch.zeros_like(state)
        changes_since_save = torch.zeros(
            batch, N_ENERGY_CHANGES, dtype=torch.float64, device=device
        )
        rate_since_save = torch.zeros(batch, dtype=torch.float64, device=device)
        energy_at_save = energy

        for _ in range(steps_per_save):
            time = start_time + step * dt
            step += 1

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
            if needs_undamped:
                if equilibrium is None:
                    undamped_state = damped_state / step_damping
                else:
                    departure = (damped_state - equilibrium) / step_damping
                    undamped_state = equilibrium + departure

            # 2. Forcing increments, added by operator splitting
            solver_step = StepState(
                time, undamped_state, damped_state, hyperdiffusion, drag
            )
            step_increments = [
                None if f is None else f.increment(solver_step) for f in forcings
            ]
            if skebs_forcing is not None:
                rate_since_save += skebs_forcing.mean_rate
            state = damped_state
            for increment in step_increments:
                if increment is not None:
                    state = state + increment
                    increments_since_save += increment

            if return_energy_budget:
                changes, energy = energy_changes_of_step(
                    dynamics,
                    energy,
                    undamped_state,
                    damped_state,
                    step_increments,
                    hyperdiffusion,
                    drag,
                    has_relaxation,
                    surface_geopotential,
                )
                changes_since_save += changes

        if save == skip_nt:
            first_frame_state = state

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
                pattern_frames.append(pattern_frame())

    output = {
        "data": torch.stack(frames, dim=1),
        "final_state": state,
        "first_frame_state": first_frame_state,
    }
    if return_additional_input_fields:
        increment_frames.append(torch.zeros_like(frames[-1]))  # last frame
        output["additional_input_fields"] = torch.stack(increment_frames, dim=1)
    if return_energy_budget:
        last_row = torch.zeros(batch, len(ENERGY_BUDGET_NAMES), dtype=torch.float64)
        last_row[:, 0] = energy.cpu()
        budget_frames.append(last_row)
        output["energy_budget"] = torch.stack(budget_frames, dim=1)
    if return_forcing_state:
        output["forcing_state"] = torch.stack(pattern_frames, dim=1)
    return output
