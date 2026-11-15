"""Time integration of a batch of shallow water simulations on the sphere."""

from __future__ import annotations

import torch

from autosim.types import TensorLike

from .dynamics import (
    SphereShallowWaterDynamics,
    adams_bashforth_step,
    spectral_to_fields,
)


def integrate_trajectories(
    dynamics: SphereShallowWaterDynamics,
    state: TensorLike,
    n_saves: int,
    steps_per_save: int,
    skip_nt: int,
    coriolis: TensorLike | None = None,
    surface_geopotential: TensorLike | None = None,
) -> dict[str, TensorLike]:
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

    Returns:
        A dictionary with

        - ``"data"``: the saved fields (B, n_saves + 1 - skip_nt, *grid.shape, 3),
          float32 on the CPU;
        - ``"final_state"``: the spectral state at the last time, (B, 3, L, L).
    """
    frames = []
    if skip_nt == 0:
        fields = spectral_to_fields(dynamics, state, surface_geopotential)
        frames.append(fields.float().cpu())

    previous_tendencies: list[TensorLike] = []
    for save in range(1, n_saves + 1):
        for _ in range(steps_per_save):
            state, previous_tendencies = adams_bashforth_step(
                dynamics, state, previous_tendencies, coriolis, surface_geopotential
            )
        if save >= skip_nt:
            fields = spectral_to_fields(dynamics, state, surface_geopotential)
            frames.append(fields.float().cpu())

    data = torch.stack(frames, dim=1)  # (B, n_frames, *grid.shape, 3)
    return {"data": data, "final_state": state}
