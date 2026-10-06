"""Dataset adapters shared by deterministic one-dimensional PDE simulators."""

from __future__ import annotations

import math
from abc import abstractmethod

import torch
from einops import rearrange, repeat

from autosim.simulations.base import SpatioTemporalSimulator


def save_times(T: float, dt_save: float) -> torch.Tensor:
    """Include the initial state and the terminal time, even off the save grid."""
    if not math.isfinite(T) or T < 0:
        msg = "T must be finite and non-negative"
        raise ValueError(msg)
    if not math.isfinite(dt_save) or dt_save <= 0:
        msg = "dt_save must be finite and positive"
        raise ValueError(msg)
    times = torch.arange(math.floor(T / dt_save) + 1, dtype=torch.float64) * dt_save
    if float(times[-1]) == T or (
        len(times) > 1 and math.isclose(float(times[-1]), T, rel_tol=0, abs_tol=1e-12)
    ):
        times[-1] = T
    else:
        times = torch.cat((times, torch.tensor([T], dtype=torch.float64)))
    return times


def check_state(state: torch.Tensor, channels: int) -> None:
    """Validate a physical field shaped ``(..., nx, channels)``."""
    if state.ndim < 2 or state.shape[-1] != channels or state.shape[-2] < 8:
        msg = f"state must have shape (..., nx, {channels}) with nx >= 8"
        raise ValueError(msg)
    if state.dtype not in (torch.float32, torch.float64):
        msg = "state must use float32 or float64"
        raise TypeError(msg)
    if not bool(torch.isfinite(state).all()) or state.numel() == 0:
        msg = "state must be nonempty and finite"
        raise ValueError(msg)


def check_solver_settings(L: float, cfl: float, dt_max: float) -> None:
    """Validate shared physical geometry and internal timestep controls."""
    if not all(math.isfinite(value) and value > 0 for value in (L, cfl, dt_max)):
        msg = "L, cfl and dt_max must be finite and positive"
        raise ValueError(msg)
    if cfl > 0.5:
        msg = "cfl must be at most 0.5"
        raise ValueError(msg)


def generator(random_seed: int | None) -> torch.Generator:
    """Create a local sampling generator without changing the global RNG."""
    rng = torch.Generator()
    if random_seed is None:
        rng.seed()
    else:
        rng.manual_seed(random_seed)
    return rng


def check_range(bounds: tuple[float, float], name: str, positive: bool = False) -> None:
    """Validate finite, ordered sampling bounds."""
    if len(bounds) != 2 or not all(math.isfinite(value) for value in bounds):
        msg = f"{name} must contain two finite bounds"
        raise ValueError(msg)
    if bounds[0] > bounds[1] or (positive and bounds[0] <= 0):
        msg = f"{name} must have ordered bounds" + (
            " greater than zero" if positive else ""
        )
        raise ValueError(msg)


class Deterministic1DSimulator(SpatioTemporalSimulator):
    """Keep field sampling separate from scalar-parameter legacy forwarding.

    ``forward`` evolves a supplied reference field, or the sampler's seed-zero
    field. Dataset generation samples fresh initial fields explicitly and saves
    them as the first frame. Neither evolution path consumes random numbers.
    """

    def __init__(
        self,
        *,
        parameter_name: str,
        parameter_value: float,
        parameters_range: dict[str, tuple[float, float]] | None,
        output_names: list[str],
        nx: int,
        L: float,
        T: float,
        dt_save: float,
        return_timeseries: bool,
        initial_state: torch.Tensor | None,
        grid_offset: float,
        log_level: str,
    ) -> None:
        if not isinstance(nx, int) or nx < 8:
            msg = "nx must be an integer at least 8"
            raise ValueError(msg)
        if not math.isfinite(L) or L <= 0:
            msg = "L must be finite and positive"
            raise ValueError(msg)
        if not math.isfinite(parameter_value) or parameter_value <= 0:
            msg = f"{parameter_name} must be finite and positive"
            raise ValueError(msg)
        bounds = (
            parameters_range
            if parameters_range is not None
            else {parameter_name: (parameter_value, parameter_value)}
        )
        if set(bounds) != {parameter_name}:
            msg = f"parameters_range must contain only {parameter_name}"
            raise ValueError(msg)
        check_range(bounds[parameter_name], parameter_name, positive=True)
        super().__init__(bounds, output_names, log_level)
        self.parameter_name = parameter_name
        self.nx, self.L, self.T, self.dt_save = nx, L, T, dt_save
        self.return_timeseries = return_timeseries
        self.times = save_times(T, dt_save)
        self.x = (torch.arange(nx, dtype=torch.float64) + grid_offset) * (L / nx)
        self.initial_state = None
        if initial_state is not None:
            field = torch.as_tensor(initial_state, dtype=torch.float64).clone()
            check_state(field, len(output_names))
            if field.shape != (nx, len(output_names)):
                msg = "initial_state must have shape (nx, channels)"
                raise ValueError(msg)
            self.initial_state = field

    def _fixed_initial_states(self, n: int) -> torch.Tensor | None:
        if not isinstance(n, int) or n < 0:
            msg = "n must be a non-negative integer"
            raise ValueError(msg)
        if self.initial_state is None:
            return None
        return repeat(self.initial_state, "x c -> b x c", b=n).clone()

    def _resolve_parameter(self, value: float | None) -> float:
        if value is not None:
            return value
        lower, upper = self.parameters_range[self.parameter_name]
        if lower != upper:
            msg = (
                f"Supply {self.parameter_name} explicitly when parameters_range varies"
            )
            raise ValueError(msg)
        return lower

    @abstractmethod
    def sample_initial_conditions(
        self, n: int, random_seed: int | None = None
    ) -> torch.Tensor:
        """Sample physical fields shaped ``(batch, nx, channels)``."""

    @abstractmethod
    def _rollout(self, state: torch.Tensor, parameter: float) -> torch.Tensor:
        """Evolve a supplied field, retaining the initial frame."""

    def _prepare_initial_conditions(
        self,
        states: torch.Tensor,
        parameters: torch.Tensor,  # noqa: ARG002
    ) -> tuple[torch.Tensor, dict]:
        """Prepare dataset forecast states and optional sampling metadata."""
        return states, {}

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evolve the deterministic reference field for one physical parameter."""
        if x.shape != (1, 1):
            msg = "forward expects one scalar parameter with shape (1, 1)"
            raise ValueError(msg)
        state = self.sample_initial_conditions(1, random_seed=0)[0]
        trajectory = self._rollout(state, float(x[0, 0]))
        if not self.return_timeseries:
            trajectory = trajectory[-1:]
        return rearrange(trajectory.float(), "t x c -> 1 (t x c)")

    def forward_samples_spatiotemporal(
        self,
        n: int,
        random_seed: int | None = None,
        ensure_exact_n: bool = False,  # noqa: ARG002 -- return all n or fail explicitly
    ) -> dict:
        """Sample fields and parameters; return trajectories and coordinates.

        The first frame is the actual initial condition when
        ``return_timeseries=True``. Coordinates carry a batch axis so stratified
        datasets can concatenate them using the existing payload machinery.
        Invalid evolutions raise rather than silently changing the distribution.
        """
        states = self.sample_initial_conditions(n, random_seed)
        # SciPy's variable-range scaler cannot reduce an empty sample array.
        parameters = (
            self.sample_inputs(n, random_seed)
            if n
            else torch.empty((0, 1), dtype=torch.float32)
        )
        times = self.times if self.return_timeseries else self.times[-1:]
        with torch.no_grad():
            states, metadata = self._prepare_initial_conditions(states, parameters)
            trajectories = [
                self._rollout(state, float(parameter[0]))
                for state, parameter in zip(states, parameters, strict=True)
            ]
        if n:
            data = torch.stack(trajectories)
            if not self.return_timeseries:
                data = data[:, -1:]
        else:
            data = torch.empty((0, len(times), self.nx, len(self.output_names)))
        return {
            "data": rearrange(data.float(), "b t x c -> b t x 1 c"),
            "constant_scalars": parameters,
            "constant_fields": None,
            "x": repeat(self.x, "x -> b x", b=n),
            "times": repeat(times, "t -> b t", b=n),
            **metadata,
        }
