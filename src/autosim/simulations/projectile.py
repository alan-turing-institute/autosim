"""Projectile motion simulators with drag."""

# from https://github.com/alan-turing-institute/mogp-emulator/blob/main/mogp_emulator/demos/projectile.py
import numpy as np
import torch
from scipy.integrate import solve_ivp

from autosim.simulations.base import Simulator
from autosim.types import NumpyLike, TensorLike


class Projectile(Simulator):
    """Simulator of projectile motion.

    A projectile is launched from an initial height of 2 meters at an  angle of 45
    degrees and falls under the influence of gravity and air resistance. Drag is
    proportional to the square of the velocity. We would like to determine the distance
    travelled by the projectile as a function of the drag coefficient and the launch
    velocity.
    """

    def __init__(
        self,
        parameters_range: dict[str, tuple[float, float]] | None = None,
        output_names: list[str] | None = None,
        log_level: str = "progress_bar",
    ):
        """Initialize the single-output projectile simulator."""
        if parameters_range is None:
            parameters_range = {"c": (-5.0, 1.0), "v0": (0.0, 1000)}
        if output_names is None:
            output_names = ["distance"]
        super().__init__(parameters_range, output_names, log_level)

    def _forward(self, x: TensorLike) -> TensorLike:
        """Simulate the projectile motion and return the distance travelled.

        Args:
            x: Dictionary of input parameter values to simulate:
                - `c`: the drag coefficient on a log scale
                - `v0`: velocity

        Returns:
            Distance travelled by projectile.
        """
        assert x.shape[0] == 1, (
            f"Simulator._forward expects a single input, got {x.shape[0]}"
        )
        y = simulate_projectile(x.cpu().numpy()[0])
        return torch.tensor([y]).view(-1, 1)


class ProjectileMultioutput(Simulator):
    """Multi-output simulator of projectile motion.

    Simulator of projectile motion that outputs both the distance travelled by the
    projectile and its velocity on impact.
    """

    def __init__(
        self,
        parameters_range: dict[str, tuple[float, float]] | None = None,
        output_names: list[str] | None = None,
        log_level: str = "progress_bar",
    ):
        """Initialize the multi-output projectile simulator."""
        if parameters_range is None:
            parameters_range = {"c": (-5.0, 1.0), "v0": (0.0, 1000)}
        if output_names is None:
            output_names = ["distance", "impact_velocity"]
        super().__init__(parameters_range, output_names, log_level)

    def _forward(self, x: TensorLike) -> TensorLike:
        """Simulate the projectile motion with multiple outputs.

        Simulate projectile motion and return the distance travelled and impact
        velocity.

        Args:
            x: Dictionary of input parameter values to simulate:
                - `c`: the drag coefficient on a log scale
                - `v0`: velocity

        Returns:
            Distance travelled by projectile and impact velocity.
        """
        assert x.shape[0] == 1, (
            f"Simulator._forward expects a single input, got {x.shape[0]}"
        )
        y = simulate_projectile_multioutput(x.cpu().numpy()[0])
        return torch.tensor([y])


def f(t: float, y: NumpyLike, c: float):  # noqa: ARG001
    r"""Compute RHS of system of differential equations, returning vector derivative.

    The state is ``(v_x, v_y, x, y)`` and the drag model is:

    .. math::

        \frac{dv_x}{dt} &= -c v_x \sqrt{v_x^2 + v_y^2} \\
        \frac{dv_y}{dt} &= -g - c v_y \sqrt{v_x^2 + v_y^2}

    Args:
        t: Time variable (not used).
        y: Array of dependent variables (vx, vy, x, y).
        c: Drag coefficient (non-negative).
    """
    # check inputs and extract
    assert len(y) == 4
    assert c >= 0.0

    vx = y[0]
    vy = y[1]

    # calculate derivatives
    dydt = np.zeros(4)

    dydt[0] = -c * vx * np.sqrt(vx**2 + vy**2)
    dydt[1] = -9.8 - c * vy * np.sqrt(vx**2 + vy**2)
    dydt[2] = vx
    dydt[3] = vy

    return dydt


def event(t: float, y: NumpyLike, c: float) -> float:  # noqa: ARG001
    """Event to trigger end of integration. Stops when projectile hits ground.

    Args:
        t: Time variable (not used).
        y: Array of dependent variables (vx, vy, x, y).
        c: Drag coefficient (non-negative).

    Returns:
        The height of the projectile.
    """
    assert len(y) == 4
    assert c >= 0.0

    return y[3]


# indicate this event terminates the simulation
event.terminal = True  # pyright: ignore[reportFunctionMemberAccess]


def simulator_base(x: NumpyLike):
    """Simulate ODE system for projectile motion with drag.

    Returns distance projectile travels.

    Args:
        x: Array of input parameters (c, v0).

    Returns:
        Results of ODE integration.
    """
    # unpack values

    assert len(x) == 2
    assert x[1] > 0.0

    c = 10.0 ** x[0]
    v0 = x[1]

    # set initial conditions

    y0 = np.zeros(4)

    y0[0] = v0 / np.sqrt(2.0)
    y0[1] = v0 / np.sqrt(2.0)
    y0[3] = 2.0

    # run simulation
    return solve_ivp(f, (0.0, 1.0e8), y0, events=event, args=(c,))


def simulate_projectile(x: NumpyLike) -> float:
    """Return the distance travelled by the projectile.

    Distance is obtained by solving the ODE system for projectile motion with drag.

    Args:
        x: Array of input parameters (c, v0).

    Returns:
        Distance travelled by projectile.
    """
    results = simulator_base(x)

    return results.y_events[0][0][2]


def simulate_projectile_multioutput(x: NumpyLike) -> tuple[float, float]:
    """Return the distance travelled by the projectile and its impact velocity.

    Simulator to solve ODE system with multiple outputs.

    Args:
        x: Array of input parameters (c, v0).

    Returns:
        float, float
            Distance travelled by projectile and its velocity on impact.
    """
    results = simulator_base(x)

    return (
        results.y_events[0][0][2],
        np.sqrt(results.y_events[0][0][0] ** 2 + results.y_events[0][0][1] ** 2),
    )
