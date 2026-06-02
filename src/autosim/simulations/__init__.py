"""Stable simulator implementations."""

from .advection_diffusion import AdvectionDiffusion
from .advection_diffusion_multichannel import AdvectionDiffusionMultichannel
from .epidemic import Epidemic
from .flow_problem import FlowProblem
from .projectile import Projectile, ProjectileMultioutput
from .seir import SEIRSimulator
from .spatiotemporal import (
    ConditionedNavierStokes2D,
    GrayScott,
    GrossPitaevskiiEquation2D,
)

ALL_SIMULATORS = [
    AdvectionDiffusion,
    AdvectionDiffusionMultichannel,
    ConditionedNavierStokes2D,
    GrayScott,
    GrossPitaevskiiEquation2D,
    Epidemic,
    SEIRSimulator,
    FlowProblem,
    Projectile,
    ProjectileMultioutput,
]

__all__ = [
    "AdvectionDiffusion",
    "AdvectionDiffusionMultichannel",
    "ConditionedNavierStokes2D",
    "Epidemic",
    "FlowProblem",
    "GrayScott",
    "GrossPitaevskiiEquation2D",
    "Projectile",
    "ProjectileMultioutput",
    "SEIRSimulator",
]

SIMULATOR_REGISTRY = {simulator.__name__: simulator for simulator in ALL_SIMULATORS}
