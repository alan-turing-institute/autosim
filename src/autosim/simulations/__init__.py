"""Stable simulator implementations."""

from .advection_diffusion import AdvectionDiffusion
from .epidemic import Epidemic
from .flow_problem import FlowProblem
from .projectile import Projectile, ProjectileMultioutput
from .seir import SEIRSimulator
from .spatiotemporal import (
    AdvectionDiffusionMultichannel,
    ConditionedNavierStokes2D,
    GrayScott,
    GrossPitaevskiiEquation2D,
    ReactionDiffusion,
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
    ReactionDiffusion,
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
    "ReactionDiffusion",
    "SEIRSimulator",
]

SIMULATOR_REGISTRY = {simulator.__name__: simulator for simulator in ALL_SIMULATORS}
