"""Stable simulator implementations."""

from .epidemic import Epidemic
from .flow_problem import FlowProblem
from .projectile import Projectile, ProjectileMultioutput
from .seir import SEIRSimulator
from .spatiotemporal import (
    AdvectionDiffusion,
    AdvectionDiffusionMultichannel,
    ConditionedNavierStokes2D,
    GrayScott,
    GrossPitaevskiiEquation2D,
    ReactionDiffusion,
)
from .synthetic import GaussianField

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
    GaussianField,
]

__all__ = [
    "AdvectionDiffusion",
    "AdvectionDiffusionMultichannel",
    "ConditionedNavierStokes2D",
    "Epidemic",
    "FlowProblem",
    "GaussianField",
    "GrayScott",
    "GrossPitaevskiiEquation2D",
    "Projectile",
    "ProjectileMultioutput",
    "ReactionDiffusion",
    "SEIRSimulator",
]

SIMULATOR_REGISTRY = {simulator.__name__: simulator for simulator in ALL_SIMULATORS}
