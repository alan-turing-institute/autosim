from .advection_diffusion import AdvectionDiffusion
from .advection_diffusion_multichannel import AdvectionDiffusionMultichannel
from .advection_diffusion_non_periodic import AdvectionDiffusionNonPeriodic
from .epidemic import Epidemic
from .flow_problem import FlowProblem
from .projectile import Projectile, ProjectileMultioutput
from .reaction_diffusion_non_periodic import ReactionDiffusionNonPeriodic
from .seir import SEIRSimulator

ALL_SIMULATORS = [
    AdvectionDiffusion,
    AdvectionDiffusionMultichannel,
    AdvectionDiffusionNonPeriodic,
    Epidemic,
    SEIRSimulator,
    FlowProblem,
    Projectile,
    ProjectileMultioutput,
    ReactionDiffusionNonPeriodic,
]

__all__ = [
    "AdvectionDiffusion",
    "AdvectionDiffusionMultichannel",
    "AdvectionDiffusionNonPeriodic",
    "Epidemic",
    "FlowProblem",
    "Projectile",
    "ProjectileMultioutput",
    "ReactionDiffusionNonPeriodic",
    "SEIRSimulator",
]

SIMULATOR_REGISTRY = dict(zip(__all__, ALL_SIMULATORS, strict=False))
