from .advection_diffusion import AdvectionDiffusion
from .advection_diffusion_multichannel import AdvectionDiffusionMultichannel
from .epidemic import Epidemic
from .flow_problem import FlowProblem
from .projectile import Projectile, ProjectileMultioutput
from .reaction_diffusion import ReactionDiffusion
from .seir import SEIRSimulator
from .gray_scott import GrayScott

ALL_SIMULATORS = [
    ReactionDiffusion,
    AdvectionDiffusion,
    AdvectionDiffusionMultichannel,
    Epidemic,
    SEIRSimulator,
    FlowProblem,
    Projectile,
    ProjectileMultioutput,
    GrayScott,
]

__all__ = [
    "AdvectionDiffusion",
    "AdvectionDiffusionMultichannel",
    "Epidemic",
    "FlowProblem",
    "Projectile",
    "ProjectileMultioutput",
    "ReactionDiffusion",
    "SEIRSimulator",
    "GrayScott",
]

SIMULATOR_REGISTRY = dict(zip(__all__, ALL_SIMULATORS, strict=False))
