from .advection_diffusion import AdvectionDiffusion
from .advection_diffusion_multichannel import AdvectionDiffusionMultichannel
from .compressible_fluid import CompressibleFluid1D
from .epidemic import Epidemic
from .flow_problem import FlowProblem
from .gray_scott import GrayScott
from .hydrodynamics_2d import Hydrodynamics2D
from .projectile import Projectile, ProjectileMultioutput
from .reaction_diffusion import ReactionDiffusion
from .seir import SEIRSimulator

ALL_SIMULATORS = [
    ReactionDiffusion,
    AdvectionDiffusion,
    AdvectionDiffusionMultichannel,
    CompressibleFluid1D,
    Hydrodynamics2D,
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
    "CompressibleFluid1D",
    "Epidemic",
    "FlowProblem",
    "GrayScott",
    "Hydrodynamics2D",
    "Projectile",
    "ProjectileMultioutput",
    "ReactionDiffusion",
    "SEIRSimulator",
]

SIMULATOR_REGISTRY = dict(zip(__all__, ALL_SIMULATORS, strict=False))
