from .advection_diffusion import AdvectionDiffusion
from .advection_diffusion_multichannel import AdvectionDiffusionMultichannel
from .epidemic import Epidemic
from .flow_problem import FlowProblem
from .kolmogorov_flow import KolmogorovFlow
from .projectile import Projectile, ProjectileMultioutput
from .seir import SEIRSimulator

ALL_SIMULATORS = [
    AdvectionDiffusion,
    AdvectionDiffusionMultichannel,
    Epidemic,
    SEIRSimulator,
    FlowProblem,
    KolmogorovFlow,
    Projectile,
    ProjectileMultioutput,
]

__all__ = [
    "AdvectionDiffusion",
    "AdvectionDiffusionMultichannel",
    "Epidemic",
    "FlowProblem",
    "KolmogorovFlow",
    "Projectile",
    "ProjectileMultioutput",
    "SEIRSimulator",
]

SIMULATOR_REGISTRY = dict(zip(__all__, ALL_SIMULATORS, strict=False))
