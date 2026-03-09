from .compressible_fluid import CompressibleFluid2D
from .gray_scott import GrayScott
from .gross_pitaevskii import GrossPitaevskiiEquation2D
from .hydrodynamics_2d import Hydrodynamics2D
from .lattice_boltzmann import LatticeBoltzmann
from .navier_stokes_conditioned import ConditionedNavierStokes2D
from .reaction_diffusion import ReactionDiffusion
from .shallow_water import ShallowWater2D

ALL_SIMULATORS = [
    ReactionDiffusion,
    CompressibleFluid2D,
    Hydrodynamics2D,
    LatticeBoltzmann,
    GrayScott,
    ConditionedNavierStokes2D,
]

__all__ = [
    "CompressibleFluid2D",
    "ConditionedNavierStokes2D",
    "GrayScott",
    "GrossPitaevskiiEquation2D",
    "Hydrodynamics2D",
    "LatticeBoltzmann",
    "ReactionDiffusion",
    "ShallowWater2D",
]

SIMULATOR_REGISTRY = dict(zip(__all__, ALL_SIMULATORS, strict=False))
