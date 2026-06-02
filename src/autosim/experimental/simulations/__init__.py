"""Experimental simulator implementations."""

from .compressible_fluid import CompressibleFluid2D
from .gray_scott import GrayScott
from .gross_pitaevskii import GrossPitaevskiiEquation2D
from .hydrodynamics_2d import Hydrodynamics2D
from .lattice_boltzmann import LatticeBoltzmann
from .navier_stokes_conditioned import ConditionedNavierStokes2D
from .reaction_diffusion import ReactionDiffusion
from .shallow_water import ShallowWater2D

ALL_SIMULATORS = [
    CompressibleFluid2D,
    ConditionedNavierStokes2D,
    GrayScott,
    GrossPitaevskiiEquation2D,
    Hydrodynamics2D,
    LatticeBoltzmann,
    ReactionDiffusion,
    ShallowWater2D,
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

SIMULATOR_REGISTRY = {simulator.__name__: simulator for simulator in ALL_SIMULATORS}
