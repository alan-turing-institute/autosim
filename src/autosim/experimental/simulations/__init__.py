"""Experimental simulator implementations."""

from .compressible_fluid import CompressibleFluid2D
from .hydrodynamics_2d import Hydrodynamics2D
from .lattice_boltzmann import LatticeBoltzmann
from .shallow_water import ShallowWater2D

ALL_SIMULATORS = [
    CompressibleFluid2D,
    Hydrodynamics2D,
    LatticeBoltzmann,
    ShallowWater2D,
]

__all__ = [
    "CompressibleFluid2D",
    "Hydrodynamics2D",
    "LatticeBoltzmann",
    "ShallowWater2D",
]

SIMULATOR_REGISTRY = {simulator.__name__: simulator for simulator in ALL_SIMULATORS}
