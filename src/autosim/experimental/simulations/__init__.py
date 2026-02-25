from .compressible_fluid import CompressibleFluid2D
from .hydrodynamics_2d import Hydrodynamics2D
from .lattice_boltzmann import LatticeBoltzmann

ALL_SIMULATORS = [
    CompressibleFluid2D,
    Hydrodynamics2D,
    LatticeBoltzmann,
]

__all__ = [
    "CompressibleFluid2D",
    "Hydrodynamics2D",
    "LatticeBoltzmann",
]

SIMULATOR_REGISTRY = dict(zip(__all__, ALL_SIMULATORS, strict=False))
