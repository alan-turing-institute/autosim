"""Experimental simulator implementations."""

from .compressible_fluid import CompressibleFluid2D
from .correlated_diffusion_1d import CorrelatedDiffusion1D
from .correlated_diffusion_2d import CorrelatedDiffusion2D
from .cox_ingersoll_ross import CoxIngersollRoss
from .double_well import DoubleWell
from .gray_scott_stochastic import GrayScottStochastic
from .hydrodynamics_2d import Hydrodynamics2D
from .lattice_boltzmann import LatticeBoltzmann
from .lorenz96 import Lorenz96
from .lorenz96_correlated import Lorenz96Correlated
from .multivariate_ou_lens import MultivariateOULens
from .ornstein_uhlenbeck import OrnsteinUhlenbeck
from .shallow_water import ShallowWater2D

ALL_SIMULATORS = [
    CompressibleFluid2D,
    Hydrodynamics2D,
    LatticeBoltzmann,
    ShallowWater2D,
    OrnsteinUhlenbeck,
    CoxIngersollRoss,
    DoubleWell,
    Lorenz96,
    Lorenz96Correlated,
    CorrelatedDiffusion1D,
    CorrelatedDiffusion2D,
    MultivariateOULens,
    GrayScottStochastic,
]

__all__ = [
    "CompressibleFluid2D",
    "CorrelatedDiffusion1D",
    "CorrelatedDiffusion2D",
    "CoxIngersollRoss",
    "DoubleWell",
    "GrayScottStochastic",
    "Hydrodynamics2D",
    "LatticeBoltzmann",
    "Lorenz96",
    "Lorenz96Correlated",
    "MultivariateOULens",
    "OrnsteinUhlenbeck",
    "ShallowWater2D",
]

SIMULATOR_REGISTRY = {simulator.__name__: simulator for simulator in ALL_SIMULATORS}
