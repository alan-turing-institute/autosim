"""Experimental simulator implementations."""

from .burgers_1d import Burgers1D, simulate_burgers_1d
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
from .shallow_water_1d import ShallowWater1D, simulate_shallow_water_1d
from .shallow_water_stepper import advance_swe_2d

ALL_SIMULATORS = [
    Burgers1D,
    CompressibleFluid2D,
    Hydrodynamics2D,
    LatticeBoltzmann,
    ShallowWater2D,
    ShallowWater1D,
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
    "Burgers1D",
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
    "ShallowWater1D",
    "ShallowWater2D",
    "advance_swe_2d",
    "simulate_burgers_1d",
    "simulate_shallow_water_1d",
]

SIMULATOR_REGISTRY = {simulator.__name__: simulator for simulator in ALL_SIMULATORS}
