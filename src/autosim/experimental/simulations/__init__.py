from .compressible_fluid import CompressibleFluid2D
from .correlated_diffusion_1d import CorrelatedDiffusion1D
from .correlated_diffusion_2d import CorrelatedDiffusion2D
from .cox_ingersoll_ross import CoxIngersollRoss
from .double_well import DoubleWell
from .gray_scott import GrayScott
from .gray_scott_stochastic import GrayScottStochastic
from .gross_pitaevskii import GrossPitaevskiiEquation2D
from .hydrodynamics_2d import Hydrodynamics2D
from .lattice_boltzmann import LatticeBoltzmann
from .lorenz96 import Lorenz96
from .lorenz96_correlated import Lorenz96Correlated
from .multivariate_ou_lens import MultivariateOULens
from .navier_stokes_conditioned import ConditionedNavierStokes2D
from .ornstein_uhlenbeck import OrnsteinUhlenbeck
from .reaction_diffusion import ReactionDiffusion
from .shallow_water import ShallowWater2D

ALL_SIMULATORS = [
    ReactionDiffusion,
    CompressibleFluid2D,
    Hydrodynamics2D,
    LatticeBoltzmann,
    GrayScott,
    ConditionedNavierStokes2D,
    OrnsteinUhlenbeck,
    DoubleWell,
    Lorenz96,
    GrayScottStochastic,
    MultivariateOULens,
    CorrelatedDiffusion1D,
    CorrelatedDiffusion2D,
    Lorenz96Correlated,
    CoxIngersollRoss,
]

__all__ = [
    "CompressibleFluid2D",
    "ConditionedNavierStokes2D",
    "CorrelatedDiffusion1D",
    "CorrelatedDiffusion2D",
    "CoxIngersollRoss",
    "DoubleWell",
    "GrayScott",
    "GrayScottStochastic",
    "GrossPitaevskiiEquation2D",
    "Hydrodynamics2D",
    "LatticeBoltzmann",
    "Lorenz96",
    "Lorenz96Correlated",
    "MultivariateOULens",
    "OrnsteinUhlenbeck",
    "ReactionDiffusion",
    "ShallowWater2D",
]

SIMULATOR_REGISTRY = dict(zip(__all__, ALL_SIMULATORS, strict=False))
