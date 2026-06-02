from autosim.experimental.simulations import (
    ConditionedNavierStokes2D as ExperimentalConditionedNavierStokes2D,
)
from autosim.experimental.simulations import (
    GrayScott as ExperimentalGrayScott,
)
from autosim.experimental.simulations import (
    GrossPitaevskiiEquation2D as ExperimentalGrossPitaevskiiEquation2D,
)
from autosim.experimental.simulations.gray_scott import (
    GrayScott as ExperimentalGrayScottModule,
)
from autosim.experimental.simulations.gross_pitaevskii import (
    GrossPitaevskiiEquation2D as ExperimentalGrossPitaevskiiEquation2DModule,
)
from autosim.experimental.simulations.navier_stokes_conditioned import (
    ConditionedNavierStokes2D as ExperimentalConditionedNavierStokes2DModule,
)
from autosim.simulations import (
    ConditionedNavierStokes2D,
    GrayScott,
    GrossPitaevskiiEquation2D,
)
from autosim.simulations.spatiotemporal import (
    ConditionedNavierStokes2D as SpatioTemporalConditionedNavierStokes2D,
)
from autosim.simulations.spatiotemporal import (
    GrayScott as SpatioTemporalGrayScott,
)
from autosim.simulations.spatiotemporal import (
    GrossPitaevskiiEquation2D as SpatioTemporalGrossPitaevskiiEquation2D,
)


def test_promoted_simulators_export_from_stable_paths() -> None:
    assert ConditionedNavierStokes2D is SpatioTemporalConditionedNavierStokes2D
    assert GrayScott is SpatioTemporalGrayScott
    assert GrossPitaevskiiEquation2D is SpatioTemporalGrossPitaevskiiEquation2D


def test_promoted_experimental_imports_remain_compatible() -> None:
    assert ExperimentalConditionedNavierStokes2D is ConditionedNavierStokes2D
    assert ExperimentalConditionedNavierStokes2DModule is ConditionedNavierStokes2D
    assert ExperimentalGrayScott is GrayScott
    assert ExperimentalGrayScottModule is GrayScott
    assert ExperimentalGrossPitaevskiiEquation2D is GrossPitaevskiiEquation2D
    assert ExperimentalGrossPitaevskiiEquation2DModule is GrossPitaevskiiEquation2D
