from autosim.simulations import (
    AdvectionDiffusionMultichannel,
    ConditionedNavierStokes2D,
    GrayScott,
    GrossPitaevskiiEquation2D,
    ReactionDiffusion,
)
from autosim.simulations.spatiotemporal import (
    AdvectionDiffusionMultichannel as SpatioTemporalAdvectionDiffusionMultichannel,
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
from autosim.simulations.spatiotemporal import (
    ReactionDiffusion as SpatioTemporalReactionDiffusion,
)


def test_promoted_simulators_export_from_stable_paths() -> None:
    assert (
        AdvectionDiffusionMultichannel is SpatioTemporalAdvectionDiffusionMultichannel
    )
    assert ConditionedNavierStokes2D is SpatioTemporalConditionedNavierStokes2D
    assert GrayScott is SpatioTemporalGrayScott
    assert GrossPitaevskiiEquation2D is SpatioTemporalGrossPitaevskiiEquation2D
    assert ReactionDiffusion is SpatioTemporalReactionDiffusion
