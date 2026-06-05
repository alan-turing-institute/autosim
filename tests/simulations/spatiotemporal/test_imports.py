from autosim.simulations import (
    AdvectionDiffusion,
    AdvectionDiffusionMultichannel,
    ConditionedNavierStokes2D,
    GrayScott,
    GrossPitaevskiiEquation2D,
    ReactionDiffusion,
    spatiotemporal,
)


def test_promoted_simulators_export_from_stable_paths() -> None:
    assert AdvectionDiffusion is spatiotemporal.AdvectionDiffusion
    assert (
        AdvectionDiffusionMultichannel is spatiotemporal.AdvectionDiffusionMultichannel
    )
    assert ConditionedNavierStokes2D is spatiotemporal.ConditionedNavierStokes2D
    assert GrayScott is spatiotemporal.GrayScott
    assert GrossPitaevskiiEquation2D is spatiotemporal.GrossPitaevskiiEquation2D
    assert ReactionDiffusion is spatiotemporal.ReactionDiffusion
